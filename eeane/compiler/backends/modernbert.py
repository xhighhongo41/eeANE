"""ModernBERT compile backend (v0.6実装計画.md §4.2, ported from poc/).

This module is the eeANE-side home of the conversion logic proven by the
PoC scripts (``poc/convert_common.py``, ``poc/convert_embedding.py``,
``poc/convert_reranker.py`` and ``poc/common.py``). The PoC tree is frozen
as a historical record, so the code here -- not ``poc/`` -- is from now on
the single source of truth for the ModernBERT patches, the wrappers, the
fixed trace/sanity fixtures and the FP32 reference computations.

The two monkeypatches are mandatory parts of the conversion, not optional
tweaks (v0.6実装計画.md §2-1):

* :func:`patch_rotate_half` makes the model convertible at all under
  coremltools 9.0 + numpy 2.x.
* :func:`patch_eager_attention_rank4` keeps the compiled model loadable on
  the ANE for batch sizes greater than one.

Importing this module pulls in ``torch``/``transformers``; it therefore
requires the ``[compile]`` extra and must never be imported from the
``eeane serve`` code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.models.modernbert import modeling_modernbert

# Model kinds understood by this backend.
KIND_EMBEDDING = "embedding"
KIND_RERANKER = "reranker"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING, KIND_RERANKER)

# Core ML graph output name per kind (embeddings vs raw relevance logits).
OUTPUT_NAMES: dict[str, str] = {KIND_EMBEDDING: "embedding", KIND_RERANKER: "logits"}

# Short Japanese sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "これは変換用のサンプル文です。"

# Fixed sanity-check sentences (short / medium / long) exercising different
# amounts of padding under the same fixed sequence length.
SANITY_TEXTS: list[str] = [
    "検索クエリ: 日本の首都はどこですか。",
    "検索文書: 東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。",
    "トピック: 機械学習モデルを専用のアクセラレータ上で動かすと、消費電力を抑えつつ"
    "高い推論スループットを得られる場合がある。",
]

# Short Japanese (query, document) pair used as the example input for
# torch.jit.trace of a reranker.
TRACE_EXAMPLE_PAIR: tuple[str, str] = (
    "これは変換用のサンプル質問です。",
    "これは変換用のサンプル文書です。",
)

# Fixed sanity-check pairs: relevant / irrelevant / partially related.
SANITY_PAIRS: list[tuple[str, str]] = [
    # Relevant pair
    (
        "日本の首都はどこですか。",
        "東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。",
    ),
    # Irrelevant pair
    (
        "日本の首都はどこですか。",
        "機械学習モデルを専用のアクセラレータ上で動かすと、消費電力を抑えつつ高い推論スループットを得られる場合がある。",
    ),
    # Partially related pair
    (
        "機械学習の推論を高速化する方法",
        "毎朝のコーヒーにはカフェインが含まれており集中力を高めてくれる。",
    ),
]

# Indices into SANITY_PAIRS used by the reranker ordering check.
RELEVANT_PAIR_INDEX = 0
IRRELEVANT_PAIR_INDEX = 1

# Filler rows used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. The empty strings encode to special tokens
# only, so the row still has a non-empty attention mask (a fully masked row
# would risk NaN, v0.3実装計画.md §4.2).
BATCH_PADDING_TEXT = ""
BATCH_PADDING_PAIR: tuple[str, str] = ("", "")

# Captured at import time so patch_eager_attention_rank4() can delegate the
# non-eager attention paths to the untouched upstream implementation, and so
# that applying the patch twice cannot recurse.
_ORIGINAL_ATTENTION_FORWARD = modeling_modernbert.ModernBertAttention.forward


def patch_rotate_half() -> None:
    """Replace ModernBert's ``rotate_half`` with a static-shape equivalent.

    The upstream implementation slices with ``x.shape[-1] // 2``, which
    traces to ``aten::size -> floor_divide -> aten::Int``. coremltools 9.0
    cannot convert that ``aten::Int`` under numpy 2.x and raises
    "only 0-dimensional arrays can be converted to Python scalars"
    (v0.1実装計画.md §4.8 C1). ``torch.chunk`` with a constant chunk count
    yields the identical result without any dynamic shape arithmetic; this
    is exact because RoPE head dimensions are always even (enforced by
    :meth:`ModernBertBackend.apply_patches`).
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    modeling_modernbert.rotate_half = rotate_half


def patch_eager_attention_rank4() -> None:
    """Rewrite ModernBert's eager attention without rank-5 intermediates.

    Upstream splits the fused projection with
    ``qkv.view(bs, -1, 3, heads, head_dim).transpose(3, 1).unbind(dim=2)``,
    which materializes rank-5 tensors. At B=1 the leading batch axis is
    degenerate and the ANE compiler copes, but for B>1 loading the compiled
    model on CPU_AND_NE fails with
    "MILCompilerForANE error: failed to compile ANE model using ANEF.
    Error=_ANECompiler : ANECCompile() FAILED." and the whole model silently
    falls back to CPU (where the ``finfo.min`` mask makes the outputs NaN,
    the v0.1 known limitation).

    Because ``Wqkv`` emits the three projections contiguously along the last
    axis with the 3 as the outermost sub-index, ``chunk(3, dim=-1)``
    followed by a rank-4 ``view``/``transpose`` selects exactly the same
    elements as the upstream rank-5 path. The rewrite is therefore
    bit-exact (verified: max |patched - upstream| = 0.0 on ruri-v3-310m)
    and only changes tensor rank, never semantics.

    Only the ``eager`` attention path is rewritten; ``sdpa`` and
    ``flash_attention_2`` keep the upstream implementation, so the FP32
    baselines used by the sanity checks are unaffected. Re-applying the
    patch is safe: the delegation target is the implementation captured at
    import time, so it can never recurse into itself.

    Raises:
        ValueError: If a patched module's head geometry contradicts the
            ``heads * head_dim == all_head_size`` layout assumption.
    """

    def eager_attention_rank4(
        module: modeling_modernbert.ModernBertAttention,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        local_attention: tuple[int, int],
        bs: int,
        dim: int,
        output_attentions: bool = False,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Eager attention over a rank-3 ``qkv`` of shape (B, S, 3*H*D)."""
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        heads, head_dim = module.num_heads, module.head_dim
        # (B, S, 3*H*D) -> three (B, H, S, D) tensors, all rank 4.
        query, key, value = (
            part.view(bs, -1, heads, head_dim).transpose(1, 2) for part in qkv.chunk(3, dim=-1)
        )
        query, key = modeling_modernbert.apply_rotary_pos_emb(query, key, cos, sin)

        scale = head_dim**-0.5
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
        if local_attention != (-1, -1):
            attention_mask = sliding_window_mask
        attn_weights = attn_weights + attention_mask
        # Upstream upcasts the softmax to fp32 before casting back.
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=module.attention_dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, dim)
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)

    def attention_forward(
        self: modeling_modernbert.ModernBertAttention,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Replacement for ``ModernBertAttention.forward``."""
        if self.config._attn_implementation != "eager":
            return _ORIGINAL_ATTENTION_FORWARD(
                self, hidden_states, output_attentions=output_attentions, **kwargs
            )
        if self.all_head_size != self.num_heads * self.head_dim:
            raise ValueError(
                "unexpected ModernBert head geometry "
                f"({self.num_heads} heads x {self.head_dim} != {self.all_head_size}); "
                "the rank-4 qkv split assumption does not hold"
            )
        # The rank-5 view of the upstream forward is skipped entirely.
        attn_outputs = eager_attention_rank4(
            self,
            qkv=self.Wqkv(hidden_states),
            local_attention=self.local_attention,
            bs=hidden_states.shape[0],
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.out_drop(self.Wo(attn_outputs[0]))
        return (hidden_states,) + attn_outputs[1:]

    modeling_modernbert.ModernBertAttention.forward = attention_forward


def patch_mask_fill_value(model: torch.nn.Module, fill_value: float) -> None:
    """Override ModernBert attention mask generation with a finite fill value.

    ``ModernBertModel._update_attention_mask`` fills masked positions with
    ``torch.finfo(float32).min``, which becomes ``-inf`` once the graph is
    cast to FP16 and can make softmax produce NaN for fully masked rows.
    This monkeypatch reproduces the same masks with a finite fill value.

    Args:
        model: The loaded ModernBertModel instance (the backbone, not a
            SequenceClassification wrapper; see
            :meth:`ModernBertBackend.apply_patches`).
        fill_value: Finite value used for masked positions (e.g. -30000.0).
    """
    local_attention = model.config.local_attention

    def _update(attention_mask: torch.Tensor, output_attentions: bool = False) -> tuple:
        seq_len = attention_mask.shape[-1]
        global_mask = (1.0 - attention_mask[:, None, None, :].float()) * fill_value  # (B,1,1,S)
        rows = torch.arange(seq_len).unsqueeze(0)
        distance = (rows - rows.T).abs()  # (S, S)
        window = (distance <= local_attention // 2)[None, None, :, :]  # (1,1,S,S)
        sliding_window_mask = global_mask.masked_fill(~window, fill_value)  # (B,1,S,S)
        return global_mask, sliding_window_mask

    model._update_attention_mask = _update


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the sequence dimension.

    Single source of truth shared by the Core ML conversion wrapper
    (:class:`EmbeddingWrapper`) and the PyTorch baseline
    (:func:`encode_pytorch`). The formula must stay identical to
    v0.1実装計画.md §4.2; do not change it without updating both consumers.

    Args:
        hidden: Last hidden state, shape (B, S, H).
        attention_mask: Attention mask, shape (B, S).

    Returns:
        Pooled embeddings, shape (B, H).
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, S, 1)
    summed = (hidden * mask).sum(dim=1)  # (B, H)
    count = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
    return summed / count


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable sigmoid.

    Branches on the sign of ``x`` so ``np.exp`` is only ever evaluated on
    non-positive arguments, avoiding overflow for large-magnitude inputs
    (see v0.2実装計画.md §4.4).

    Args:
        x: Input array (raw logits).

    Returns:
        Array of the same shape as ``x``, with values in (0, 1).
    """
    is_positive = x >= 0
    exp_neg_abs = np.exp(-np.abs(x))
    return np.where(is_positive, 1.0 / (1.0 + exp_neg_abs), exp_neg_abs / (1.0 + exp_neg_abs))


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    Args:
        tokenizer: Tokenizer returned by :meth:`ModernBertBackend.load`.
        texts: Input sentences (prefixes, if any, must already be applied
            by the caller).
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(texts), seq_len)`` and dtype ``np.int32``.
    """
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize (query, document) pairs into fixed-shape int32 arrays.

    Delegates to the tokenizer's built-in pair encoding
    (``tokenizer(queries, documents, ...)``) so that the
    ``<s> query </s> <s> document </s>`` template (v0.2実装計画.md §2.2) is
    produced by the tokenizer's post_processor rather than reimplemented
    here. ``truncation=True`` uses the tokenizer's default
    ``longest_first`` strategy across both sequences. Any key other than
    ``input_ids``/``attention_mask`` returned by the tokenizer (e.g.
    ``token_type_ids``) is discarded, since the reranker forward only
    accepts those two inputs.

    Args:
        tokenizer: Tokenizer returned by :meth:`ModernBertBackend.load`.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for padding/truncation.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(pairs), seq_len)`` and dtype ``np.int32``.
    """
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    encoded = tokenizer(
        queries,
        documents,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def encode_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    seq_len: int,
) -> np.ndarray:
    """Compute FP32 baseline embeddings with a batch-size-1 loop.

    Args:
        model: Embedding model loaded by :meth:`ModernBertBackend.load`.
        tokenizer: Tokenizer for the same model directory.
        texts: Input sentences.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Embeddings array of shape (len(texts), hidden_size), dtype float32.
    """
    batch = tokenize_batch(tokenizer, texts, seq_len)
    hidden_size = model.config.hidden_size
    embeddings = np.empty((len(texts), hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(texts)):
            # nn.Embedding lookup requires int64 indices; tokenize_batch
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs[0]  # (1, S, H)
            pooled = mean_pool(hidden, attention_mask)  # (1, H)
            embeddings[i] = pooled.numpy().astype(np.float32)
    return embeddings


def score_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
) -> np.ndarray:
    """Compute FP32 baseline raw reranker logits with a batch-size-1 loop.

    Args:
        model: Reranker model loaded by :meth:`ModernBertBackend.load`.
        tokenizer: Tokenizer for the same model directory.
        pairs: List of (query, document) pairs.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Raw logits array of shape (len(pairs),), dtype float32.
    """
    batch = tokenize_pairs(tokenizer, pairs, seq_len)
    scores = np.empty(len(pairs), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(pairs)):
            # nn.Embedding lookup requires int64 indices; tokenize_pairs
            # returns int32 for Core ML compatibility, so cast here.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs[0]  # (1, 1)
            scores[i] = logits.reshape(-1)[0].item()
    return scores


class EmbeddingWrapper(torch.nn.Module):
    """Wraps ModernBertModel and performs masked mean pooling in-graph.

    Output matches sentence-transformers (Transformer + mean Pooling,
    no normalization) for ruri-v3-310m.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the backbone model.

        Args:
            model: ModernBertModel loaded in eval/FP32 mode with
                ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute pooled sentence embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Pooled embeddings, shape (B, hidden_size).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs[0]  # (B, S, H)
        return mean_pool(hidden, attention_mask)


class RerankerWrapper(torch.nn.Module):
    """Wraps ModernBertForSequenceClassification and exposes raw logits.

    The Core ML graph reproduces the HF forward as-is (CLS pooling +
    classification head, logits output). Sigmoid is applied outside the
    graph in Python post-processing (see v0.2実装計画.md §2.2).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Store the classification model.

        Args:
            model: ModernBertForSequenceClassification loaded in eval/FP32
                mode with ``config.return_dict = False``.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute raw relevance logits.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S).

        Returns:
            Raw logits, shape (B, 1).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs[0]  # logits (B, 1)


class ModernBertBackend:
    """Compile backend for the ModernBERT architecture family.

    Provisional v0.6 interface (v0.6実装計画.md §4.2): the methods are the
    minimum ``pipeline.py`` needs to convert ModernBERT, deliberately not
    generalised into an abstract base class -- the real multi-architecture
    interface is decided in v0.7. Every method is stateless; the caller
    owns the loaded model and tokenizer.

    Typical call order::

        model, tokenizer = backend.load(model_dir, kind)
        backend.apply_patches(model)
        wrapper = backend.wrap(model, kind)
        example = backend.tokenize(
            tokenizer, kind, [backend.trace_example(kind)] * batch, seq_len
        )
    """

    name = "ModernBert"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(
        self, model_dir: Path, kind: str, attn: str = "eager"
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the FP32 model and its tokenizer from a HF model directory.

        Args:
            model_dir: Local HuggingFace-format model directory (read-only,
                v0.6実装計画.md §2-11).
            kind: ``"embedding"`` (``AutoModel``) or ``"reranker"``
                (``AutoModelForSequenceClassification``).
            attn: Attention implementation to request. ``"eager"`` is the
                path the patches rewrite (conversion); ``"sdpa"`` is used
                for the FP32 reference.

        Returns:
            Tuple of the model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs) and its tokenizer.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        loader = AutoModel if kind == KIND_EMBEDDING else AutoModelForSequenceClassification
        model = loader.from_pretrained(model_dir, attn_implementation=attn, dtype=torch.float32)
        model.config.return_dict = False
        return model.eval(), tokenizer

    def apply_patches(self, model: PreTrainedModel, mask_fill_value: float | None = None) -> None:
        """Apply the mandatory (and optional) ModernBERT graph patches.

        ``patch_rotate_half`` and ``patch_eager_attention_rank4`` are
        mandatory constituents of the conversion, applied at every batch
        size so that one graph shape is produced instead of mixing rank-5
        (B=1) and rank-4 (B>1) variants (v0.6実装計画.md §2-1). Both patch
        global ``transformers`` symbols, so they affect every ModernBert
        instance in the process; both are semantically equivalent to
        upstream, and re-applying them is harmless.

        Args:
            model: Model returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value; when
                given, :func:`patch_mask_fill_value` is applied to the
                backbone of ``model``.

        Raises:
            ValueError: If the RoPE head dimension is odd (which would make
                the ``chunk``-based ``rotate_half`` rewrite inexact), or if
                no ModernBert backbone can be found for the mask patch.
        """
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"odd RoPE head dim ({head_dim}) is incompatible with patch_rotate_half"
            )
        patch_rotate_half()
        patch_eager_attention_rank4()
        if mask_fill_value is not None:
            patch_mask_fill_value(_resolve_backbone(model), mask_fill_value)

    def wrap(self, model: PreTrainedModel, kind: str) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for ``kind``.

        Args:
            model: Model returned by :meth:`load`.
            kind: Model kind.

        Returns:
            :class:`EmbeddingWrapper` (in-graph mean pooling) or
            :class:`RerankerWrapper` (raw logits), in eval mode.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        wrapper_class = EmbeddingWrapper if kind == KIND_EMBEDDING else RerankerWrapper
        return wrapper_class(model).eval()

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return OUTPUT_NAMES[kind]

    def trace_example(self, kind: str) -> Any:
        """Return the fixed raw example input used for ``torch.jit.trace``.

        Args:
            kind: Model kind.

        Returns:
            A sentence (embedding) or a (query, document) pair (reranker).
            The caller replicates it to B rows before tracing so the traced
            graph already carries the target batch size.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return TRACE_EXAMPLE_TEXT if kind == KIND_EMBEDDING else TRACE_EXAMPLE_PAIR

    def sanity_inputs(self, kind: str) -> list[Any]:
        """Return the fixed raw sanity-check inputs for ``kind``.

        Returns:
            A fresh list of sentences (embedding) or (query, document)
            pairs (reranker); callers may mutate it freely.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return list(SANITY_TEXTS) if kind == KIND_EMBEDDING else list(SANITY_PAIRS)

    def padding_input(self, kind: str) -> Any:
        """Return the filler input used to pad a partial batch.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return BATCH_PADDING_TEXT if kind == KIND_EMBEDDING else BATCH_PADDING_PAIR

    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, kind: str, inputs: list[Any], seq_len: int
    ) -> dict[str, np.ndarray]:
        """Tokenize raw inputs into fixed-shape int32 Core ML arrays.

        Args:
            tokenizer: Tokenizer returned by :meth:`load`.
            kind: Model kind, selecting single-sequence vs pair encoding.
            inputs: Sentences (embedding) or (query, document) pairs
                (reranker).
            seq_len: Fixed sequence length S.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` of shape
            ``(len(inputs), seq_len)`` and dtype ``np.int32``.

        Raises:
            ValueError: If ``kind`` is unsupported, ``inputs`` is empty, or
                ``seq_len`` is not positive.
        """
        self._check_kind(kind)
        if not inputs:
            raise ValueError("no inputs to tokenize")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer (got {seq_len})")
        if kind == KIND_EMBEDDING:
            return tokenize_batch(tokenizer, list(inputs), seq_len)
        return tokenize_pairs(tokenizer, [(query, doc) for query, doc in inputs], seq_len)

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Compute the FP32 (sdpa) reference outputs for ``inputs``.

        Loads a second copy of the model with the untouched ``sdpa``
        attention path -- the patches only rewrite ``eager`` -- runs it row
        by row, and releases it again (the FP32 weights are ~1.2 GB for a
        310M model).

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind.
            inputs: Sentences or (query, document) pairs.
            seq_len: Fixed sequence length S.

        Returns:
            Pooled embeddings of shape (N, hidden_size) for ``embedding``,
            raw logits of shape (N,) for ``reranker``; dtype float32.

        Raises:
            ValueError: If ``kind`` is unsupported or ``inputs`` is empty.
        """
        self._check_kind(kind)
        if not inputs:
            raise ValueError("no inputs to score")
        model, tokenizer = self.load(model_dir, kind, attn="sdpa")
        try:
            if kind == KIND_EMBEDDING:
                return encode_pytorch(model, tokenizer, list(inputs), seq_len)
            return score_pytorch(model, tokenizer, [(q, d) for q, d in inputs], seq_len)
        finally:
            del model
            gc.collect()

    def _check_kind(self, kind: str) -> None:
        """Validate a model kind against :data:`SUPPORTED_KINDS`.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        if kind not in SUPPORTED_KINDS:
            supported = ", ".join(SUPPORTED_KINDS)
            raise ValueError(f"unsupported kind '{kind}' for {self.name} (supported: {supported})")


def _resolve_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """Return the ModernBertModel backbone owning ``_update_attention_mask``.

    The mask helper lives on the backbone, which *is* the model for an
    embedding model but is ``model.model`` for a SequenceClassification
    model.

    Raises:
        ValueError: If neither the model nor its ``model`` attribute owns
            ``_update_attention_mask``.
    """
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "_update_attention_mask"):
            return candidate
    raise ValueError(
        f"cannot locate a ModernBert backbone with _update_attention_mask on "
        f"{type(model).__name__}; the mask fill patch cannot be applied"
    )
