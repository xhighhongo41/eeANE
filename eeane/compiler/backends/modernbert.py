"""ModernBERT compile backend, ported from poc/.

This module is the eeANE-side home of the conversion logic proven by the
PoC scripts (``poc/convert_common.py``, ``poc/convert_embedding.py``,
``poc/convert_reranker.py`` and ``poc/common.py``). The PoC tree is frozen
as a historical record, so the code here -- not ``poc/`` -- is from now on
the single source of truth for the ModernBERT patches, the wrappers, the
fixed trace fixtures, this family's own Japanese sanity fixtures and the
FP32 reference computations.

The three monkeypatches are mandatory parts of the conversion, not
optional tweaks, and each of them is needed under transformers 5:

* :func:`patch_rotate_half` makes the model convertible at all under
  coremltools 9.0 + numpy 2.x: the upstream ``rotate_half`` still slices
  with a traced size, which records a cast the converter rejects.
* :func:`patch_eager_attention_rank4` keeps the compiled model loadable on
  the ANE for batch sizes greater than one: the upstream attention still
  splits the fused projection through a rank-5 tensor.
* :func:`patch_mask_builders` makes the graph convertible and keeps it
  finite in FP16: the backbone now builds its masks through the
  framework's shared mask helpers, which record an op the converter has
  no translation for and fill masked positions with the float32 minimum.

Everything that is not specific to this architecture -- pooling, the
sentence-transformers declaration readers, the stable sigmoid,
fixed-shape tokenization, the FP32 baselines, the traceable wrappers and
the per-language sanity sets -- lives in
:mod:`eeane.compiler.backends.common` and is re-exported here, so that
the names stay reachable under this module.

Importing this module pulls in ``torch``/``transformers``; it therefore
requires the ``[compile]`` extra and must never be imported from the
``eeane serve`` code path (see :mod:`eeane.compiler`).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.models.modernbert import modeling_modernbert

from eeane.compiler.backends.base import (
    SCORE_SPACE_PROBABILITY,
    LoadedModel,
    PairTemplate,
    SanitySpec,
)
from eeane.compiler.backends.common import (
    POOLING_CLS,
    POOLING_DIRNAME,
    POOLING_MEAN,
    POOLING_MODE_KEYS,
    POOLING_MODE_PREFIX,
    SANITY_IRRELEVANT_INDEX,
    SANITY_LANGUAGE_JA,
    SANITY_RELEVANT_INDEX,
    ClsEmbeddingWrapper,
    EmbeddingWrapper,
    RerankerWrapper,
    encode_pytorch,
    load_dense,
    mean_pool,
    override_sanity_set,
    read_pooling_mode,
    score_pytorch,
    sigmoid_np,
    tokenize_batch,
    tokenize_pairs,
)
from eeane.compiler.backends.common import (
    SANITY_PAIR_SETS as SHARED_SANITY_PAIR_SETS,
)
from eeane.compiler.backends.common import (
    SANITY_TEXT_SETS as SHARED_SANITY_TEXT_SETS,
)
from eeane.compiler.backends.masks import MASK_FILL_VALUE, bidirectional_masks

# Public surface of this module, including the architecture-independent
# helpers it re-exports from :mod:`eeane.compiler.backends.common`.
__all__ = [
    "EMBEDDING_WRAPPERS",
    "MASK_FILL_VALUE",
    "OUTPUT_NAMES",
    "POOLING_DIRNAME",
    "POOLING_MODE_KEYS",
    "POOLING_MODE_PREFIX",
    "SANITY_PAIRS",
    "SANITY_PAIR_SETS",
    "SANITY_SPECS",
    "SANITY_TEXTS",
    "SANITY_TEXT_SETS",
    "SUPPORTED_KINDS",
    "ClsEmbeddingWrapper",
    "EmbeddingWrapper",
    "ModernBertBackend",
    "RerankerWrapper",
    "encode_pytorch",
    "load_dense",
    "mean_pool",
    "patch_eager_attention_rank4",
    "patch_mask_builders",
    "patch_rotate_half",
    "read_pooling_mode",
    "score_pytorch",
    "sigmoid_np",
    "tokenize_batch",
    "tokenize_pairs",
]

# Model kinds understood by this backend.
KIND_EMBEDDING = "embedding"
KIND_RERANKER = "reranker"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING, KIND_RERANKER)

# Core ML graph output name per kind (embeddings vs raw relevance logits).
OUTPUT_NAMES: dict[str, str] = {KIND_EMBEDDING: "embedding", KIND_RERANKER: "logits"}

# Traceable wrapper per detected pooling mode.
EMBEDDING_WRAPPERS: dict[str, type[torch.nn.Module]] = {
    POOLING_MEAN: EmbeddingWrapper,
    POOLING_CLS: ClsEmbeddingWrapper,
}

# Model directory file and key the effective maximum sequence length is
# read from. This architecture indexes positions from 0 with no reserved
# offset, so the configured position budget is the usable length.
CONFIG_FILENAME = "config.json"
MAX_POSITION_KEY = "max_position_embeddings"

# Short Japanese sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "これは変換用のサンプル文です。"

# Fixed Japanese sanity-check sentences (short / medium / long) exercising
# different amounts of padding under the same fixed sequence length. They
# replace the shared Japanese set below: the accuracy numbers recorded for
# the already-verified models of this family were measured on these exact
# sentences, and rewording them would move those numbers.
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

# Fixed Japanese sanity-check pairs: relevant / irrelevant / partially
# related. They replace the shared Japanese set below, for the reason the
# sentences above are kept for.
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

# The per-language sanity sets this backend serves: the shared ones, with
# the Japanese set replaced by the fixtures above.
SANITY_TEXT_SETS: tuple[tuple[str, tuple[str, ...]], ...] = override_sanity_set(
    SHARED_SANITY_TEXT_SETS, SANITY_LANGUAGE_JA, tuple(SANITY_TEXTS)
)
SANITY_PAIR_SETS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = override_sanity_set(
    SHARED_SANITY_PAIR_SETS, SANITY_LANGUAGE_JA, tuple(SANITY_PAIRS)
)

# Sanity fixtures per kind, as handed to the pipeline and the self-check.
# Every pair set is ordered relevant, irrelevant, partially related, so the
# reranker is expected to score pair 0 of a set above pair 1; embeddings are
# compared row by row against their own baseline and carry no ordering
# expectation.
SANITY_SPECS: dict[str, SanitySpec] = {
    KIND_EMBEDDING: SanitySpec(input_sets=SANITY_TEXT_SETS),
    KIND_RERANKER: SanitySpec(
        input_sets=SANITY_PAIR_SETS,
        relevant_index=SANITY_RELEVANT_INDEX,
        irrelevant_index=SANITY_IRRELEVANT_INDEX,
    ),
}

# Filler rows used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. The empty strings encode to special tokens
# only, so the row still has a non-empty attention mask (a fully masked row
# would risk NaN).
BATCH_PADDING_TEXT = ""
BATCH_PADDING_PAIR: tuple[str, str] = ("", "")

# Captured at import time so patch_eager_attention_rank4() can delegate the
# non-eager attention paths to the untouched upstream implementation, and so
# that applying the patch twice cannot recurse.
_ORIGINAL_ATTENTION_FORWARD = modeling_modernbert.ModernBertAttention.forward

# Captured at import time for the same two reasons: patch_mask_builders()
# delegates every mask it does not build itself to these upstream builders,
# and applying it twice must not make a replacement delegate to another one.
_ORIGINAL_CREATE_FULL_MASK = modeling_modernbert.create_bidirectional_mask
_ORIGINAL_CREATE_SLIDING_MASK = modeling_modernbert.create_bidirectional_sliding_window_mask


def patch_rotate_half() -> None:
    """Replace ModernBert's ``rotate_half`` with a static-shape equivalent.

    The upstream implementation slices with ``x.shape[-1] // 2``, which
    traces to ``aten::size -> floor_divide -> aten::Int``. coremltools 9.0
    cannot convert that ``aten::Int`` under numpy 2.x and raises
    "only 0-dimensional arrays can be converted to Python scalars".
    ``torch.chunk`` with a constant chunk count yields the identical result
    without any dynamic shape arithmetic; this is exact because RoPE head
    dimensions are always even (enforced by
    :meth:`ModernBertBackend.apply_patches`).
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    modeling_modernbert.rotate_half = rotate_half


def patch_eager_attention_rank4() -> None:
    """Rewrite ModernBert's eager attention without rank-5 intermediates.

    Upstream splits the fused projection with
    ``qkv.view(*input_shape, 3, -1, head_dim).unbind(dim=-3)``, which
    materializes a rank-5 tensor. At B=1 the leading batch axis is
    degenerate and the ANE compiler copes, but for B>1 loading the compiled
    model on CPU_AND_NE fails with
    "MILCompilerForANE error: failed to compile ANE model using ANEF.
    Error=_ANECompiler : ANECCompile() FAILED." and the whole model silently
    falls back to CPU.

    Because ``Wqkv`` emits the three projections contiguously along the last
    axis with the 3 as the outermost sub-index, ``chunk(3, dim=-1)``
    followed by a rank-4 ``view``/``transpose`` selects exactly the same
    elements as the upstream rank-5 path. The rotary embedding and the
    attention itself stay upstream's own ``apply_rotary_pos_emb`` and
    ``eager_attention_forward``, so the rewrite is bit-exact against the
    upstream eager path and only changes tensor rank, never semantics.

    Only the ``eager`` attention path is rewritten; ``sdpa`` and
    ``flash_attention_2`` keep the upstream implementation, so the FP32
    baselines used by the sanity checks are unaffected. Re-applying the
    patch is safe: the delegation target is the implementation captured at
    import time, so it can never recurse into itself.

    Raises:
        ValueError: If a patched module's head geometry contradicts the
            ``heads * head_dim == hidden_size`` layout assumption.
    """

    def attention_forward(
        self: modeling_modernbert.ModernBertAttention,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Replacement for ``ModernBertAttention.forward``."""
        if self.config._attn_implementation != "eager":
            return _ORIGINAL_ATTENTION_FORWARD(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )
        heads, head_dim = self.config.num_attention_heads, self.head_dim
        if heads * head_dim != self.config.hidden_size:
            raise ValueError(
                "unexpected ModernBert head geometry "
                f"({heads} heads x {head_dim} != {self.config.hidden_size}); "
                "the rank-4 qkv split assumption does not hold"
            )
        bs = hidden_states.shape[0]
        # (B, S, 3*H*D) -> three (B, H, S, D) tensors, all rank 4: the
        # rank-5 view of the upstream forward is skipped entirely.
        query, key, value = (
            part.view(bs, -1, heads, head_dim).transpose(1, 2)
            for part in self.Wqkv(hidden_states).chunk(3, dim=-1)
        )
        cos, sin = position_embeddings
        # Looked up on the module at call time, so that the rotary
        # embedding uses the rotate_half of patch_rotate_half().
        query, key = modeling_modernbert.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
        # The layer's mask already encodes its sliding window, so none is
        # passed: upstream's eager function ignores it anyway.
        attn_output, attn_weights = modeling_modernbert.eager_attention_forward(
            self,
            query,
            key,
            value,
            attention_mask,
            scaling=head_dim**-0.5,
            dropout=self.attention_dropout if self.training else 0.0,
        )
        attn_output = attn_output.reshape(bs, -1, heads * head_dim)
        return self.out_drop(self.Wo(attn_output)), attn_weights

    modeling_modernbert.ModernBertAttention.forward = attention_forward


def patch_mask_builders() -> None:
    """Build ModernBert's eager attention masks in-graph with a finite fill value.

    The backbone builds one additive mask per layer type -- the full mask
    and the sliding-window mask -- by calling the framework's shared mask
    helpers through two module-level names of the modeling module. On the
    eager path that construction stands in the way of the conversion
    twice: it materializes the mask with a ``new_ones`` op, which the Core
    ML converter has no translation for, and it fills the masked positions
    with ``torch.finfo(float32).min``, which becomes ``-inf`` once the graph
    runs in FP16 and makes softmax produce NaN for a fully masked row (a
    padding query whose whole window is padding, for instance).

    Both names are rebound to replacements that, on the eager path and
    given a 2-D mask, build the two masks with
    :func:`~eeane.compiler.backends.masks.bidirectional_masks` instead:
    static-shape operations only, and :data:`MASK_FILL_VALUE` in the masked
    positions. The positions they hide are exactly the ones upstream hides
    -- the configured window is an inclusive radius, so the builder's
    exclusive half-width is that radius plus one -- and only the fill value
    differs. Each layer still receives the 4-D mask of its type, and the
    2-D mask the backbone was given still reaches a classification head
    that pools with it.

    Every other case -- a non-eager implementation (in particular the
    ``sdpa`` FP32 reference, which thereby stays an independent check of
    these masks), no mask, or a mask that is not 2-D -- is delegated to the
    upstream builder captured at import time, so re-applying the patch is
    harmless.
    """

    def create_bidirectional_mask(
        config: Any,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Replacement for the full-attention mask builder."""
        built = _in_graph_masks(config, attention_mask)
        if built is None:
            return _ORIGINAL_CREATE_FULL_MASK(
                config, inputs_embeds, attention_mask, *args, **kwargs
            )
        return built[0]

    def create_bidirectional_sliding_window_mask(
        config: Any,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Replacement for the sliding-window mask builder."""
        built = _in_graph_masks(config, attention_mask)
        if built is None:
            return _ORIGINAL_CREATE_SLIDING_MASK(
                config, inputs_embeds, attention_mask, *args, **kwargs
            )
        return built[1]

    modeling_modernbert.create_bidirectional_mask = create_bidirectional_mask
    modeling_modernbert.create_bidirectional_sliding_window_mask = (
        create_bidirectional_sliding_window_mask
    )


class ModernBertBackend:
    """Compile backend for the ModernBERT architecture family.

    Implements the backend interface declared in
    :mod:`eeane.compiler.backends.base`, which documents what each member
    is for, in which order the pipeline calls them, and the rules an
    implementation must follow. Every method is stateless: all per-model
    state travels in the :class:`~eeane.compiler.backends.base.LoadedModel`
    handle, so one instance can serve several compile runs.
    """

    name = "ModernBert"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 model and its tokenizer from a HF model directory.

        Args:
            model_dir: Local HuggingFace-format model directory. It is
                only ever read from.
            kind: ``"embedding"`` (``AutoModel``) or ``"reranker"``
                (``AutoModelForSequenceClassification``).
            attn: Attention implementation to request. ``"eager"`` is the
                path the patches rewrite (conversion); ``"sdpa"`` is used
                for the FP32 reference.

        Returns:
            A handle holding the model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs), its tokenizer, its configuration and -- for an
            embedding model -- the pooling and the Dense projection
            declared by the model directory.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, if
                the pooling of an embedding model cannot be determined, or
                if its declared module chain cannot be reproduced.
        """
        self._check_kind(kind)
        # Both declarations are read before the weights: an undeclared
        # pooling or an unreproducible module chain makes the model
        # uncompilable, and finding that out first avoids loading
        # gigabytes of FP32 parameters for nothing. A reranker has
        # neither: its score comes from its own classification head.
        embedding = kind == KIND_EMBEDDING
        pooling = read_pooling_mode(model_dir) if embedding else None
        dense, dense_config = load_dense(model_dir) if embedding else (None, None)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        loader = AutoModel if embedding else AutoModelForSequenceClassification
        model = loader.from_pretrained(model_dir, attn_implementation=attn, dtype=torch.float32)
        model.config.return_dict = False
        return LoadedModel(
            model=model.eval(),
            tokenizer=tokenizer,
            config=model.config,
            model_dir=model_dir,
            kind=kind,
            attn=attn,
            pooling=pooling,
            dense=dense,
            dense_config=dense_config,
        )

    def apply_patches(
        self, loaded: LoadedModel, mask_fill_value: float | None = None
    ) -> dict[str, Any]:
        """Apply the mandatory ModernBERT graph patches.

        :func:`patch_rotate_half`, :func:`patch_eager_attention_rank4` and
        :func:`patch_mask_builders` are mandatory constituents of the
        conversion, applied at every batch size so that one graph shape is
        produced instead of mixing rank-5 (B=1) and rank-4 (B>1) variants.
        All three patch global ``transformers`` symbols, so they affect
        every ModernBert instance in the process; all three compute what
        upstream computes -- the masks with a finite fill value in place of
        the float32 minimum -- and re-applying them is harmless.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value.
                The patched graph always builds its masks itself with
                :data:`MASK_FILL_VALUE` (already a finite value); the
                argument is therefore only accepted when it names that
                very value.

        Returns:
            ``{"rotate_half_static": True, "eager_attention_rank4": True,
            "mask_builders_static": True, "mask_fill_value":
            MASK_FILL_VALUE}``: the three rewrites are always applied by
            this backend, and the fill value of the masks the graph builds
            is always recorded with them.

        Raises:
            ValueError: If the RoPE head dimension is odd (which would make
                the ``chunk``-based ``rotate_half`` rewrite inexact), or if
                a mask fill value other than :data:`MASK_FILL_VALUE` is
                requested. Nothing is rebound in either case.
        """
        config = loaded.config
        head_dim = config.hidden_size // config.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"odd RoPE head dim ({head_dim}) is incompatible with patch_rotate_half"
            )
        if mask_fill_value is not None and mask_fill_value != MASK_FILL_VALUE:
            raise ValueError(
                f"{self.name} builds its attention masks inside the traced graph with a "
                f"mask fill value of {MASK_FILL_VALUE}; {mask_fill_value} cannot be applied"
            )
        patch_rotate_half()
        patch_eager_attention_rank4()
        patch_mask_builders()
        return {
            "rotate_half_static": True,
            "eager_attention_rank4": True,
            "mask_builders_static": True,
            "mask_fill_value": MASK_FILL_VALUE,
        }

    def wrap(self, loaded: LoadedModel) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`; for an embedding
                model its ``pooling`` selects the wrapper and its
                ``dense`` is applied after that pooling.

        Returns:
            The pooling wrapper matching ``loaded.pooling`` for an
            embedding model, or the raw-logits wrapper for a reranker, in
            eval mode.

        Raises:
            ValueError: If ``loaded.kind`` is not supported by this
                backend, or if an embedding handle carries a pooling mode
                no wrapper implements.
        """
        self._check_kind(loaded.kind)
        if loaded.kind == KIND_RERANKER:
            return RerankerWrapper(loaded.model).eval()
        wrapper_class = EMBEDDING_WRAPPERS.get(loaded.pooling or "")
        if wrapper_class is None:
            supported = ", ".join(EMBEDDING_WRAPPERS)
            raise ValueError(
                f"unsupported pooling '{loaded.pooling}' for {self.name} (supported: {supported})"
            )
        return wrapper_class(loaded.model, dense=loaded.dense).eval()

    def output_name(self, kind: str) -> str:
        """Return the Core ML graph output name used for ``kind``.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return OUTPUT_NAMES[kind]

    def max_seq_len(self, model_dir: Path) -> int | None:
        """Return the effective maximum sequence length of ``model_dir``.

        Positions are indexed from 0 with no reserved offset, so the
        configured position budget is the usable sequence length. Only
        ``config.json`` is read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Returns:
            The configured positive position budget, or ``None`` when the
            file is absent/unreadable/unparsable or the value is missing
            or not a positive integer. ``None`` means "unknown", and the
            caller then imposes no limit.
        """
        try:
            raw = (model_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
            config = json.loads(raw)
        except (OSError, ValueError):
            # A malformed config is reported by the dispatch step; an
            # optional bucket check must not turn it into a second error.
            return None
        if not isinstance(config, dict):
            return None
        value = config.get(MAX_POSITION_KEY)
        # bool is a subclass of int, but a JSON ``true`` is not a length.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    def pair_template(self, model_dir: Path, kind: str) -> PairTemplate | None:
        """Return how this model wants a (query, document) pair spelled out.

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind the template is asked for.

        Returns:
            Always ``None``: this architecture encodes a pair through the
            tokenizer's own pair encoding, so there is nothing to shape.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        # This architecture hands both sequences to the tokenizer, whose
        # own post_processor produces the model's pair template; there is
        # nothing left for a caller to spell out.
        return None

    def reranker_score_space(self) -> str:
        """Return the space this backend's reranker scores are faithful in.

        Returns:
            :data:`~eeane.compiler.backends.base.SCORE_SPACE_PROBABILITY`:
            the classification head emits one calibrated relevance value
            per pair, and the probability its sigmoid produces is the
            number the server hands out.
        """
        return SCORE_SPACE_PROBABILITY

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

    def sanity_spec(self, kind: str) -> SanitySpec:
        """Return the fixed sanity-check inputs and metadata for ``kind``.

        Returns:
            The immutable specification for ``kind``: sentences for an
            embedding model, (query, document) pairs plus the expected
            relevant/irrelevant indices for a reranker.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return SANITY_SPECS[kind]

    def padding_input(self, kind: str) -> Any:
        """Return the filler input used to pad a partial batch.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return BATCH_PADDING_TEXT if kind == KIND_EMBEDDING else BATCH_PADDING_PAIR

    def tokenize(
        self, loaded: LoadedModel, inputs: list[Any], seq_len: int
    ) -> dict[str, np.ndarray]:
        """Tokenize raw inputs into fixed-shape int32 Core ML arrays.

        Args:
            loaded: Handle returned by :meth:`load`; its tokenizer encodes
                the inputs and its kind selects single-sequence vs pair
                encoding.
            inputs: Sentences (embedding) or (query, document) pairs
                (reranker).
            seq_len: Fixed sequence length S.

        Returns:
            Dict with ``input_ids`` and ``attention_mask`` of shape
            ``(len(inputs), seq_len)`` and dtype ``np.int32``.

        Raises:
            ValueError: If ``loaded.kind`` is unsupported, ``inputs`` is
                empty, or ``seq_len`` is not positive.
        """
        self._check_kind(loaded.kind)
        if not inputs:
            raise ValueError("no inputs to tokenize")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be a positive integer (got {seq_len})")
        if loaded.kind == KIND_EMBEDDING:
            return tokenize_batch(loaded.tokenizer, list(inputs), seq_len)
        return tokenize_pairs(loaded.tokenizer, [(query, doc) for query, doc in inputs], seq_len)

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
        loaded = self.load(model_dir, kind, attn="sdpa")
        try:
            if kind == KIND_EMBEDDING:
                # The baseline must pool and project exactly like the
                # traced wrapper, so it follows both declarations of this
                # directory.
                return encode_pytorch(
                    loaded.model,
                    loaded.tokenizer,
                    list(inputs),
                    seq_len,
                    pooling=loaded.pooling,
                    dense=loaded.dense,
                )
            return score_pytorch(
                loaded.model, loaded.tokenizer, [(q, d) for q, d in inputs], seq_len
            )
        finally:
            del loaded
            gc.collect()

    def _check_kind(self, kind: str) -> None:
        """Validate a model kind against :data:`SUPPORTED_KINDS`.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        if kind not in SUPPORTED_KINDS:
            supported = ", ".join(SUPPORTED_KINDS)
            raise ValueError(f"unsupported kind '{kind}' for {self.name} (supported: {supported})")


def _in_graph_masks(
    config: Any, attention_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build the (full, sliding) masks :func:`patch_mask_builders` hands the layers.

    Args:
        config: Configuration the backbone hands its mask builders.
        attention_mask: Mask the backbone was given.

    Returns:
        Both additive masks, each of shape ``(B, 1, S, S)``, on the eager
        path for a 2-D mask; ``None`` for every case the upstream builders
        keep (a non-eager implementation, no mask, or a mask that is not
        2-D).
    """
    if config._attn_implementation != "eager":
        return None
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return None
    # ``sliding_window`` is an inclusive radius (a key is visible when
    # ``abs(query - key) <= radius``); the builder takes an exclusive one.
    return bidirectional_masks(
        attention_mask, window=config.sliding_window + 1, fill_value=MASK_FILL_VALUE
    )
