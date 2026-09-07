"""Qwen3 compile backend for decoder-style embedding models.

Every other backend in this package compiles an encoder: a bidirectional
stack whose attention mask is a plain padding mask and whose sentence
vector is pooled over all positions. This one compiles a decoder -- a
causal language-model backbone reused as a text encoder, with grouped
attention heads, rotary positions and a sentence vector read off the last
real token of the row.

Three properties of that family decide how the conversion is built here:

* **The attention mask is built in-graph.** The wrapper hands the
  backbone a 4-D additive mask that already combines causality and
  padding, which bypasses the framework's own mask construction entirely
  while computing the same numbers. No mask helper has to be rewritten.
* **Two upstream constructs are replaced** before tracing:
  :func:`patch_rotate_half` removes shape arithmetic the Core ML
  converter cannot fold, and :func:`patch_repeat_kv` removes the rank-5
  intermediate that makes the Neural Engine compiler reject the attention
  subgraph. Both replacements compute exactly what upstream computes.
* **Pooling reads the last real token**, not the first one and not an
  average, because in a causal stack that is the only position that has
  attended to the whole sequence.

The model kind cannot be told from the architecture name: a decoder-style
embedding model and a decoder-style reranker are both published as
``Qwen3ForCausalLM``. :mod:`eeane.compiler.dispatch` therefore resolves
the kind from the sentence-transformers module declaration of the model
directory instead. This backend compiles the embedding kind.

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
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

from eeane.compiler.backends.base import LoadedModel, SanitySpec
from eeane.compiler.backends.common import (
    POOLING_DIRNAME,
    POOLING_LASTTOKEN,
    POOLING_MODE_KEYS,
    POOLING_MODE_PREFIX,
    SANITY_TEXT_SETS,
    encode_pytorch,
    last_token_pool,
    load_dense,
    read_pooling_mode,
    tokenize_batch,
)

# Public surface of this module, including the architecture-independent
# helpers it re-exports from :mod:`eeane.compiler.backends.common`.
__all__ = [
    "CONFIG_FILENAME",
    "EMBEDDING_WRAPPERS",
    "MASK_FILL_VALUE",
    "MAX_POSITION_KEY",
    "MODEL_TYPE",
    "OUTPUT_NAMES",
    "POOLING_DIRNAME",
    "POOLING_MODE_KEYS",
    "POOLING_MODE_PREFIX",
    "SANITY_SPECS",
    "SANITY_TEXT_SETS",
    "SUPPORTED_KINDS",
    "CausalLastTokenWrapper",
    "Qwen3Backend",
    "build_causal_padding_mask",
    "encode_pytorch",
    "last_token_pool",
    "load_dense",
    "patch_repeat_kv",
    "patch_rotate_half",
    "read_pooling_mode",
    "tokenize_batch",
]

# Model kinds understood by this backend. The decoder-style reranker of
# this family scores a pair by reading a single vocabulary logit, which is
# a different graph from the one built here; it is not compiled yet.
KIND_EMBEDDING = "embedding"
SUPPORTED_KINDS: tuple[str, ...] = (KIND_EMBEDDING,)

# Core ML graph output name per kind.
OUTPUT_NAMES: dict[str, str] = {KIND_EMBEDDING: "embedding"}

# Additive value written into the masked-out positions of the attention
# mask. The value the framework itself uses is the float32 minimum, which
# is not representable in FP16: once the converted graph runs in FP16 it
# becomes -inf, and a row whose keys are all masked then computes
# -inf - (-inf) = NaN inside the softmax. -1e4 is exact in FP16, drives
# exp() to exactly 0.0 in both precisions, and keeps such a row finite.
MASK_FILL_VALUE = -1e4

# Model directory file, and the keys read from it. ``model_type`` is what
# distinguishes this architecture from its relatives, which the
# architecture-name prefix alone does not; positions are rotary and index
# from 0, so the configured position budget is the usable length.
CONFIG_FILENAME = "config.json"
MODEL_TYPE_KEY = "model_type"
MAX_POSITION_KEY = "max_position_embeddings"

# The HF ``model_type`` this backend implements.
MODEL_TYPE = "qwen3"

# Short English sentence used as the example input for torch.jit.trace.
TRACE_EXAMPLE_TEXT = "This is a short sample sentence used for conversion."

# Filler row used to pad the last sanity batch when the number of sanity
# inputs is not a multiple of B. It is a real sentence rather than an
# empty string: every row of a causal batch must keep at least one
# attendable position, since a fully masked row can produce NaN.
BATCH_PADDING_TEXT = "This sentence only fills an unused row of the batch."

# Sanity fixtures per kind, as handed to the pipeline and the self-check.
# The shared per-language sets are used unchanged: no model of this family
# has been measured on fixtures of its own, so there is nothing to keep.
SANITY_SPECS: dict[str, SanitySpec] = {KIND_EMBEDDING: SanitySpec(input_sets=SANITY_TEXT_SETS)}


def patch_rotate_half() -> None:
    """Replace Qwen3's ``rotate_half`` with a static-shape equivalent.

    The upstream implementation splits the last dimension with
    ``x[..., : x.shape[-1] // 2]``. The Python-level division on a traced
    shape records an ``aten::Int`` node in the TorchScript graph, which
    the Core ML converter cannot fold into a static slice.
    ``torch.chunk`` with a constant chunk count selects the same two
    halves without consulting the shape at graph level; the results are
    identical whenever the head dimension is even, which
    :meth:`Qwen3Backend.apply_patches` enforces.

    Rebinds the module-level function, which the rotary helper looks up at
    call time, so the replacement takes effect for every Qwen3 model in
    the process. Re-applying it is harmless.
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    modeling_qwen3.rotate_half = rotate_half


def patch_repeat_kv() -> None:
    """Replace the grouped-query key/value expansion with a rank-4 form.

    Grouped-query attention stores fewer key/value heads than query heads
    and expands them before the attention product. Upstream does that
    through a rank-5 intermediate
    (``x[:, :, None, :, :].expand(...).reshape(...)``); the Neural Engine
    compiler rejects that subgraph, and rejecting one subgraph makes the
    whole compiled model fall back to the CPU.

    ``repeat_interleave`` stays rank-4 and repeats each key/value head
    ``n_rep`` times in place, which is exactly the upstream head order
    ``head = kv_index * n_rep + rep_index``. As upstream does, an
    expansion by one returns the input untouched.

    Rebinds the module-level function, so the replacement takes effect for
    every Qwen3 model in the process. Re-applying it is harmless.
    """

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        return hidden_states.repeat_interleave(n_rep, dim=1)

    modeling_qwen3.repeat_kv = repeat_kv


def build_causal_padding_mask(
    attention_mask: torch.Tensor, fill_value: float = MASK_FILL_VALUE
) -> torch.Tensor:
    """Build the additive 4-D causal + padding attention mask.

    The result is ``0.0`` where query position ``i`` may attend to key
    position ``j`` -- that is, where ``j <= i`` (causal) and position ``j``
    is not padding -- and ``fill_value`` everywhere else.

    Passing this tensor as the model's ``attention_mask`` replaces the
    framework's own mask construction with these few operations: the mask
    utilities forward a 4-D mask to the attention implementation
    unchanged. The numbers are the ones the framework's path produces
    (with the finite fill value substituted), so no mask helper has to be
    rewritten for the conversion, which is what keeps the FP32 baseline --
    which does go through the framework's path -- an independent check.

    Only ``torch.arange`` comparisons, a broadcast and ``torch.where`` are
    used, with no data-dependent control flow: the causal half is a
    constant the tracer folds away, and the padding half stays a broadcast
    of the input mask.

    Args:
        attention_mask: 2-D mask of shape ``(B, S)``, integer or float,
            where a non-zero value marks a real token.
        fill_value: Additive value written into masked positions; see
            :data:`MASK_FILL_VALUE` for why it is a finite number.

    Returns:
        Float32 tensor of shape ``(B, 1, S, S)``.
    """
    seq_len = attention_mask.shape[-1]
    positions = torch.arange(seq_len, device=attention_mask.device)
    # causal[i, j] is True when key j is at or before query i.
    causal = positions.unsqueeze(0) <= positions.unsqueeze(-1)  # (S, S)
    keep = causal[None, None, :, :] & attention_mask.to(torch.bool)[:, None, None, :]
    zero = torch.zeros((), dtype=torch.float32, device=attention_mask.device)
    fill = torch.full((), fill_value, dtype=torch.float32, device=attention_mask.device)
    return torch.where(keep, zero, fill)


class CausalLastTokenWrapper(torch.nn.Module):
    """Wraps a causal backbone, masks it in-graph and pools its last token.

    The output matches a sentence-transformers model whose modules are a
    Transformer, last-token Pooling and the declared Dense projections,
    without normalization: the compiled graph always returns the raw
    vector, and normalizing is left to the server's own configuration.

    Both the attention mask and the pooling are computed inside the
    wrapper, so the exported program takes exactly the two int32 tensors
    Core ML feeds it and leaves no pre- or post-processing to reimplement
    on the host.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dense: torch.nn.Module | None = None,
        fill_value: float = MASK_FILL_VALUE,
    ) -> None:
        """Store the backbone, the projection and the mask fill value.

        Args:
            model: Backbone returning the last hidden state first, loaded
                in eval/FP32 mode with ``config.return_dict = False`` and
                ``config.use_cache = False``.
            dense: Projection applied to the pooled vector, as built by
                ``build_dense``; ``None`` for a model that declares none.
            fill_value: Additive value for masked positions; see
                :data:`MASK_FILL_VALUE`.
        """
        super().__init__()
        self.model = model
        self.dense = dense
        self.fill_value = fill_value

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute last-token-pooled (and, if declared, projected) embeddings.

        Args:
            input_ids: Token ids, shape (B, S).
            attention_mask: Attention mask, shape (B, S); non-zero marks a
                real token, and padding is on the right.

        Returns:
            Embeddings of shape (B, hidden_size), or (B, projected width)
            when a projection was given. They are **not** L2-normalized.
        """
        mask = build_causal_padding_mask(attention_mask, self.fill_value)  # (B, 1, S, S)
        outputs = self.model(input_ids=input_ids, attention_mask=mask)
        hidden = outputs[0]  # (B, S, H)
        pooled = last_token_pool(hidden, attention_mask)  # (B, H)
        if self.dense is None:
            return pooled
        return self.dense(pooled)


# Traceable wrapper per detected pooling mode. In a causal stack no
# position but the last real token has seen the whole sequence, so the
# encoder pooling modes describe nothing this backend could compute and
# are refused rather than approximated.
EMBEDDING_WRAPPERS: dict[str, type[torch.nn.Module]] = {POOLING_LASTTOKEN: CausalLastTokenWrapper}


class Qwen3Backend:
    """Compile backend for the Qwen3 decoder-style embedding family.

    Implements the backend interface declared in
    :mod:`eeane.compiler.backends.base`, which documents what each member
    is for, in which order the pipeline calls them, and the rules an
    implementation must follow. Every method is stateless: all per-model
    state travels in the :class:`~eeane.compiler.backends.base.LoadedModel`
    handle, so one instance can serve several compile runs.
    """

    name = "Qwen3"
    supported_kinds: tuple[str, ...] = SUPPORTED_KINDS

    def load(self, model_dir: Path, kind: str, attn: str = "eager") -> LoadedModel:
        """Load the FP32 model and its tokenizer from a HF model directory.

        Args:
            model_dir: Local HuggingFace-format model directory. It is
                only ever read from.
            kind: Model kind to load the model as; see
                :data:`SUPPORTED_KINDS`.
            attn: Attention implementation to request. ``"eager"`` is the
                path the patches rewrite and therefore the only one a
                conversion may use; ``"sdpa"`` reaches a separate
                implementation that the patches never touch, which is what
                makes it the FP32 reference path.

        Returns:
            A handle holding the model in eval/FP32 mode with
            ``config.return_dict = False`` (``torch.jit.trace`` needs tuple
            outputs) and ``config.use_cache = False``, its tokenizer, its
            configuration, the declared pooling and the declared Dense
            projection.

        Raises:
            ValueError: If ``kind`` is not supported by this backend, if
                the directory declares another architecture, if its
                pooling cannot be determined, or if its declared module
                chain cannot be reproduced.
        """
        self._check_kind(kind)
        # Everything that can make a model uncompilable is decided from
        # the declarations first: loading gigabytes of FP32 parameters for
        # a model that is then refused is pure waste. The architecture is
        # checked before the pooling because it is the more fundamental
        # refusal of the two.
        self._check_model_type(model_dir)
        pooling = read_pooling_mode(model_dir)
        dense, dense_config = load_dense(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # AutoModel returns the backbone alone: the projection onto the
        # vocabulary that a causal checkpoint also carries is never part
        # of an embedding graph.
        model = AutoModel.from_pretrained(model_dir, attn_implementation=attn, dtype=torch.float32)
        model.config.return_dict = False
        # Without this the backbone builds and updates a key/value cache,
        # which ends up in the traced graph as state that a fixed-shape
        # single-pass encoder has no use for.
        model.config.use_cache = False
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
        """Apply the mandatory Qwen3 graph patches.

        Both rewrites are constituents of the conversion rather than
        optional tweaks -- without them the model either fails to convert
        or converts into a program that runs on the CPU -- so both are
        always applied. They patch global ``transformers`` symbols and
        therefore affect every Qwen3 instance in the process; both are
        semantically equivalent to upstream, and re-applying them is
        harmless.

        Args:
            loaded: Handle returned by :meth:`load`.
            mask_fill_value: Optional finite attention-mask fill value.
                This architecture needs no mask patch at all, because its
                wrapper builds the mask itself with :data:`MASK_FILL_VALUE`
                (already a finite value); the argument is therefore only
                accepted when it names that very value.

        Returns:
            The applied rewrites plus the mask fill value the traced graph
            uses, recorded verbatim in the compiled variant's metadata.

        Raises:
            ValueError: If the rotary head dimension is odd (which would
                make the ``chunk``-based ``rotate_half`` inexact), or if a
                mask fill value other than :data:`MASK_FILL_VALUE` is
                requested.
        """
        head_dim = _rope_head_dim(loaded.config)
        if head_dim % 2 != 0:
            raise ValueError(
                f"odd RoPE head dim ({head_dim}) is incompatible with patch_rotate_half"
            )
        if mask_fill_value is not None and mask_fill_value != MASK_FILL_VALUE:
            raise ValueError(
                f"{self.name} builds its attention mask inside the traced graph with a "
                f"mask fill value of {MASK_FILL_VALUE}; {mask_fill_value} cannot be applied"
            )
        patch_rotate_half()
        patch_repeat_kv()
        return {
            "rotate_half_static": True,
            "repeat_kv_rank4": True,
            "mask_fill_value": MASK_FILL_VALUE,
        }

    def wrap(self, loaded: LoadedModel) -> torch.nn.Module:
        """Wrap the loaded model into the traceable module for its kind.

        Args:
            loaded: Handle returned by :meth:`load`; its ``pooling``
                selects the wrapper and its ``dense`` is applied after
                that pooling.

        Returns:
            The wrapper matching ``loaded.pooling``, in eval mode.

        Raises:
            ValueError: If ``loaded.kind`` is not supported by this
                backend, or if the handle carries a pooling mode no
                wrapper implements.
        """
        self._check_kind(loaded.kind)
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

        Positions are rotary and indexed from 0 with no reserved offset,
        so the configured position budget is the usable sequence length.
        Only ``config.json`` is read; no weights are loaded.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Returns:
            The configured positive position budget, or ``None`` when the
            file is absent/unreadable/unparsable or the value is missing
            or not a positive integer. ``None`` means "unknown", and the
            caller then imposes no limit.
        """
        config = _read_config(model_dir)
        if config is None:
            # A malformed config is reported by the dispatch step; an
            # optional bucket check must not turn it into a second error.
            return None
        value = config.get(MAX_POSITION_KEY)
        # bool is a subclass of int, but a JSON ``true`` is not a length.
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    def trace_example(self, kind: str) -> Any:
        """Return the fixed raw example input used for ``torch.jit.trace``.

        Args:
            kind: Model kind.

        Returns:
            A sentence. The caller replicates it to B rows before tracing
            so the traced graph already carries the target batch size.

        Raises:
            ValueError: If ``kind`` is not supported by this backend.
        """
        self._check_kind(kind)
        return TRACE_EXAMPLE_TEXT

    def sanity_spec(self, kind: str) -> SanitySpec:
        """Return the fixed sanity-check inputs and metadata for ``kind``.

        Returns:
            The immutable specification for ``kind``: sentences compared
            row by row against their own baseline, with no expected
            ordering.

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
        return BATCH_PADDING_TEXT

    def tokenize(
        self, loaded: LoadedModel, inputs: list[Any], seq_len: int
    ) -> dict[str, np.ndarray]:
        """Tokenize raw inputs into fixed-shape int32 Core ML arrays.

        The shared helper pads on the right, which is what the last-token
        pooling of this backend assumes: the pooled position of a row is
        the last one its attention mask marks as real.

        Args:
            loaded: Handle returned by :meth:`load`; its tokenizer encodes
                the inputs.
            inputs: Sentences to encode.
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
        return tokenize_batch(loaded.tokenizer, list(inputs), seq_len)

    def reference_outputs(
        self, model_dir: Path, kind: str, inputs: list[Any], seq_len: int
    ) -> np.ndarray:
        """Compute the FP32 (sdpa) reference outputs for ``inputs``.

        Loads a second copy of the model with the untouched ``sdpa``
        attention path -- the patches only rewrite the eager one -- runs it
        row by row and releases it again.

        The baseline also hands the model the plain 2-D attention mask, so
        the causal mask is built by the framework rather than by the
        wrapper under test: the two sides of the self-check compute the
        same function through two independent mask implementations.

        Args:
            model_dir: Local HuggingFace-format model directory.
            kind: Model kind.
            inputs: Sentences to encode.
            seq_len: Fixed sequence length S.

        Returns:
            Pooled embeddings of shape (N, width), dtype float32.

        Raises:
            ValueError: If ``kind`` is unsupported or ``inputs`` is empty.
        """
        self._check_kind(kind)
        if not inputs:
            raise ValueError("no inputs to encode")
        loaded = self.load(model_dir, kind, attn="sdpa")
        try:
            # The baseline must pool and project exactly like the traced
            # wrapper, so it follows both declarations of this directory.
            return encode_pytorch(
                loaded.model,
                loaded.tokenizer,
                list(inputs),
                seq_len,
                pooling=loaded.pooling,
                dense=loaded.dense,
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

    def _check_model_type(self, model_dir: Path) -> None:
        """Validate that ``model_dir`` holds the architecture this backend implements.

        The backend is selected by an architecture-name prefix, which
        related decoders of the same family also match (a
        mixture-of-experts variant, for one). Those are different
        architectures with different modules, so the declared
        ``model_type`` -- the only field that names the implementation --
        decides whether this backend may load the directory at all.

        Args:
            model_dir: Local HuggingFace-format model directory.

        Raises:
            ValueError: If the configuration cannot be read or declares
                another ``model_type``.
        """
        config = _read_config(model_dir)
        declared = config.get(MODEL_TYPE_KEY) if config is not None else None
        if declared != MODEL_TYPE:
            raise ValueError(
                f"'{model_dir / CONFIG_FILENAME}' declares {MODEL_TYPE_KEY}={declared!r}, but "
                f"the {self.name} backend implements the '{MODEL_TYPE}' architecture "
                "only; a related architecture of the same family needs a backend of its own"
            )


def _read_config(model_dir: Path) -> dict[str, Any] | None:
    """Read a model directory's ``config.json`` without loading any weights.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        The parsed configuration, or ``None`` when the file is absent,
        unreadable, unparsable or not a JSON object.
    """
    try:
        raw = (model_dir / CONFIG_FILENAME).read_text(encoding="utf-8")
        config = json.loads(raw)
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _rope_head_dim(config: Any) -> int:
    """Return the width one attention head applies the rotary embedding to.

    This family declares the head width explicitly, and that declaration
    is not always ``hidden_size / num_attention_heads`` -- the published
    checkpoints use a wider head than the even split would give. The
    declared value therefore wins, and the split is only the fallback for
    a configuration that leaves it out.

    Args:
        config: Model configuration of the loaded model.

    Returns:
        The rotary head dimension.
    """
    declared = getattr(config, "head_dim", None)
    if isinstance(declared, int) and not isinstance(declared, bool):
        return declared
    return config.hidden_size // config.num_attention_heads
