"""Shared building blocks for exporting decoder-style embedding models to Core ML.

This package explores whether a decoder-only (causal LM) text embedding
model of the Qwen3 family can be traced with ``torch.jit.trace`` and
converted to a Core ML ``mlprogram`` that runs on the Apple Neural
Engine. This module holds the parts every script in that experiment
shares: tokenization into fixed-shape int32 arrays, construction of the
4-D additive attention mask that replaces the framework's own mask
generation, last-token pooling, an unpatched FP32 reference encoder, and
the small result/metadata helpers.

Two conventions are fixed here and assumed by every consumer:

* Padding goes on the **left**. The model this package targets pools the
  final token of the sequence, so left padding turns pooling into a plain
  ``hidden[:, -1, :]`` read instead of a per-row gather, which keeps the
  traced graph free of data-dependent indexing.
* The attention mask handed to the model is **4-D and additive**. The
  transformers mask utilities pass a 4-D tensor straight through to the
  attention implementation, so building the causal + padding mask here
  bypasses the framework's own (harder to trace) mask construction while
  producing the same attention weights.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from eeane.compiler.backends.common import SANITY_TEXT_SETS

# Hub id of the embedding model this experiment targets.
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

# Instruction prefix the model expects in front of a query. It is plain
# text concatenated ahead of the query, not a tokenizer special token.
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

# Documents are embedded without any instruction prefix.
DOCUMENT_PROMPT = ""

# Additive value written into masked-out positions of the attention mask.
# ``torch.finfo(float32).min`` (the value the framework itself uses) is
# not representable in FP16: once the converted graph runs in FP16 it
# becomes -inf, and a fully masked attention row then produces
# -inf - (-inf) = NaN inside softmax. -1e4 is exact in FP16, still drives
# exp() to exactly 0.0 in both FP16 and FP32, and keeps fully masked rows
# finite (they degenerate to a uniform average, which is what the
# framework's own mask handling does as well).
MASK_FILL_VALUE = -1e4

# Minimum cosine similarity between the converted model and the FP32
# reference for a conversion to be considered faithful.
SANITY_COSINE_THRESHOLD = 0.99

# Minimum cosine similarity between the same text encoded alone and
# encoded inside a larger batch; batching must not change the result
# beyond floating-point noise.
BATCH_CONSISTENCY_COSINE_THRESHOLD = 0.99999

# Minimum percentage of compute-unit ops that must land on the Neural
# Engine before a conversion is reported as healthy.
NE_PLACEMENT_WARN_THRESHOLD = 90.0


def resolve_model_dir(model_id_or_dir: str) -> Path:
    """Resolve a local directory or a Hub id to a local model directory.

    Args:
        model_id_or_dir: Either a path to an existing local
            HuggingFace-format directory, or a Hub model id to download.

    Returns:
        Path to the local model directory. For a Hub id this is the
        snapshot directory; an already downloaded snapshot is reused from
        the local cache instead of being fetched again.
    """
    candidate = Path(model_id_or_dir)
    if candidate.is_dir():
        return candidate
    return Path(snapshot_download(model_id_or_dir))


def load_tokenizer(model_dir: Path, padding_side: str = "left") -> PreTrainedTokenizerBase:
    """Load the tokenizer with an explicit padding side.

    The padding side is passed explicitly because the target model ships
    no ``padding_side`` in its tokenizer configuration and would
    otherwise default to right padding, which does not match the
    last-token pooling this package relies on.

    Args:
        model_dir: Local HuggingFace-format model directory.
        padding_side: ``"left"`` (the default used throughout) or
            ``"right"``.

    Returns:
        The loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(str(model_dir), padding_side=padding_side)


def tokenize_batch(
    tokenizer: Any,
    texts: Sequence[str],
    seq_len: int,
    prompt: str = "",
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    The end-of-text token is appended by the tokenizer's own
    post-processor, so it must not be added here; doing so would emit it
    twice. Padding uses whatever side the tokenizer was configured with
    (see :func:`load_tokenizer`); it is deliberately not overridden per
    call so that a single tokenizer object cannot silently produce
    batches with two different conventions.

    Args:
        tokenizer: Tokenizer whose call signature follows the
            HuggingFace one (any object accepting the keyword arguments
            used below and returning ``input_ids``/``attention_mask``).
        texts: Input texts, without any instruction prefix.
        seq_len: Fixed sequence length used for padding and truncation.
        prompt: Optional instruction prefix concatenated in front of
            every text; an empty string prepends nothing.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(texts), seq_len)`` and dtype ``np.int32``. Any other key
        the tokenizer returns is dropped.
    """
    prompted = [prompt + text for text in texts] if prompt else list(texts)
    encoded = tokenizer(
        prompted,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="np",
    )
    return {
        "input_ids": np.asarray(encoded["input_ids"]).astype(np.int32),
        "attention_mask": np.asarray(encoded["attention_mask"]).astype(np.int32),
    }


def build_causal_padding_mask(
    attention_mask: torch.Tensor, fill_value: float = MASK_FILL_VALUE
) -> torch.Tensor:
    """Build the additive 4-D causal + padding attention mask.

    The result is ``0.0`` where query position ``i`` may attend to key
    position ``j`` -- that is, where ``j <= i`` (causal) and position
    ``j`` is not padding -- and ``fill_value`` everywhere else. The
    transformers mask utilities forward a 4-D mask to the attention
    implementation unchanged, so passing this tensor as ``attention_mask``
    replaces the framework's own mask construction with these few ops.

    Only ``torch.arange`` comparisons and ``torch.where`` are used, so the
    causal half of the mask is a constant the tracer can fold away, and
    the padding half stays a simple broadcast of the input mask.

    Args:
        attention_mask: 2-D mask of shape ``(B, S)``, integer or float,
            where non-zero marks a real token.
        fill_value: Additive value written into masked positions; see
            :data:`MASK_FILL_VALUE` for why it is a finite number.

    Returns:
        Float32 tensor of shape ``(B, 1, S, S)``.
    """
    seq_len = attention_mask.shape[-1]
    positions = torch.arange(seq_len, device=attention_mask.device)
    # causal[i, j] is True when key j is at or before query i.
    causal = positions.unsqueeze(0) <= positions.unsqueeze(-1)
    keep = causal[None, None, :, :] & attention_mask.to(torch.bool)[:, None, None, :]
    zero = torch.zeros((), dtype=torch.float32, device=attention_mask.device)
    fill = torch.full((), fill_value, dtype=torch.float32, device=attention_mask.device)
    return torch.where(keep, zero, fill)


def last_token_pool_left(hidden: torch.Tensor) -> torch.Tensor:
    """Pool the final sequence position of every row.

    Valid **only for left-padded** batches: with padding on the left the
    final position always holds a real token, so the pooled vector is a
    static slice and the traced graph needs no data-dependent gather. For
    right-padded batches use :func:`last_token_pool_gather` instead.

    Args:
        hidden: Last hidden state of shape ``(B, S, H)``.

    Returns:
        Pooled embeddings of shape ``(B, H)``.
    """
    return hidden[:, -1, :]


def last_token_pool_gather(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last unpadded position of every row.

    The right-padding counterpart of :func:`last_token_pool_left`: each
    row is indexed at ``attention_mask.sum(dim=1) - 1``. A row with no
    real tokens at all would give index ``-1``; the index is clamped to
    ``0`` so such a row reads the first position rather than wrapping
    around to the last one.

    Args:
        hidden: Last hidden state of shape ``(B, S, H)``.
        attention_mask: 2-D mask of shape ``(B, S)``, integer or float.

    Returns:
        Pooled embeddings of shape ``(B, H)``.
    """
    lengths = (attention_mask.sum(dim=1).to(torch.long) - 1).clamp(min=0)
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, lengths]


def load_reference_model(model_dir: Path) -> torch.nn.Module:
    """Load the unpatched FP32 reference model.

    This model is the accuracy baseline, so it is loaded without any of
    the conversion patches in :mod:`poc_qwen.patches` and is never
    traced.

    Args:
        model_dir: Local HuggingFace-format model directory.

    Returns:
        The model in eval mode, FP32 precision, with caching disabled.
    """
    model = AutoModel.from_pretrained(
        str(model_dir), attn_implementation="sdpa", torch_dtype=torch.float32
    )
    model.config.use_cache = False
    return model.eval()


def encode_reference(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    seq_len: int,
    prompt: str = "",
    pool: str = "auto",
) -> np.ndarray:
    """Compute reference embeddings one row at a time.

    The plain 2-D attention mask is passed through, so the framework
    builds its own causal mask: the reference must not share the 4-D mask
    path under test. Rows are encoded one at a time to bound peak memory.

    Args:
        model: Reference model from :func:`load_reference_model` (any
            HuggingFace model exposing ``config`` and returning the last
            hidden state first).
        tokenizer: Tokenizer from :func:`load_tokenizer`.
        texts: Input texts, without any instruction prefix.
        seq_len: Fixed sequence length used for tokenization.
        prompt: Optional instruction prefix; see :func:`tokenize_batch`.
        pool: ``"left"`` for :func:`last_token_pool_left`, ``"gather"``
            for :func:`last_token_pool_gather`, or ``"auto"`` to pick
            from the tokenizer's padding side.

    Returns:
        Embeddings of shape ``(len(texts), hidden_size)``, dtype
        float32, **not** L2-normalized.

    Raises:
        ValueError: If ``pool`` is not one of the modes above.
    """
    if pool == "auto":
        pool = "left" if getattr(tokenizer, "padding_side", "right") == "left" else "gather"
    if pool not in {"left", "gather"}:
        raise ValueError(f"unknown pooling mode: {pool!r} (expected 'left', 'gather' or 'auto')")

    hidden_size = model.config.hidden_size
    embeddings = np.empty((len(texts), hidden_size), dtype=np.float32)
    if not texts:
        return embeddings

    batch = tokenize_batch(tokenizer, texts, seq_len, prompt=prompt)
    with torch.no_grad():
        for i in range(len(texts)):
            # Embedding lookups need int64 indices, while tokenize_batch
            # returns int32 for Core ML compatibility.
            input_ids = torch.from_numpy(batch["input_ids"][i : i + 1]).long()
            attention_mask = torch.from_numpy(batch["attention_mask"][i : i + 1]).long()
            hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            if pool == "left":
                pooled = last_token_pool_left(hidden)
            else:
                pooled = last_token_pool_gather(hidden, attention_mask)
            embeddings[i] = pooled.reshape(-1).numpy().astype(np.float32)
    return embeddings


def cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally shaped arrays.

    Args:
        a: Array of shape ``(N, D)``.
        b: Array of shape ``(N, D)``.

    Returns:
        Similarities of shape ``(N,)``. Zero-norm rows are protected with
        a small epsilon so they yield 0.0 instead of NaN.
    """
    dot = np.sum(a * b, axis=1)
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return dot / denom


def sanity_text_sets() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the shared per-language sanity text sets.

    Returns:
        Tuple of ``(language code, texts)`` pairs, reusing the same
        fixtures the rest of the project verifies its models with so
        results stay comparable across backends.
    """
    return SANITY_TEXT_SETS


def build_versions_info() -> dict[str, str]:
    """Collect the library and runtime versions recorded in result files.

    Returns:
        Version strings keyed by component. ``coremltools`` is reported
        as ``"unavailable"`` when it cannot be imported, so the tracing
        and accuracy scripts still record their environment on a machine
        without it.
    """
    try:
        import coremltools as ct

        coremltools_version = ct.__version__
    except ImportError:
        coremltools_version = "unavailable"
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "coremltools": coremltools_version,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def write_result_json(path: Path, payload: dict) -> None:
    """Write a result payload as UTF-8 JSON, creating parent directories.

    Every result file records the environment it was produced on, so a
    payload without a ``versions`` key gets one from
    :func:`build_versions_info`.

    Args:
        path: Destination file; missing parent directories are created.
        payload: JSON-serializable mapping. It is not modified: the
            ``versions`` key is added to a shallow copy.
    """
    record = dict(payload)
    record.setdefault("versions", build_versions_info())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
