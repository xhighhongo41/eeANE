"""Shared utilities for the eeANE PoC scripts.

This module is the single source of truth for tokenization, the PyTorch
FP32 baseline embedding computation, and deterministic test corpus/prefix
loading. The Core ML conversion script (``poc/convert_embedding.py``) and
the verification scripts (``poc/verify_accuracy.py``,
``poc/benchmark_latency.py``) all depend on :func:`mean_pool` so that the
in-graph pooling and the manual PyTorch baseline compute the identical
formula (see v0.1実装計画.md §4.2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Local model directory, overridable via the EEANE_MODEL_DIR env var.
DEFAULT_MODEL_DIR: Path = Path(
    os.environ.get("EEANE_MODEL_DIR", str(_REPO_ROOT / "models" / "ruri-v3-310m"))
)

# Local reranker model directory, overridable via the EEANE_RERANKER_DIR env
# var (v0.2実装計画.md §4.4).
DEFAULT_RERANKER_DIR: Path = Path(
    os.environ.get("EEANE_RERANKER_DIR", str(_REPO_ROOT / "models" / "ruri-v3-reranker-310m"))
)

# Hand-written reranker query set (v0.2実装計画.md §4.5), consumed by
# verify_reranker_accuracy.py and benchmark_latency.py.
RERANK_QUERIES_PATH: Path = _REPO_ROOT / "testdata" / "rerank_queries.json"

# Fixed Aozora Bunko test corpus directory (populated by T3).
CORPUS_DIR: Path = _REPO_ROOT / "testdata" / "corpus"

# ruri-v3 instruction prefixes used to prepend to raw sentences.
PREFIXES: dict[str, str] = {
    "none": "",
    "topic": "トピック: ",
    "query": "検索クエリ: ",
    "document": "検索文書: ",
}

# Hand-written Japanese sentences covering varied topics (science, history,
# technology, daily life), used to build the prefix-confirmation test set
# (§4.6). Kept as a fixed constant so results are reproducible across runs.
PREFIX_TEST_SENTENCES: list[str] = [
    # Questions (4)
    "光合成はどのようにして太陽の光を化学エネルギーに変換しているのですか。",
    "明治維新によって日本の政治体制はどのように変化したのでしょうか。",
    "ニューラルネットワークの学習がうまく収束しないのはなぜですか。",
    "毎朝コーヒーを飲むと眠気が覚めるのはどうしてなのでしょうか。",
    # Statements (4)
    "光合成は植物が太陽光のエネルギーを使って栄養分を作り出す仕組みである。",
    "明治維新は江戸幕府が倒れて近代的な中央集権国家が誕生した出来事だ。",
    "ニューラルネットワークは層を重ねることで複雑な関数を近似できる。",
    "毎朝のコーヒーにはカフェインが含まれており集中力を高めてくれる。",
]


def load_tokenizer(model_dir: Path) -> PreTrainedTokenizerBase:
    """Load the SentencePiece tokenizer bundled with the ruri model.

    Args:
        model_dir: Path to the local HuggingFace-format model directory.

    Returns:
        The tokenizer loaded via ``AutoTokenizer.from_pretrained``.
    """
    return AutoTokenizer.from_pretrained(model_dir)


def tokenize_batch(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize texts into fixed-shape int32 arrays for Core ML input.

    Args:
        tokenizer: Tokenizer returned by :func:`load_tokenizer`.
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


def load_torch_model(model_dir: Path, attn: str = "sdpa") -> PreTrainedModel:
    """Load the PyTorch FP32 model used as the accuracy baseline.

    Args:
        model_dir: Path to the local HuggingFace-format model directory.
        attn: Attention implementation to request (e.g. ``"sdpa"``, ``"eager"``).

    Returns:
        The model in eval mode, FP32 precision.
    """
    model = AutoModel.from_pretrained(
        model_dir, attn_implementation=attn, torch_dtype=torch.float32
    )
    return model.eval()


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over the sequence dimension.

    Single source of truth shared by the Core ML conversion wrapper
    (EmbeddingWrapper) and the PyTorch baseline (encode_pytorch). The
    formula must stay identical to v0.1実装計画.md §4.2; do not change it
    without updating both consumers.

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


def encode_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    seq_len: int,
) -> np.ndarray:
    """Compute FP32 baseline embeddings with a batch-size-1 loop.

    Args:
        model: Model returned by :func:`load_torch_model`.
        tokenizer: Tokenizer returned by :func:`load_tokenizer`.
        texts: Input sentences.
        seq_len: Fixed sequence length used for tokenization.

    Returns:
        Embeddings array of shape (len(texts), 768), dtype float32.
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


def cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped arrays.

    Args:
        a: Array of shape (N, D).
        b: Array of shape (N, D).

    Returns:
        Cosine similarities, shape (N,). Zero-norm rows are protected with
        a small epsilon to avoid division by zero.
    """
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denom = np.maximum(norm_a * norm_b, 1e-12)
    return dot / denom


def load_corpus_paragraphs(max_kokoro: int = 30, min_chars: int = 40) -> list[str]:
    """Load a deterministic paragraph list from the fixed Aozora corpus.

    Splits all three works on blank lines, strips each paragraph, and
    keeps only paragraphs with at least ``min_chars`` characters. All
    filtered paragraphs are kept for kumonoito and sangetsuki, while only
    the first ``max_kokoro`` filtered paragraphs are kept for kokoro. File
    order is fixed: kumonoito -> sangetsuki -> kokoro.

    Args:
        max_kokoro: Maximum number of leading paragraphs to take from
            kokoro.txt (the other two works contribute all paragraphs).
        min_chars: Minimum paragraph length (in characters) to keep.

    Returns:
        Paragraphs in file order (kumonoito, sangetsuki, kokoro).
    """
    paragraphs: list[str] = []
    file_limits = [
        (CORPUS_DIR / "kumonoito.txt", None),
        (CORPUS_DIR / "sangetsuki.txt", None),
        (CORPUS_DIR / "kokoro.txt", max_kokoro),
    ]
    for path, limit in file_limits:
        text = path.read_text(encoding="utf-8")
        blocks = _split_paragraphs(text)
        blocks = [block for block in blocks if len(block) >= min_chars]
        if limit is not None:
            blocks = blocks[:limit]
        paragraphs.extend(blocks)
    return paragraphs


def _split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines.

    Consecutive blank lines (including whitespace-only lines) are treated
    as a paragraph separator. Uses ``str.splitlines`` to be agnostic to
    the line-ending style.

    Args:
        text: Full text to split.

    Returns:
        Stripped paragraph strings (empty paragraphs are excluded).
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def load_reranker_torch_model(model_dir: Path, attn: str = "sdpa") -> PreTrainedModel:
    """Load the PyTorch FP32 reranker model used as the accuracy baseline.

    Args:
        model_dir: Path to the local HuggingFace-format model directory.
        attn: Attention implementation to request (e.g. ``"sdpa"``, ``"eager"``).

    Returns:
        The ``ModernBertForSequenceClassification`` model in eval mode, FP32
        precision (see v0.2実装計画.md §2.2).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, attn_implementation=attn, torch_dtype=torch.float32
    )
    return model.eval()


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], seq_len: int
) -> dict[str, np.ndarray]:
    """Tokenize (query, document) pairs into fixed-shape int32 arrays.

    Delegates to the tokenizer's built-in pair encoding
    (``tokenizer(queries, documents, ...)``) so that the
    ``<s> query </s> <s> document </s>`` template (v0.2実装計画.md §2.2) is
    produced by the tokenizer's post_processor rather than reimplemented
    here. ``truncation=True`` uses the tokenizer's default
    ``longest_first`` strategy across both sequences (v0.2実装計画.md
    §4.4). Any key other than ``input_ids``/``attention_mask`` returned by
    the tokenizer (e.g. ``token_type_ids``) is discarded, since the
    reranker forward only accepts those two inputs.

    Args:
        tokenizer: Tokenizer returned by :func:`load_tokenizer`.
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


def score_pytorch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    seq_len: int,
) -> np.ndarray:
    """Compute FP32 baseline raw reranker logits with a batch-size-1 loop.

    Args:
        model: Model returned by :func:`load_reranker_torch_model`.
        tokenizer: Tokenizer returned by :func:`load_tokenizer`.
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the Spearman rank correlation between two score arrays.

    Uses double-argsort ordinal ranks followed by Pearson correlation
    (v0.2実装計画.md §4.4 reference implementation). Continuous score
    inputs are assumed: tied values are broken by array position
    (ordinal ranks) rather than averaged, so this is not the exact
    tie-corrected Spearman formula. scipy is intentionally not used (no
    new dependency).

    Args:
        a: First score array, shape (N,).
        b: Second score array, shape (N,).

    Returns:
        Spearman rank correlation coefficient in [-1, 1].
    """
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / max(denom, 1e-12))


def load_rerank_queries() -> list[dict]:
    """Load the hand-written reranker query set (v0.2実装計画.md §4.5).

    Returns:
        List of dicts with keys ``id``, ``query``, and ``source_work``, in
        the order stored on disk at :data:`RERANK_QUERIES_PATH`.
    """
    with RERANK_QUERIES_PATH.open(encoding="utf-8") as f:
        return json.load(f)
