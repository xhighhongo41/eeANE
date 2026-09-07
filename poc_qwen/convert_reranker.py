"""Convert the generative Qwen3 reranker into a Core ML program.

``Qwen/Qwen3-Reranker-0.6B`` is not a classifier: it is a causal language
model that is asked, through a fixed chat prompt, whether a document
satisfies a query, and its relevance signal is the logit of the token
``"yes"`` against the logit of ``"no"`` at the final position. The
checkpoint ties its output projection to the input embedding matrix, so
those two logits are two rows of the embedding table; the whole
vocabulary-wide projection can therefore be replaced by a ``(2, hidden)``
linear layer baked into the graph, which keeps the exported program the
same size as the plain backbone.

The pipeline mirrors the embedding converter: load the backbone with eager
attention -> install the numerically equivalent conversion patches -> wrap
it so the 4-D attention mask, the last-token pooling and the yes/no
projection all happen inside the graph -> ``torch.jit.trace`` ->
``ct.convert`` to an ``mlprogram`` with fixed ``(B, S)`` int32 inputs ->
``.mlpackage`` -> ``xcrun coremlcompiler`` -> ``.mlmodelc`` -> sanity check
on CPU_AND_NE against an unpatched FP32 reference. A JSON file recording
the options, the prompt layout, the applied patches, the per-stage timings
and the sanity results is written next to the artifacts.

The graph emits the two raw logits; the softmax that turns them into a
relevance probability is applied outside the model, matching the
convention the other rerankers in this project follow of exporting raw
scores and leaving the normalization to the consumer.

The sanity check never aborts the run: a conversion whose accuracy is bad
is a measurement worth keeping, so the numbers are always written to the
JSON file and the failure is reported through the exit code (1) instead.

Usage:
    uv run python poc_qwen/convert_reranker.py --seq-len 512 --batch 1
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, PreTrainedTokenizerBase

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    # Allow `python poc_qwen/convert_reranker.py` to import the package.
    sys.path.insert(0, str(_REPO_ROOT))

from poc_qwen import common, convert, patches  # noqa: E402

# Hub id of the reranker this script targets.
DEFAULT_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

# Name requested for the single graph output.
OUTPUT_NAME = "yes_no_logits"

# Default destination for the artifacts and their metadata file.
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "models" / "compiled" / "qwen3-reranker-0.6b"

# Grouped-query expansion strategy used unless overridden; artifacts built
# with any other strategy get it appended to their file name.
DEFAULT_REPEAT_KV_MODE = "repeat_interleave"

# Number of pipeline stages reported in the progress output.
_TOTAL_STAGES = 8

# Fixed chat prompt the model was trained with. The control tokens are
# spelled out as plain text here, so both halves must be tokenized with
# ``add_special_tokens=False`` (see :func:`encode_prompt_affixes`).
PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Task description inserted into the prompt when none is given.
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Answer tokens whose logits form the graph output, in the row order of
# the projection weight: row 0 is "no", row 1 is "yes".
NO_TOKEN = "no"
YES_TOKEN = "yes"

# Token ids the two answer tokens are expected to resolve to. They are
# still looked up through the tokenizer at runtime; these values only
# guard against silently scoring the wrong rows of the embedding matrix
# if the script is ever pointed at a different checkpoint.
EXPECTED_NO_TOKEN_ID = 2152
EXPECTED_YES_TOKEN_ID = 9693

# Fixed (query, document) pair used as the example input for
# torch.jit.trace. Only its shape reaches the graph, so the content is
# irrelevant; a constant pair keeps tracing independent of the test data.
TRACE_EXAMPLE_PAIR: tuple[str, str] = (
    "What is the capital of France?",
    "Paris is the capital and the most populous city of France.",
)

# Fixed Aozora Bunko test corpus used to build the sanity pairs.
CORPUS_DIR = _REPO_ROOT / "testdata" / "corpus"
RERANK_QUERIES_PATH = _REPO_ROOT / "testdata" / "rerank_queries.json"

# Works of the corpus, in the fixed order the sanity pairs are built in.
CORPUS_WORKS: tuple[str, ...] = ("kumonoito", "sangetsuki", "kokoro")

# Shortest paragraph accepted as a sanity document. Very short paragraphs
# are section headings and one-line fragments, which carry too little
# signal to rank meaningfully.
SANITY_MIN_PARAGRAPH_CHARS = 40

# Per query: leading paragraphs of the query's own work used as relevant
# documents, plus the leading paragraph of every other work as an
# irrelevant one. With three works this yields 3 * (3 + 2) = 15 pairs.
SANITY_RELATED_DOCS = 3

# Largest accepted absolute difference between the "yes" probability of
# the compiled model and of the FP32 reference.
SCORE_ABS_TOLERANCE = 0.02

# Reference score gap below which a reordering is not counted as a
# ranking mismatch. FP16 rounding of a probability in [0, 1] is on the
# order of 1e-3 near 1.0, so two documents whose reference scores differ
# by less than this have no meaningful order to preserve in the first
# place.
RANK_TIE_TOLERANCE = 0.001


class YesNoRerankerWrapper(torch.nn.Module):
    """Wraps the backbone and emits the yes/no logits of the final position.

    The attention mask, the pooling and the two-way output projection are
    all computed inside the wrapper, so the exported program takes exactly
    the two int32 tensors Core ML feeds it and returns one pair of logits
    per row, with no host-side pre- or post-processing left to
    reimplement beyond the softmax.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        yes_no_weight: torch.Tensor,
        fill_value: float = common.MASK_FILL_VALUE,
    ) -> None:
        """Store the backbone, the output projection and the mask setting.

        Args:
            model: Backbone returning the last hidden state first, loaded
                with eager attention, ``use_cache = False`` and
                ``return_dict = False``.
            yes_no_weight: Output projection of shape ``(2, hidden_size)``
                whose **row 0 scores the "no" token and row 1 the "yes"
                token**. It is registered as a buffer so tracing bakes it
                into the graph as a constant.
            fill_value: Additive value for masked positions; see
                ``poc_qwen.common.MASK_FILL_VALUE``.

        Raises:
            ValueError: If ``yes_no_weight`` is not a ``(2, hidden_size)``
                matrix.
        """
        super().__init__()
        if yes_no_weight.dim() != 2 or yes_no_weight.shape[0] != 2:
            raise ValueError(
                f"yes_no_weight must have shape (2, hidden_size), got {tuple(yes_no_weight.shape)}"
            )
        self.model = model
        self.fill_value = fill_value
        self.register_buffer("yes_no_weight", yes_no_weight.detach().to(torch.float32).clone())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Score a batch of prompt-formatted token id rows.

        Args:
            input_ids: Token ids of shape ``(B, S)``, left padded so the
                final position always holds a real token.
            attention_mask: 2-D mask of shape ``(B, S)``; non-zero marks a
                real token.

        Returns:
            Logits of shape ``(B, 2)``: column 0 is the "no" logit and
            column 1 the "yes" logit. They are **not** normalized;
            :func:`yes_probability` turns them into a relevance score.
        """
        mask4d = common.build_causal_padding_mask(attention_mask, self.fill_value)
        hidden = self.model(input_ids=input_ids, attention_mask=mask4d)[0]
        pooled = common.last_token_pool_left(hidden)
        return torch.nn.functional.linear(pooled, self.yes_no_weight)


def resolve_yes_no_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    """Look up the answer token ids and verify them against the expected ones.

    The ids are read from the tokenizer rather than hardcoded, but they
    are also checked: the converted graph bakes exactly two rows of the
    embedding matrix into a constant, so a checkpoint whose vocabulary
    places "yes"/"no" elsewhere would produce an artifact that silently
    scores unrelated tokens.

    Args:
        tokenizer: Tokenizer of the model being converted.

    Returns:
        ``(no_token_id, yes_token_id)``, in the row order of the output
        projection built by :func:`build_yes_no_weight`.

    Raises:
        RuntimeError: If either token is missing from the vocabulary or
            resolves to an unexpected id.
    """
    # A tokenizer that owns an unknown-token id answers with it instead of
    # failing, so an absent answer token has to be recognized by that id.
    unk_id = getattr(tokenizer, "unk_token_id", None)
    resolved: dict[str, int] = {}
    for token in (NO_TOKEN, YES_TOKEN):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unk_id is not None and token_id == unk_id):
            raise RuntimeError(f"the tokenizer has no id for the answer token {token!r}")
        resolved[token] = int(token_id)
    expected = {NO_TOKEN: EXPECTED_NO_TOKEN_ID, YES_TOKEN: EXPECTED_YES_TOKEN_ID}
    if resolved != expected:
        raise RuntimeError(
            f"unexpected answer token ids {resolved}, expected {expected}; this script bakes the "
            "matching two rows of the embedding matrix into the graph and must not be used with a "
            "checkpoint whose vocabulary differs"
        )
    return resolved[NO_TOKEN], resolved[YES_TOKEN]


def build_yes_no_weight(
    model: torch.nn.Module, no_token_id: int, yes_token_id: int
) -> torch.Tensor:
    """Extract the two-row output projection from the input embeddings.

    The checkpoint sets ``tie_word_embeddings``, so the language modelling
    head is the transpose of the input embedding matrix. Only two of its
    rows are ever needed, and taking just those two keeps the exported
    graph free of the vocabulary-sized matrix multiply.

    Args:
        model: Backbone exposing ``get_input_embeddings()``.
        no_token_id: Vocabulary id of the "no" token (becomes row 0).
        yes_token_id: Vocabulary id of the "yes" token (becomes row 1).

    Returns:
        Float32 tensor of shape ``(2, hidden_size)``, detached from the
        model's parameters.

    Raises:
        RuntimeError: If the model exposes no input embedding matrix, or
            if an id lies outside it.
    """
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise RuntimeError("the model exposes no input embedding matrix to tie the head to")
    vocab_size = int(weight.shape[0])
    for name, token_id in (("no", no_token_id), ("yes", yes_token_id)):
        if not 0 <= token_id < vocab_size:
            raise RuntimeError(
                f"the {name!r} token id {token_id} is outside the vocabulary of size {vocab_size}"
            )
    rows = torch.tensor([no_token_id, yes_token_id], dtype=torch.long)
    return weight.detach().index_select(0, rows).to(torch.float32).clone()


def encode_prompt_affixes(tokenizer: PreTrainedTokenizerBase) -> tuple[list[int], list[int]]:
    """Tokenize the fixed prompt prefix and suffix.

    ``add_special_tokens=False`` is required: both halves already spell
    out the chat control tokens (``<|im_start|>`` and friends) as text, so
    letting the tokenizer contribute its own would duplicate them and
    change the prompt the model was trained on.

    Args:
        tokenizer: Tokenizer of the model being converted.

    Returns:
        ``(prefix_ids, suffix_ids)``.
    """
    prefix_ids = list(tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False))
    return prefix_ids, suffix_ids


def format_body(instruction: str, query: str, document: str) -> str:
    """Render the variable part of the prompt for one pair.

    Args:
        instruction: Task description shown to the model.
        query: Search query.
        document: Candidate document.

    Returns:
        The prompt body that goes between the fixed prefix and suffix.
    """
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def resolve_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """Pick the id written into padded positions.

    Padded positions are masked out of every attention row, so the id only
    has to be a valid vocabulary index; the tokenizer's own pad token is
    preferred and the end-of-text token is the usual fallback for decoder
    checkpoints that ship without one.

    Args:
        tokenizer: Tokenizer of the model being converted.

    Returns:
        The padding token id.

    Raises:
        ValueError: If the tokenizer defines neither token.
    """
    for candidate in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if candidate is not None:
            return int(candidate)
    raise ValueError("the tokenizer defines neither a pad token nor an eos token to pad with")


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[tuple[str, str]],
    seq_len: int,
    instruction: str,
) -> dict[str, np.ndarray]:
    """Build fixed-shape int32 inputs for a list of (query, document) pairs.

    Only the body is truncated: the prefix carries the system message the
    model was trained with and the suffix carries the assistant turn whose
    final position is read out, so dropping tokens from either would
    change what the last position means.

    Padding goes on the **left**, which is what makes the pooled position
    a static ``hidden[:, -1, :]`` slice inside the traced graph.

    Args:
        tokenizer: Tokenizer of the model being converted.
        pairs: ``(query, document)`` pairs to score.
        seq_len: Fixed sequence length ``S``.
        instruction: Task description inserted into every prompt.

    Returns:
        Dict with ``input_ids`` and ``attention_mask``, each of shape
        ``(len(pairs), seq_len)`` and dtype ``np.int32``.

    Raises:
        ValueError: If ``seq_len`` leaves no room for the body.
    """
    prefix_ids, suffix_ids = encode_prompt_affixes(tokenizer)
    budget = seq_len - len(prefix_ids) - len(suffix_ids)
    if budget < 1:
        raise ValueError(
            f"seq_len {seq_len} is too short: the prompt needs {len(prefix_ids)} prefix and "
            f"{len(suffix_ids)} suffix tokens, leaving no room for the query and the document"
        )
    pad_token_id = resolve_pad_token_id(tokenizer)
    input_ids = np.full((len(pairs), seq_len), pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((len(pairs), seq_len), dtype=np.int32)
    for row, (query, document) in enumerate(pairs):
        body = format_body(instruction, query, document)
        body_ids = list(tokenizer.encode(body, add_special_tokens=False))[:budget]
        ids = prefix_ids + body_ids + suffix_ids
        input_ids[row, seq_len - len(ids) :] = ids
        attention_mask[row, seq_len - len(ids) :] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def yes_probability(logits: np.ndarray) -> np.ndarray:
    """Turn yes/no logits into the relevance probability of "yes".

    This is ``exp(log_softmax(logits))[:, 1]``, written as a shifted
    softmax so that large logits cannot overflow.

    Args:
        logits: Array of shape ``(N, 2)``; column 0 is "no", column 1 is
            "yes".

    Returns:
        Probabilities of shape ``(N,)``, in ``[0, 1]``.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated[:, 1] / exponentiated.sum(axis=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert a generative Qwen3 reranker to a Core ML program."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hub model id or local HuggingFace-format directory (default: %(default)s).",
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Fixed sequence length S (default: %(default)s)."
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="Fixed batch size B (default: %(default)s)."
    )
    parser.add_argument(
        "--precision",
        choices=sorted(convert.PRECISIONS),
        default="fp16",
        help="Compute precision of the converted program (default: %(default)s).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(convert.TARGETS),
        default="macos13",
        help="Minimum deployment target (default: %(default)s).",
    )
    parser.add_argument(
        "--mask-fill-value",
        type=float,
        default=common.MASK_FILL_VALUE,
        help=(
            "Additive value written into masked attention positions; must be negative "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rmsnorm-mode",
        choices=list(patches.RMSNORM_MODES),
        default="upstream",
        help=(
            "RMSNorm variant: 'upstream' keeps the original formula, 'scaled' divides the "
            "input before squaring so the sum of squares cannot overflow FP16 "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--rmsnorm-scale",
        type=float,
        default=64.0,
        help="Positive divisor used by --rmsnorm-mode scaled (default: %(default)s).",
    )
    parser.add_argument(
        "--repeat-kv-mode",
        choices=list(patches.REPEAT_KV_MODES),
        default=DEFAULT_REPEAT_KV_MODE,
        help="Operator used to expand grouped-query key/value heads (default: %(default)s).",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help=(
            "Task description inserted into every prompt. It is tokenized on the host and is "
            "therefore not baked into the artifact (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory the artifacts and the metadata file are written to (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-mlpackage",
        action="store_true",
        help="Keep the intermediate .mlpackage instead of deleting it after compilation.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Convert only, without loading the FP32 reference model for the sanity check.",
    )
    return parser.parse_args(argv)


def build_stem(args: argparse.Namespace) -> str:
    """Build the base name shared by every artifact of one conversion.

    The base name is ``s{S}_b{B}_{precision}_{target}``; every option that
    changes what the artifact computes adds a suffix, so two conversions
    that differ in any way can never overwrite each other's files. The
    instruction is deliberately not part of the name: it is host-side
    prompt text and never reaches the graph.

    Args:
        args: Parsed command line arguments.

    Returns:
        The artifact base name.
    """
    stem = f"s{args.seq_len}_b{args.batch}_{args.precision}_{args.target}"
    if args.rmsnorm_mode != "upstream":
        stem += f"_rms{args.rmsnorm_scale:g}"
    if args.mask_fill_value != common.MASK_FILL_VALUE:
        stem += f"_fill{args.mask_fill_value:g}"
    if args.repeat_kv_mode != DEFAULT_REPEAT_KV_MODE:
        stem += f"_{args.repeat_kv_mode.replace('_', '')}"
    return stem


def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on blank lines.

    Args:
        text: Full text to split.

    Returns:
        Stripped paragraph strings; empty paragraphs are excluded.
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


def load_work_paragraphs(work: str, min_chars: int = SANITY_MIN_PARAGRAPH_CHARS) -> list[str]:
    """Load the candidate paragraphs of one corpus work.

    Args:
        work: Stem of the file under ``testdata/corpus``.
        min_chars: Shortest paragraph kept.

    Returns:
        Paragraphs in file order.

    Raises:
        RuntimeError: If the corpus file is missing.
    """
    path = CORPUS_DIR / f"{work}.txt"
    if not path.is_file():
        raise RuntimeError(f"missing corpus file for the sanity check: {path}")
    return [
        block
        for block in split_paragraphs(path.read_text(encoding="utf-8"))
        if len(block) >= min_chars
    ]


def build_sanity_pairs() -> list[dict[str, Any]]:
    """Build the fixed (query, document) pairs the sanity check scores.

    One query per corpus work is taken from the hand-written query set,
    and each is paired with the leading paragraphs of its own work
    (relevant) plus the leading paragraph of every other work
    (irrelevant). Everything is positional, so the same pairs are produced
    on every run without needing a random seed.

    Returns:
        Pair records with the query, the document, their labels and
        whether the document comes from the query's own work.

    Raises:
        RuntimeError: If the query set or the corpus cannot supply the
            fixed selection.
    """
    if not RERANK_QUERIES_PATH.is_file():
        raise RuntimeError(f"missing query set for the sanity check: {RERANK_QUERIES_PATH}")
    queries = json.loads(RERANK_QUERIES_PATH.read_text(encoding="utf-8"))
    paragraphs = {work: load_work_paragraphs(work) for work in CORPUS_WORKS}
    for work, blocks in paragraphs.items():
        if len(blocks) < SANITY_RELATED_DOCS:
            raise RuntimeError(
                f"corpus work {work!r} has only {len(blocks)} usable paragraphs, "
                f"{SANITY_RELATED_DOCS} are needed"
            )

    pairs: list[dict[str, Any]] = []
    for work in CORPUS_WORKS:
        query = next((item for item in queries if item.get("source_work") == work), None)
        if query is None:
            raise RuntimeError(f"the query set holds no query for the corpus work {work!r}")
        documents = [
            (f"{work}#{index}", paragraphs[work][index], True)
            for index in range(SANITY_RELATED_DOCS)
        ]
        documents.extend(
            (f"{other}#0", paragraphs[other][0], False) for other in CORPUS_WORKS if other != work
        )
        for label, document, related in documents:
            pairs.append(
                {
                    "query_id": str(query["id"]),
                    "query": str(query["query"]),
                    "document_label": label,
                    "document": document,
                    "related": related,
                }
            )
    return pairs


def _fill_batch(rows: np.ndarray, batch_size: int) -> np.ndarray:
    """Pad a partial chunk of tokenized rows up to the fixed batch size.

    Args:
        rows: Chunk of shape ``(n, S)`` with ``n <= batch_size``.
        batch_size: Fixed batch size ``B`` of the converted model.

    Returns:
        Array of shape ``(batch_size, S)``; ``rows`` itself when already
        full.
    """
    missing = batch_size - rows.shape[0]
    if missing <= 0:
        return rows
    # The chunk's own last row is repeated rather than some filler text:
    # the extra outputs are discarded anyway, and duplicating a real row
    # guarantees no padded row is fully masked (which would make the
    # attention softmax degenerate and could produce NaN).
    return np.concatenate([rows, np.repeat(rows[-1:], missing, axis=0)], axis=0)


def predict_pairs(
    compiled: ct.models.CompiledMLModel,
    tokens: dict[str, np.ndarray],
    batch_size: int,
    output_key: str,
) -> np.ndarray:
    """Score tokenized rows with a model whose batch size is fixed.

    Args:
        compiled: Loaded compiled model.
        tokens: Tokenized rows, each value of shape ``(N, S)``.
        batch_size: Fixed batch size ``B`` of the compiled model.
        output_key: Output name resolved from the model specification.

    Returns:
        Logits of shape ``(N, 2)``, dtype float32.

    Raises:
        ValueError: If ``tokens`` holds no rows, or if the model does not
            return two values per row.
    """
    n_rows = int(tokens["input_ids"].shape[0])
    if n_rows == 0:
        raise ValueError("no rows to predict")
    rows: list[np.ndarray] = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        prediction = compiled.predict(
            {key: _fill_batch(tokens[key][start:end], batch_size) for key in tokens}
        )
        values = np.asarray(prediction[output_key], dtype=np.float32).reshape(batch_size, -1)
        if values.shape[1] != 2:
            raise ValueError(
                f"output {output_key!r} has width {values.shape[1]}, expected 2 (no, yes)"
            )
        rows.extend(values[: end - start])
    return np.stack(rows)


def score_reference(
    model: torch.nn.Module, yes_no_weight: torch.Tensor, tokens: dict[str, np.ndarray]
) -> np.ndarray:
    """Score tokenized rows with the unpatched FP32 reference model.

    The plain 2-D attention mask is passed through, so the framework
    builds its own causal mask: the reference must not share the 4-D mask
    path under test. Rows are scored one at a time to bound peak memory.

    Args:
        model: Reference model from ``poc_qwen.common.load_reference_model``.
        yes_no_weight: Output projection of shape ``(2, hidden_size)``
            taken from the reference model itself.
        tokens: Tokenized rows, each value of shape ``(N, S)``.

    Returns:
        Logits of shape ``(N, 2)``, dtype float32.
    """
    n_rows = int(tokens["input_ids"].shape[0])
    logits = np.empty((n_rows, 2), dtype=np.float32)
    with torch.no_grad():
        for row in range(n_rows):
            # Embedding lookups need int64 indices, while tokenize_pairs
            # returns int32 for Core ML compatibility.
            input_ids = torch.from_numpy(tokens["input_ids"][row : row + 1]).long()
            attention_mask = torch.from_numpy(tokens["attention_mask"][row : row + 1]).long()
            hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            pooled = common.last_token_pool_left(hidden)
            values = torch.nn.functional.linear(pooled, yes_no_weight)
            logits[row] = values.reshape(-1).numpy().astype(np.float32)
    return logits


def check_ranking(
    labels: Sequence[str],
    coreml_scores: np.ndarray,
    reference_scores: np.ndarray,
    tolerance: float = RANK_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Compare how one query's documents are ordered by the two models.

    Every unordered document pair is inspected and counted as discordant
    when the two models disagree about which document scores higher. A
    disagreement is forgiven when the two reference scores differ by less
    than ``tolerance``, because such an order is below the rounding width
    of the FP16 arithmetic the compiled model runs in and therefore
    carries no information. Checking all pairs rather than only
    neighbours is not stricter than it sounds: if two documents are within
    ``tolerance`` of each other, every document ranked between them is
    inside that same interval, so a forgiven pair is always a run of
    adjacent near-ties.

    Args:
        labels: Document labels, one per score.
        coreml_scores: Scores from the compiled model, shape ``(n,)``.
        reference_scores: Scores from the FP32 reference, shape ``(n,)``.
        tolerance: Reference score gap below which order is meaningless.

    Returns:
        Dict with both orderings, the discordant and forgiven pair counts
        and the pass/fail flag.
    """
    coreml_order = [labels[index] for index in np.argsort(-coreml_scores, kind="stable")]
    reference_order = [labels[index] for index in np.argsort(-reference_scores, kind="stable")]
    discordant = 0
    forgiven = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            reference_gap = float(reference_scores[i] - reference_scores[j])
            coreml_gap = float(coreml_scores[i] - coreml_scores[j])
            if np.sign(reference_gap) == np.sign(coreml_gap):
                continue
            if abs(reference_gap) < tolerance:
                forgiven += 1
            else:
                discordant += 1
    return {
        "coreml_order": coreml_order,
        "reference_order": reference_order,
        "discordant_pairs": discordant,
        "forgiven_pairs": forgiven,
        "passed": discordant == 0,
    }


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Average a list of scores, tolerating an empty one.

    Args:
        values: Scores to average.

    Returns:
        The mean, or ``None`` when there is nothing to average.
    """
    return float(np.mean(values)) if values else None


def summarize_relevance(
    pairs: Sequence[dict[str, Any]], coreml_probs: np.ndarray, reference_probs: np.ndarray
) -> dict[str, Any]:
    """Compare the mean score of relevant and irrelevant documents.

    Recorded for information only: it says something about the model, not
    about the faithfulness of the conversion, so it never decides the
    verdict.

    Args:
        pairs: Pair records from :func:`build_sanity_pairs`.
        coreml_probs: "yes" probabilities of the compiled model.
        reference_probs: "yes" probabilities of the FP32 reference.

    Returns:
        Dict with the per-group means for both models and whether the
        relevant group scores higher.
    """
    record: dict[str, Any] = {}
    for name, probs in (("coreml", coreml_probs), ("reference", reference_probs)):
        related = [float(probs[i]) for i, pair in enumerate(pairs) if pair["related"]]
        unrelated = [float(probs[i]) for i, pair in enumerate(pairs) if not pair["related"]]
        related_mean = _mean_or_none(related)
        unrelated_mean = _mean_or_none(unrelated)
        record[name] = {
            "related_mean": related_mean,
            "unrelated_mean": unrelated_mean,
            "holds": bool(
                related_mean is not None
                and unrelated_mean is not None
                and related_mean > unrelated_mean
            ),
        }
    return record


def run_sanity_check(
    mlmodelc_path: Path,
    model_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    batch_size: int,
    output_key: str,
    instruction: str,
    no_token_id: int,
    yes_token_id: int,
) -> dict[str, Any]:
    """Compare the compiled model with an unpatched FP32 reference.

    The reference is a second, independently loaded model that uses sdpa
    attention, the framework's own 2-D mask handling and no conversion
    patches, so agreement between the two means the exported graph
    computes the intended function rather than merely reproducing its own
    assumptions. Its output projection is taken from its own embedding
    matrix as well, so the yes/no head is verified end to end.

    The check passes when every pair's "yes" probability is within
    :data:`SCORE_ABS_TOLERANCE` of the reference, when no query is ranked
    differently beyond the near-tie tolerance, and when every value is
    finite.

    Args:
        mlmodelc_path: Compiled model directory.
        model_dir: Local HuggingFace-format model directory.
        tokenizer: Tokenizer configured for left padding.
        seq_len: Fixed sequence length ``S``.
        batch_size: Fixed batch size ``B`` of the compiled model.
        output_key: Output name resolved from the model specification.
        instruction: Task description inserted into every prompt.
        no_token_id: Vocabulary id scored by column 0 of the output.
        yes_token_id: Vocabulary id scored by column 1 of the output.

    Returns:
        Dict with the per-pair comparison, the per-query ranking result,
        the informational relevance summary and the overall pass/fail
        flag.
    """
    pairs = build_sanity_pairs()
    tokens = tokenize_pairs(
        tokenizer,
        [(pair["query"], pair["document"]) for pair in pairs],
        seq_len,
        instruction,
    )

    compiled = ct.models.CompiledMLModel(
        str(mlmodelc_path), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    coreml_logits = predict_pairs(compiled, tokens, batch_size, output_key)
    # The compiled model is released before the FP32 reference is loaded
    # so the two never occupy memory at the same time.
    del compiled
    gc.collect()

    reference_model = common.load_reference_model(model_dir)
    reference_weight = build_yes_no_weight(reference_model, no_token_id, yes_token_id)
    reference_logits = score_reference(reference_model, reference_weight, tokens)
    del reference_model, reference_weight
    gc.collect()

    finite = bool(np.isfinite(coreml_logits).all()) and bool(np.isfinite(reference_logits).all())
    # A non-finite logit would turn the softmax into NaN and make every
    # comparison below meaningless, so the scores are only derived once
    # the raw outputs are known to be usable.
    coreml_probs = yes_probability(coreml_logits) if finite else np.full(len(pairs), np.nan)
    reference_probs = yes_probability(reference_logits) if finite else np.full(len(pairs), np.nan)
    abs_diff = np.abs(coreml_probs - reference_probs)

    pair_records = [
        {
            "query_id": pair["query_id"],
            "document_label": pair["document_label"],
            "related": pair["related"],
            "coreml_logits": [float(value) for value in coreml_logits[index]],
            "reference_logits": [float(value) for value in reference_logits[index]],
            "coreml_yes_prob": float(coreml_probs[index]),
            "reference_yes_prob": float(reference_probs[index]),
            "abs_diff": float(abs_diff[index]),
        }
        for index, pair in enumerate(pairs)
    ]
    max_abs_diff = float(np.max(abs_diff)) if len(pairs) else 0.0
    score_passed = finite and max_abs_diff <= SCORE_ABS_TOLERANCE

    ranking: list[dict[str, Any]] = []
    for query_id in dict.fromkeys(pair["query_id"] for pair in pairs):
        indices = [index for index, pair in enumerate(pairs) if pair["query_id"] == query_id]
        record = check_ranking(
            [pairs[index]["document_label"] for index in indices],
            coreml_probs[indices],
            reference_probs[indices],
        )
        ranking.append({"query_id": query_id, **record})
    ranking_passed = finite and all(record["passed"] for record in ranking)

    return {
        "compute_units": "CPU_AND_NE",
        "output_key": output_key,
        "batch_size": batch_size,
        "instruction": instruction,
        "score_abs_tolerance": SCORE_ABS_TOLERANCE,
        "rank_tie_tolerance": RANK_TIE_TOLERANCE,
        "pairs": pair_records,
        "max_abs_diff": max_abs_diff,
        "score_passed": score_passed,
        "ranking": ranking,
        "ranking_passed": ranking_passed,
        "relevance": summarize_relevance(pairs, coreml_probs, reference_probs),
        "finite": finite,
        "passed": bool(finite and score_passed and ranking_passed),
    }


def build_metadata(
    args: argparse.Namespace,
    model_dir: Path,
    patch_record: dict[str, Any],
    prompt: dict[str, Any],
    output_key: str,
    artifacts: dict[str, Any],
    timings: dict[str, float],
    sanity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the metadata record stored next to the artifacts.

    Args:
        args: Parsed command line arguments.
        model_dir: Resolved local model directory.
        patch_record: Records returned by ``patches.apply_patches``.
        prompt: Prompt layout recorded by :func:`build_prompt_record`.
        output_key: Output name resolved from the model specification.
        artifacts: Paths of the produced files.
        timings: Per-stage wall-clock seconds.
        sanity: Sanity result, or ``None`` when it was skipped.

    Returns:
        A JSON-serializable mapping.
    """
    return {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "seq_len": args.seq_len,
        "batch": args.batch,
        "precision": args.precision,
        "target": args.target,
        "mask_fill_value": args.mask_fill_value,
        "prompt": prompt,
        "patches": patch_record,
        "output_key": output_key,
        "artifacts": artifacts,
        "timings_sec": {key: round(value, 3) for key, value in timings.items()},
        "sanity": sanity,
    }


def build_prompt_record(
    instruction: str,
    seq_len: int,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
    no_token_id: int,
    yes_token_id: int,
) -> dict[str, Any]:
    """Describe the prompt layout the artifact was built for.

    Args:
        instruction: Task description inserted into every prompt.
        seq_len: Fixed sequence length ``S``.
        prefix_ids: Tokenized prompt prefix.
        suffix_ids: Tokenized prompt suffix.
        no_token_id: Vocabulary id scored by column 0 of the output.
        yes_token_id: Vocabulary id scored by column 1 of the output.

    Returns:
        A JSON-serializable mapping.
    """
    return {
        "instruction": instruction,
        "prefix_tokens": len(prefix_ids),
        "suffix_tokens": len(suffix_ids),
        "body_token_budget": seq_len - len(prefix_ids) - len(suffix_ids),
        "padding_side": "left",
        "logit_order": [NO_TOKEN, YES_TOKEN],
        "no_token_id": no_token_id,
        "yes_token_id": yes_token_id,
    }


def print_sanity_summary(sanity: dict[str, Any]) -> None:
    """Print the sanity result as a readable table.

    Args:
        sanity: Result returned by :func:`run_sanity_check`.
    """
    print(f"      output key      : {sanity['output_key']}")
    print(f"      finite outputs  : {sanity['finite']}")
    print("      yes probability : Core ML vs unpatched FP32 reference")
    print(f"        {'query':<14}{'document':<14}{'rel':<5}{'coreml':>10}{'ref':>10}{'diff':>10}")
    for record in sanity["pairs"]:
        related = "yes" if record["related"] else "no"
        print(
            f"        {record['query_id']:<14}{record['document_label']:<14}{related:<5}"
            f"{record['coreml_yes_prob']:>10.6f}{record['reference_yes_prob']:>10.6f}"
            f"{record['abs_diff']:>10.6f}"
        )
    verdict = "PASS" if sanity["score_passed"] else "fail"
    print(
        f"      max abs diff    : {sanity['max_abs_diff']:.6f} "
        f"(tolerance {sanity['score_abs_tolerance']})  {verdict}"
    )
    print(f"      ranking         : near-tie tolerance {sanity['rank_tie_tolerance']}")
    for record in sanity["ranking"]:
        verdict = "PASS" if record["passed"] else "fail"
        print(
            f"        {record['query_id']:<14}discordant {record['discordant_pairs']} "
            f"forgiven {record['forgiven_pairs']}  {verdict}"
        )
        print(f"          coreml   : {' > '.join(record['coreml_order'])}")
        print(f"          reference: {' > '.join(record['reference_order'])}")
    print("      relevance (informational, not part of the verdict):")
    for name, record in sanity["relevance"].items():
        related = record["related_mean"]
        unrelated = record["unrelated_mean"]
        related_text = "n/a" if related is None else f"{related:.6f}"
        unrelated_text = "n/a" if unrelated is None else f"{unrelated:.6f}"
        print(
            f"        {name:<10}related mean {related_text} vs unrelated mean {unrelated_text}"
            f"  holds {record['holds']}"
        )


def _validate(args: argparse.Namespace) -> None:
    """Reject option combinations that cannot produce a usable artifact.

    Args:
        args: Parsed command line arguments.

    Raises:
        SystemExit: With an explanatory message for any invalid option.
    """
    if args.seq_len <= 0:
        raise SystemExit("--seq-len must be a positive integer")
    if args.batch <= 0:
        raise SystemExit("--batch must be a positive integer")
    if not args.mask_fill_value < 0:
        # A non-negative (or NaN) fill value would leave padded and future
        # positions attendable, silently changing what the model computes.
        raise SystemExit(f"--mask-fill-value must be negative, got {args.mask_fill_value}")
    if args.rmsnorm_mode == "scaled" and not args.rmsnorm_scale > 0:
        raise SystemExit(f"--rmsnorm-scale must be positive, got {args.rmsnorm_scale}")


def main(argv: list[str] | None = None) -> int:
    """Run the full conversion pipeline.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 when the sanity check failed.
    """
    args = parse_args(argv)
    _validate(args)
    stem = build_stem(args)
    output_dir: Path = args.output_dir
    mlpackage_path = output_dir / f"{stem}.mlpackage"
    mlmodelc_path = output_dir / f"{stem}.mlmodelc"
    metadata_path = output_dir / f"{stem}.json"
    timings: dict[str, float] = {}
    started = time.perf_counter()

    print(f"[1/{_TOTAL_STAGES}] Resolving {args.model_id}")
    step = time.perf_counter()
    model_dir = common.resolve_model_dir(args.model_id)
    # Left padding is mandatory here: the graph reads the final position,
    # which only holds the last prompt token when the padding is on the
    # left.
    tokenizer = common.load_tokenizer(model_dir, padding_side="left")
    no_token_id, yes_token_id = resolve_yes_no_token_ids(tokenizer)
    prefix_ids, suffix_ids = encode_prompt_affixes(tokenizer)
    prompt = build_prompt_record(
        args.instruction, args.seq_len, prefix_ids, suffix_ids, no_token_id, yes_token_id
    )
    # Checked before the 1.2 GB backbone is loaded so an unusable sequence
    # length fails in a second rather than a minute.
    if prompt["body_token_budget"] < 1:
        raise SystemExit(
            f"--seq-len {args.seq_len} is too short: the prompt needs {len(prefix_ids)} prefix "
            f"and {len(suffix_ids)} suffix tokens, leaving no room for the query and the document"
        )
    timings["resolve"] = time.perf_counter() - step
    print(
        f"      model dir {model_dir} (prefix {prompt['prefix_tokens']} / suffix "
        f"{prompt['suffix_tokens']} tokens, no={no_token_id} yes={yes_token_id}) "
        f"[{timings['resolve']:.1f}s]"
    )

    print(f"[2/{_TOTAL_STAGES}] Loading the backbone (eager attention, fp32)")
    step = time.perf_counter()
    # Eager attention is mandatory: the sdpa path calls into a separate
    # implementation that never reaches the patched repeat_kv. Caching is
    # off so no cache update enters the graph, and return_dict is off so
    # tracing sees a plain tuple output.
    model = AutoModel.from_pretrained(
        str(model_dir), attn_implementation="eager", dtype=torch.float32
    ).eval()
    model.config.use_cache = False
    model.config.return_dict = False
    timings["load"] = time.perf_counter() - step
    print(f"      loaded [{timings['load']:.1f}s]")

    print(f"[3/{_TOTAL_STAGES}] Applying conversion patches")
    step = time.perf_counter()
    patch_record = patches.apply_patches(
        model,
        repeat_kv_mode=args.repeat_kv_mode,
        rmsnorm_mode=args.rmsnorm_mode,
        rmsnorm_scale=args.rmsnorm_scale,
    )
    timings["patch"] = time.perf_counter() - step
    print(
        f"      repeat_kv={args.repeat_kv_mode} rmsnorm={args.rmsnorm_mode} "
        f"({patch_record['rmsnorm']['patched_modules']} modules) [{timings['patch']:.1f}s]"
    )

    print(f"[4/{_TOTAL_STAGES}] Tracing at ({args.batch}, {args.seq_len})")
    step = time.perf_counter()
    yes_no_weight = build_yes_no_weight(model, no_token_id, yes_token_id)
    wrapper = YesNoRerankerWrapper(model, yes_no_weight, fill_value=args.mask_fill_value).eval()
    # One example pair replicated to B rows, so the traced graph already
    # carries the target batch size.
    example = tokenize_pairs(
        tokenizer, [TRACE_EXAMPLE_PAIR] * args.batch, args.seq_len, args.instruction
    )
    traced = convert.trace_model(wrapper, example)
    timings["trace"] = time.perf_counter() - step
    print(f"      traced [{timings['trace']:.1f}s]")

    print(f"[5/{_TOTAL_STAGES}] Converting to mlprogram ({args.precision}, {args.target})")
    step = time.perf_counter()
    mlmodel = convert.convert_model(
        traced, args.seq_len, args.precision, args.target, OUTPUT_NAME, batch_size=args.batch
    )
    output_key = convert.resolve_output_key(mlmodel, OUTPUT_NAME)
    timings["convert"] = time.perf_counter() - step
    # The FP32 backbone is about 2.4 GB for the target model; it is
    # released as soon as the converted program exists so that it never
    # coexists with the artifact or with the FP32 reference model.
    del traced, wrapper, model, yes_no_weight
    gc.collect()
    print(f"      output key {output_key} [{timings['convert']:.1f}s]")

    print(f"[6/{_TOTAL_STAGES}] Saving {mlpackage_path}")
    step = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if mlpackage_path.exists():
        shutil.rmtree(mlpackage_path)
    mlmodel.save(str(mlpackage_path))
    del mlmodel
    gc.collect()
    timings["save"] = time.perf_counter() - step
    print(f"      saved [{timings['save']:.1f}s]")

    print(f"[7/{_TOTAL_STAGES}] Compiling to {mlmodelc_path}")
    step = time.perf_counter()
    convert.compile_model(mlpackage_path, mlmodelc_path)
    if not args.keep_mlpackage:
        shutil.rmtree(mlpackage_path)
    timings["compile"] = time.perf_counter() - step
    print(f"      compiled [{timings['compile']:.1f}s]")

    sanity: dict[str, Any] | None = None
    if args.skip_sanity:
        print(f"[8/{_TOTAL_STAGES}] Sanity check skipped (--skip-sanity)")
    else:
        print(f"[8/{_TOTAL_STAGES}] Sanity check on CPU_AND_NE")
        step = time.perf_counter()
        sanity = run_sanity_check(
            mlmodelc_path,
            model_dir,
            tokenizer,
            args.seq_len,
            args.batch,
            output_key,
            args.instruction,
            no_token_id,
            yes_token_id,
        )
        timings["sanity"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started

    artifacts = {
        "mlpackage": str(mlpackage_path) if args.keep_mlpackage else None,
        "mlmodelc": str(mlmodelc_path),
        "metadata": str(metadata_path),
    }
    metadata = build_metadata(
        args, model_dir, patch_record, prompt, output_key, artifacts, timings, sanity
    )
    common.write_result_json(metadata_path, metadata)

    if sanity is not None:
        print_sanity_summary(sanity)
    print(f"      timings (sec)   : {metadata['timings_sec']}")
    print(f"      metadata        : {metadata_path}")
    if sanity is not None and not sanity["passed"]:
        print(f"SANITY CHECK FAILED: {mlmodelc_path}")
        return 1
    print(f"CONVERSION COMPLETE: {mlmodelc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
