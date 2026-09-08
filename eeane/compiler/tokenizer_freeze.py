"""Tokenizer freezing and equivalence verification.

``eeane serve`` must not depend on ``transformers``: at compile time the
*effective* backend tokenizer -- the one ``AutoTokenizer.from_pretrained``
builds in memory, including the ``post_processor`` that
``LlamaTokenizerFast.update_post_processor()`` rebuilds from
``tokenizer_config.json`` -- is serialized to a standalone
``tokenizer.json``, which the runtime then loads with the ``tokenizers``
library alone (see :func:`eeane.runtime.load_frozen_tokenizer`).

Freezing is only half of the contract: :func:`verify_frozen_tokenizer` is
the compile-time gate that proves the frozen file reproduces
``AutoTokenizer``'s ``input_ids``/``attention_mask`` exactly, for every
bucket length the model is compiled for. A mismatch fails the compile
instead of silently shipping a tokenizer that disagrees with the one the
accuracy checks were run against.

This module belongs to the ``[compile]`` side: it imports
``transformers`` at module load time and must never be imported from a
runtime module.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from eeane import runtime
from eeane.runtime import PairTemplate

# Markers a pair template's body format writes a request's own texts at.
# Spelled out here rather than imported from the layers under test: this
# module is the gate that proves the two sides agree, so it states the
# convention itself instead of borrowing either side's copy of it.
_QUERY_MARKER = "{query}"
_DOCUMENT_MARKER = "{document}"


class TokenizerFreezeError(RuntimeError):
    """Raised when a frozen tokenizer does not reproduce ``AutoTokenizer``.

    Attributes:
        report: The same structure :func:`verify_frozen_tokenizer` would
            have returned, with ``passed=False`` and a ``mismatches``
            list describing every difference that was found.
    """

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        """Store the verification report alongside the message.

        Args:
            message: Human-readable summary (already includes the first
                few mismatches).
            report: Structured verification report.
        """
        super().__init__(message)
        self.report = report


def freeze_tokenizer(model_dir: Path, out_path: Path) -> dict[str, Any]:
    """Serialize a model's effective backend tokenizer to ``out_path``.

    Loads the tokenizer through ``AutoTokenizer.from_pretrained`` so that
    every Python-side adjustment (notably ``update_post_processor()``) has
    been applied, then bakes the model's pad token into the backend's
    ``padding`` section before saving. The runtime therefore gets
    ``pad_id``/``pad_token`` from the file itself and needs no side-car
    metadata. ``length`` is deliberately left unset (and no ``truncation``
    section is written) because the runtime sets both per request, once
    the sequence-length bucket is known.

    ``model_dir`` is only read; the frozen file is written to ``out_path``
    (input model directories are read-only).

    Args:
        model_dir: HuggingFace distribution-format model directory, or
            any value ``AutoTokenizer.from_pretrained`` accepts.
        out_path: Destination ``tokenizer.json`` path. Parent directories
            are created if needed; an existing file is overwritten.

    Returns:
        Information about what was frozen: ``tokenizer_class``,
        ``pad_id``, ``pad_token``, ``padding_direction`` and ``path``.

    Raises:
        TokenizerFreezeError: If the tokenizer has no fast backend, or no
            pad token to bake into the padding section.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise TokenizerFreezeError(
            f"tokenizer of '{model_dir}' has no fast (Rust) backend, so it cannot be frozen; "
            "eeANE requires a model shipping tokenizer.json or a slow->fast convertible "
            "tokenizer",
            {"passed": False, "model_dir": str(model_dir)},
        )
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        raise TokenizerFreezeError(
            f"tokenizer of '{model_dir}' defines no pad token, so the runtime would have "
            "nothing to pad the fixed-length Core ML inputs with",
            {"passed": False, "model_dir": str(model_dir)},
        )

    # Bake pad_id/pad_token (but no length): the runtime reads them back
    # from the file and sets the length per request.
    backend.enable_padding(pad_id=tokenizer.pad_token_id, pad_token=tokenizer.pad_token)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backend.save(str(out_path))

    padding = backend.padding or {}
    return {
        "path": str(out_path),
        "tokenizer_class": type(tokenizer).__name__,
        "pad_id": int(tokenizer.pad_token_id),
        "pad_token": str(tokenizer.pad_token),
        "padding_direction": str(padding.get("direction", "right")),
    }


def verify_frozen_tokenizer(
    model_dir: Path,
    frozen_path: Path,
    texts: list[str],
    pairs: list[tuple[str, str]],
    buckets: list[int],
    pair_template: PairTemplate | None = None,
) -> dict[str, Any]:
    """Check that a frozen tokenizer matches ``AutoTokenizer`` exactly.

    For every bucket length in ``buckets``, the frozen tokenizer is driven
    through the very functions the server uses
    (:func:`eeane.runtime.tokenize_texts` /
    :func:`eeane.runtime.tokenize_pairs`) and compared cell by cell with
    ``tokenizer(..., padding="max_length", truncation=True,
    max_length=bucket, return_tensors="np")``. The unpadded/untruncated
    token counts (which drive bucket selection) are compared as well.

    With a ``pair_template``, the pairs take the template path on both
    sides instead: the frozen side through the server's own function, the
    reference side through an encoding assembled here from
    ``AutoTokenizer`` alone. That second assembly is written out in full
    rather than delegated to the compile-side helper on purpose -- the
    worth of this gate is that the budget arithmetic and the truncation
    rule are written twice and then held against each other, which
    sharing one implementation between the two sides would undo.

    Everything runs serially on purpose: fast tokenizers keep mutable
    Rust-side padding/truncation state.

    Args:
        model_dir: Model directory the frozen file was produced from.
        frozen_path: Frozen ``tokenizer.json`` written by
            :func:`freeze_tokenizer`.
        texts: Single-sequence verification inputs.
        pairs: ``(query, document)`` verification inputs. May be empty for
            embedding models.
        buckets: Sequence-length buckets to verify. Must be non-empty and
            strictly positive.
        pair_template: Template the model wants its pairs spelled out
            with, or ``None`` when its pairs go through the tokenizer's
            own pair encoding.

    Returns:
        A report dict with ``passed=True``, the verified ``buckets``, the
        number of ``texts``/``pairs`` compared, the number of comparisons
        performed, and how many tokens a template's fixed parts reserve
        (``None`` when no template was given).

    Raises:
        ValueError: If ``buckets`` is empty or holds a non-positive
            length, or if a template's own tokens leave no room for the
            pair inside one of the buckets.
        TokenizerFreezeError: If any comparison differs. The message names
            the bucket, the array, the sample index and the first
            differing positions.
    """
    if not buckets:
        raise ValueError("at least one bucket length is required to verify a frozen tokenizer")
    non_positive = [bucket for bucket in buckets if bucket <= 0]
    if non_positive:
        raise ValueError(f"bucket lengths must be positive, got {non_positive}")

    frozen = runtime.load_frozen_tokenizer(frozen_path)
    reference = AutoTokenizer.from_pretrained(model_dir)
    prepared = (
        None if pair_template is None else runtime.prepare_pair_template(frozen, pair_template)
    )
    affixes: tuple[list[int], list[int]] = (
        ([], []) if pair_template is None else _reference_affixes(reference, pair_template)
    )

    mismatches: list[dict[str, Any]] = []
    comparisons = 0
    for bucket in buckets:
        if texts:
            mismatches += _compare_batch(
                runtime.tokenize_texts(frozen, texts, bucket),
                _reference_texts(reference, texts, bucket),
                bucket=bucket,
                kind="texts",
            )
            comparisons += 2
        if pairs:
            mismatches += _compare_batch(
                runtime.tokenize_pairs(frozen, pairs, bucket, prepared),
                _reference_pair_batch(reference, pairs, bucket, pair_template, affixes),
                bucket=bucket,
                kind="pairs",
            )
            comparisons += 2

    # Token counts drive bucket selection at request time, so they are
    # part of the contract even though no padding/truncation applies.
    mismatches += _compare_counts(
        [runtime.count_text_tokens(frozen, text) for text in texts],
        _reference_text_counts(reference, texts),
        kind="texts",
    )
    mismatches += _compare_counts(
        [runtime.count_pair_tokens(frozen, query, document, prepared) for query, document in pairs],
        _reference_pair_count_list(reference, pairs, pair_template, affixes),
        kind="pairs",
    )
    if texts:
        comparisons += 1
    if pairs:
        comparisons += 1

    report: dict[str, Any] = {
        "passed": not mismatches,
        "model_dir": str(model_dir),
        "frozen_path": str(frozen_path),
        "buckets": list(buckets),
        "n_texts": len(texts),
        "n_pairs": len(pairs),
        "n_comparisons": comparisons,
        "pair_template": None
        if pair_template is None
        else {"prefix_tokens": len(affixes[0]), "suffix_tokens": len(affixes[1])},
        "mismatches": mismatches,
    }
    if mismatches:
        raise TokenizerFreezeError(
            f"frozen tokenizer '{frozen_path}' does not reproduce "
            f"AutoTokenizer('{model_dir}'): {len(mismatches)} mismatch(es). "
            + " | ".join(entry["summary"] for entry in mismatches[:5]),
            report,
        )
    return report


def _reference_texts(
    reference: PreTrainedTokenizerBase, texts: list[str], bucket: int
) -> dict[str, np.ndarray]:
    """Encode texts the way the v0.5 runtime did, as the comparison baseline.

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        texts: Input sentences.
        bucket: Fixed sequence length.

    Returns:
        Dict with int32 ``input_ids``/``attention_mask`` of shape
        ``(len(texts), bucket)``.
    """
    encoded = reference(
        texts,
        padding="max_length",
        truncation=True,
        max_length=bucket,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def _reference_pairs(
    reference: PreTrainedTokenizerBase, pairs: list[tuple[str, str]], bucket: int
) -> dict[str, np.ndarray]:
    """Encode pairs the way the v0.5 runtime did, as the comparison baseline.

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        pairs: ``(query, document)`` inputs.
        bucket: Fixed sequence length.

    Returns:
        Dict with int32 ``input_ids``/``attention_mask`` of shape
        ``(len(pairs), bucket)``.
    """
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    encoded = reference(
        queries,
        documents,
        padding="max_length",
        truncation=True,
        max_length=bucket,
        return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
    }


def _reference_pair_batch(
    reference: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    bucket: int,
    template: PairTemplate | None,
    affixes: tuple[Sequence[int], Sequence[int]],
) -> dict[str, np.ndarray]:
    """Build the baseline encoding of the pairs, templated or not.

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        pairs: ``(query, document)`` inputs.
        bucket: Fixed sequence length.
        template: Template the pairs are laid out with, or ``None`` for
            the tokenizer's own pair encoding.
        affixes: The template's already-encoded fixed halves; ignored
            without a template.

    Returns:
        Dict with int32 ``input_ids``/``attention_mask`` of shape
        ``(len(pairs), bucket)``.
    """
    if template is None:
        return _reference_pairs(reference, pairs, bucket)
    return _reference_templated_pairs(reference, pairs, bucket, template, affixes)


def _reference_pair_count_list(
    reference: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    template: PairTemplate | None,
    affixes: tuple[Sequence[int], Sequence[int]],
) -> list[int]:
    """Count the tokens each pair needs, by the definition its model uses.

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        pairs: ``(query, document)`` inputs.
        template: Template the pairs are laid out with, or ``None`` for
            the tokenizer's own pair encoding.
        affixes: The template's already-encoded fixed halves; ignored
            without a template.

    Returns:
        One untruncated count per input, in order.
    """
    if template is None:
        return _reference_pair_counts(reference, pairs)
    return _reference_templated_pair_counts(reference, pairs, template, affixes)


def _reference_affixes(
    reference: PreTrainedTokenizerBase, template: PairTemplate
) -> tuple[list[int], list[int]]:
    """Encode a template's two fixed halves with ``AutoTokenizer``.

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        template: Template the pairs are laid out with.

    Returns:
        ``(prefix_ids, suffix_ids)``. Special tokens are off: both halves
        already spell the model's control tokens out as text, so letting
        the tokenizer add its own would encode a different prompt.
    """
    return (
        list(reference.encode(template.prefix, add_special_tokens=False)),
        list(reference.encode(template.suffix, add_special_tokens=False)),
    )


def _reference_body(template: PairTemplate, query: str, document: str) -> str:
    """Write one pair into a template's body, as the baseline of the comparison.

    Args:
        template: Template being applied.
        query: Query text.
        document: Document text.

    Returns:
        The rendered body.
    """
    # One left-to-right pass, never ``str.format``: a text of its own
    # carrying braces (or a marker) must be written in verbatim rather
    # than reinterpreted or substituted into.
    query_at = template.body_format.find(_QUERY_MARKER)
    document_at = template.body_format.find(_DOCUMENT_MARKER)
    if query_at <= document_at:
        markers = ((_QUERY_MARKER, query), (_DOCUMENT_MARKER, document))
    else:
        markers = ((_DOCUMENT_MARKER, document), (_QUERY_MARKER, query))
    head, _, rest = template.body_format.partition(markers[0][0])
    middle, _, tail = rest.partition(markers[1][0])
    return head + markers[0][1] + middle + markers[1][1] + tail


def _reference_templated_pairs(
    reference: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    bucket: int,
    template: PairTemplate,
    affixes: tuple[Sequence[int], Sequence[int]],
) -> dict[str, np.ndarray]:
    """Assemble the templated encoding independently, as the comparison baseline.

    Deliberately written out here rather than delegated: this is the
    second, independent statement of the rule the frozen side is being
    held to (keep both fixed halves whole, cut only the body, from its
    right end, then pad on the right).

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        pairs: ``(query, document)`` inputs.
        bucket: Fixed sequence length.
        template: Template the pairs are laid out with.
        affixes: The template's already-encoded fixed halves.

    Returns:
        Dict with int32 ``input_ids``/``attention_mask`` of shape
        ``(len(pairs), bucket)``.

    Raises:
        ValueError: If the template leaves no room for the pair's text, or
            if the tokenizer defines no pad token.
    """
    prefix_ids, suffix_ids = affixes
    budget = bucket - len(prefix_ids) - len(suffix_ids)
    if budget < 1:
        raise ValueError(
            f"the pair template needs {len(prefix_ids)} prefix and {len(suffix_ids)} suffix "
            f"tokens, which leaves no room for the pair inside bucket {bucket}"
        )
    pad_id = reference.pad_token_id
    if pad_id is None:
        raise ValueError(
            f"the tokenizer of the verified model defines no pad token, so a {bucket}-token "
            "row cannot be filled"
        )
    input_ids = np.full((len(pairs), bucket), int(pad_id), dtype=np.int32)
    attention_mask = np.zeros((len(pairs), bucket), dtype=np.int32)
    for row, (query, document) in enumerate(pairs):
        body = _reference_body(template, query, document)
        body_ids = list(reference.encode(body, add_special_tokens=False))[:budget]
        ids = [*prefix_ids, *body_ids, *suffix_ids]
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _reference_templated_pair_counts(
    reference: PreTrainedTokenizerBase,
    pairs: list[tuple[str, str]],
    template: PairTemplate,
    affixes: tuple[Sequence[int], Sequence[int]],
) -> list[int]:
    """Count the tokens a templated pair needs, untruncated (baseline).

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        pairs: ``(query, document)`` inputs.
        template: Template the pairs are laid out with.
        affixes: The template's already-encoded fixed halves.

    Returns:
        One count per input, in order: the whole templated input, which is
        what a bucket is selected from.
    """
    prefix_ids, suffix_ids = affixes
    fixed = len(prefix_ids) + len(suffix_ids)
    return [
        fixed
        + len(
            reference.encode(_reference_body(template, query, document), add_special_tokens=False)
        )
        for query, document in pairs
    ]


def _reference_text_counts(reference: PreTrainedTokenizerBase, texts: list[str]) -> list[int]:
    """Count tokens per text with no padding/truncation (baseline).

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        texts: Input sentences.

    Returns:
        One token count per input, in order.
    """
    if not texts:
        return []
    return [len(ids) for ids in reference(texts)["input_ids"]]


def _reference_pair_counts(
    reference: PreTrainedTokenizerBase, pairs: list[tuple[str, str]]
) -> list[int]:
    """Count tokens per pair with no padding/truncation (baseline).

    Args:
        reference: Tokenizer built by ``AutoTokenizer.from_pretrained``.
        pairs: ``(query, document)`` inputs.

    Returns:
        One token count per input, in order.
    """
    if not pairs:
        return []
    queries = [query for query, _ in pairs]
    documents = [document for _, document in pairs]
    return [len(ids) for ids in reference(queries, documents)["input_ids"]]


def _compare_batch(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    bucket: int,
    kind: str,
) -> list[dict[str, Any]]:
    """Compare one frozen/reference encoding batch cell by cell.

    Args:
        actual: Arrays produced through :mod:`eeane.runtime`.
        expected: Arrays produced by ``AutoTokenizer``.
        bucket: Sequence length the batch was encoded at (for reporting).
        kind: ``"texts"`` or ``"pairs"`` (for reporting).

    Returns:
        One mismatch entry per differing array (empty when identical).
    """
    mismatches: list[dict[str, Any]] = []
    for name in ("input_ids", "attention_mask"):
        actual_array = actual[name]
        expected_array = expected[name]
        if actual_array.shape != expected_array.shape:
            mismatches.append(
                {
                    "summary": (
                        f"bucket {bucket} {kind} {name}: shape {actual_array.shape} != "
                        f"{expected_array.shape}"
                    ),
                    "bucket": bucket,
                    "input": kind,
                    "array": name,
                    "actual_shape": list(actual_array.shape),
                    "expected_shape": list(expected_array.shape),
                }
            )
            continue
        if np.array_equal(actual_array, expected_array):
            continue
        differences = _first_differences(actual_array, expected_array)
        first = differences[0]
        mismatches.append(
            {
                "summary": (
                    f"bucket {bucket} {kind} {name}: "
                    f"{int(np.count_nonzero(actual_array != expected_array))} differing cell(s), "
                    f"first at sample {first['index']} position {first['position']} "
                    f"(frozen={first['frozen']}, reference={first['reference']})"
                ),
                "bucket": bucket,
                "input": kind,
                "array": name,
                "n_differing_cells": int(np.count_nonzero(actual_array != expected_array)),
                "first_differences": differences,
            }
        )
    return mismatches


def _compare_counts(actual: list[int], expected: list[int], *, kind: str) -> list[dict[str, Any]]:
    """Compare the unpadded token counts that drive bucket selection.

    Args:
        actual: Counts from :mod:`eeane.runtime`.
        expected: Counts from ``AutoTokenizer``.
        kind: ``"texts"`` or ``"pairs"`` (for reporting).

    Returns:
        One mismatch entry per differing input (empty when identical).
    """
    mismatches: list[dict[str, Any]] = []
    for index, (actual_count, expected_count) in enumerate(zip(actual, expected, strict=True)):
        if actual_count == expected_count:
            continue
        mismatches.append(
            {
                "summary": (
                    f"{kind} token count at sample {index}: "
                    f"frozen={actual_count}, reference={expected_count}"
                ),
                "input": kind,
                "check": "token_count",
                "index": index,
                "frozen": actual_count,
                "reference": expected_count,
            }
        )
    return mismatches


def _first_differences(
    actual: np.ndarray, expected: np.ndarray, limit: int = 3
) -> list[dict[str, Any]]:
    """Describe the first differing cells of two equally shaped arrays.

    Args:
        actual: Array produced from the frozen tokenizer.
        expected: Array produced by ``AutoTokenizer``.
        limit: Maximum number of differing cells to describe.

    Returns:
        One dict per reported cell, with the sample index, the position
        inside the sequence and both values.
    """
    positions = np.argwhere(actual != expected)[:limit]
    return [
        {
            "index": int(index),
            "position": int(position),
            "frozen": int(actual[index, position]),
            "reference": int(expected[index, position]),
        }
        for index, position in positions
    ]
