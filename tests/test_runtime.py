"""Tests for eeane.runtime."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from eeane.config import default_config
from eeane.runtime import (
    FrozenTokenizer,
    PairTemplate,
    base64_to_floats,
    count_pair_tokens,
    count_text_tokens,
    floats_to_base64,
    l2_normalize,
    load_frozen_tokenizer,
    prepare_pair_template,
    select_bucket,
    sigmoid,
    tokenize_pairs,
    tokenize_texts,
)

_EMBEDDING_TOKENIZER_PATH = default_config().embedding_model.tokenizer
_MODEL_AVAILABLE = _EMBEDDING_TOKENIZER_PATH.is_file()

_BUCKETS = (128, 512, 1024)


@pytest.mark.parametrize(
    ("n_tokens", "expected"),
    [
        (1, (128, False)),
        (128, (128, False)),
        (129, (512, False)),
        (512, (512, False)),
        (513, (1024, False)),
        (1024, (1024, False)),
        (1025, (1024, True)),
        (0, (128, False)),
    ],
)
def test_select_bucket_boundaries(n_tokens: int, expected: tuple[int, bool]) -> None:
    """select_bucket must return the smallest bucket >= n_tokens, truncating above the max."""
    assert select_bucket(n_tokens, _BUCKETS) == expected


def test_sigmoid_zero_is_one_half() -> None:
    """sigmoid(0) must equal exactly 0.5."""
    result = sigmoid(np.array([0.0]))

    np.testing.assert_allclose(result, [0.5])


def test_sigmoid_large_inputs_saturate_without_overflow_warning() -> None:
    """sigmoid must not raise/warn on overflow and must saturate to 0/1 near +-1000."""
    x = np.array([1000.0, -1000.0])

    with warnings.catch_warnings():
        # Promote RuntimeWarning (e.g. numpy overflow in exp) to an error so
        # any overflow in a naive 1 / (1 + exp(-x)) implementation is caught.
        warnings.simplefilter("error")
        result = sigmoid(x)

    np.testing.assert_allclose(result, [1.0, 0.0], atol=1e-9)


def test_sigmoid_monotonic_and_shape_preserving() -> None:
    """sigmoid must be strictly increasing and preserve the input shape."""
    x = np.linspace(-10.0, 10.0, 21)

    result = sigmoid(x)

    assert result.shape == x.shape
    assert np.all(np.diff(result) > 0)


def test_l2_normalize_unit_norm_and_direction() -> None:
    """l2_normalize must produce unit-norm rows pointing in the original direction."""
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((5, 8)).astype(np.float32)

    normalized = l2_normalize(matrix)

    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, np.ones(5), atol=1e-6)

    dot = np.sum(matrix * normalized, axis=1)
    cosine = dot / (np.linalg.norm(matrix, axis=1) * norms)
    np.testing.assert_allclose(cosine, np.ones(5), atol=1e-6)

    assert normalized.dtype == np.float32


def test_l2_normalize_zero_row_has_no_nan_or_inf() -> None:
    """l2_normalize must not produce nan/inf for an all-zero row (eps floor)."""
    matrix = np.zeros((2, 4), dtype=np.float32)

    normalized = l2_normalize(matrix)

    assert np.all(np.isfinite(normalized))


def test_base64_roundtrip_is_bit_exact() -> None:
    """floats_to_base64 -> base64_to_floats must round-trip float32 bits exactly."""
    rng = np.random.default_rng(1)
    vector = rng.standard_normal(768).astype(np.float32)

    decoded = base64_to_floats(floats_to_base64(vector))

    assert np.array_equal(vector, decoded)


# --- frozen tokenizer loading (no model artifacts needed) ----------------


# Vocabulary of the toy tokenizer below. Besides the three content words
# it holds the two special tokens a post-processor can wrap a sequence in,
# and the pieces a brace-carrying text is split into by the whitespace
# pre-tokenizer ("{document}" becomes "{", "document", "}"), so a
# templated input can be read back id by id.
_TOY_VOCAB = {
    "<pad>": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    "<s>": 4,
    "</s>": 5,
    "{": 6,
    "}": 7,
    "query": 8,
    "document": 9,
}

# Ids of the toy tokenizer's special tokens, as its post-processor writes
# them: <s> content </s> for one sequence, <s> A </s> B </s> for a pair.
_BOS = 4
_EOS = 5


def _write_toy_tokenizer(
    path: Path, *, with_padding: bool, with_special_tokens: bool = False
) -> Path:
    """Write a minimal word-level tokenizer.json, with or without a padding section.

    Args:
        path: Destination file.
        with_padding: Whether to bake a padding section (as
            ``eeane compile`` does) before saving.
        with_special_tokens: Whether to register ``<s>``/``</s>`` as
            added tokens and add a post-processor wrapping every
            encoding in them, so a test can tell an
            ``add_special_tokens=False`` encode from a default one, and
            spell the two tokens out inside an ordinary text the way a
            chat-style wrapper does.

    Returns:
        ``path``, for chaining.
    """
    tokenizer = Tokenizer(models.WordLevel(vocab=dict(_TOY_VOCAB), unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if with_special_tokens:
        # Registered as added tokens so they are matched before the
        # pre-tokenizer, which would otherwise split "<s>" into "<", "s"
        # and ">"; they keep the ids the vocabulary already gives them.
        tokenizer.add_special_tokens(["<s>", "</s>"])
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[("<s>", _BOS), ("</s>", _EOS)],
        )
    if with_padding:
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.save(str(path))
    return path


def test_load_frozen_tokenizer_reads_pad_settings(tmp_path: Path) -> None:
    """The pad id/token/direction must come from the frozen file's padding section."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)

    frozen = load_frozen_tokenizer(path)

    assert frozen.pad_id == 0
    assert frozen.pad_token == "<pad>"
    assert frozen.pad_type_id == 0
    assert frozen.pad_direction == "right"


def test_load_frozen_tokenizer_without_padding_section_raises(tmp_path: Path) -> None:
    """A plain (non-frozen) tokenizer.json must be rejected with an `eeane compile` hint."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=False)

    with pytest.raises(ValueError, match="eeane compile"):
        load_frozen_tokenizer(path)


def test_load_frozen_tokenizer_survives_a_counting_call(tmp_path: Path) -> None:
    """Counting clears the backend's padding state; the captured settings must survive it."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)
    frozen = load_frozen_tokenizer(path)

    assert count_text_tokens(frozen, "a b c") == 3
    batch = tokenize_texts(frozen, ["a b c"], 5)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0]]


@pytest.mark.parametrize("seq_len", [0, -1])
def test_tokenize_rejects_non_positive_seq_len(tmp_path: Path, seq_len: int) -> None:
    """A zero/negative bucket length must raise instead of reaching the Rust tokenizer."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)
    frozen = load_frozen_tokenizer(path)

    with pytest.raises(ValueError, match="seq_len"):
        tokenize_texts(frozen, ["a b"], seq_len)
    with pytest.raises(ValueError, match="seq_len"):
        tokenize_pairs(frozen, [("a", "b")], seq_len)


def test_tokenize_empty_batch_keeps_the_two_dimensional_shape(tmp_path: Path) -> None:
    """An empty input list must still yield (0, seq_len) int32 arrays."""
    path = _write_toy_tokenizer(tmp_path / "tokenizer.json", with_padding=True)
    frozen = load_frozen_tokenizer(path)

    texts = tokenize_texts(frozen, [], 4)
    pairs = tokenize_pairs(frozen, [], 4)

    for batch in (texts, pairs):
        assert batch["input_ids"].shape == (0, 4)
        assert batch["attention_mask"].shape == (0, 4)
        assert batch["input_ids"].dtype == np.int32
        assert batch["attention_mask"].dtype == np.int32


# --- pair templates ------------------------------------------------------


# Template every test below shares: a two-token prefix, a body marking
# both texts once, and a two-token suffix. Written out here so each test
# can state the exact ids it expects.
_PREFIX = "<s> a"
_PREFIX_IDS = (_BOS, 1)
_BODY_FORMAT = "{query} b {document}"
_SUFFIX = "c </s>"
_SUFFIX_IDS = (3, _EOS)


def _templated_tokenizer(tmp_path: Path) -> FrozenTokenizer:
    """Load a frozen toy tokenizer whose post-processor adds special tokens.

    Args:
        tmp_path: Directory the tokenizer file is written to.

    Returns:
        The loaded :class:`eeane.runtime.FrozenTokenizer`.
    """
    path = _write_toy_tokenizer(
        tmp_path / "tokenizer.json", with_padding=True, with_special_tokens=True
    )
    return load_frozen_tokenizer(path)


@pytest.mark.parametrize(
    "body_format",
    [
        "{document} only",
        "{query} only",
        "{query} {query} {document}",
        "{query} {document} {document}",
        "no placeholder at all",
    ],
)
def test_pair_template_requires_each_placeholder_exactly_once(body_format: str) -> None:
    """A body format that does not mark both texts exactly once must be rejected."""
    with pytest.raises(ValueError):
        PairTemplate(prefix="", body_format=body_format, suffix="")


def test_pair_template_accepts_an_empty_prefix_and_suffix(tmp_path: Path) -> None:
    """A template that wraps the body in nothing is legal and encodes to no ids."""
    frozen = _templated_tokenizer(tmp_path)

    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix="", body_format=_BODY_FORMAT, suffix="")
    )

    assert prepared.prefix_ids == ()
    assert prepared.suffix_ids == ()
    assert prepared.body_format == _BODY_FORMAT


def test_prepare_pair_template_encodes_the_fixed_parts_without_special_tokens(
    tmp_path: Path,
) -> None:
    """The prefix/suffix spell their control tokens out; the tokenizer must add no more."""
    frozen = _templated_tokenizer(tmp_path)

    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    assert prepared.prefix_ids == _PREFIX_IDS
    assert prepared.suffix_ids == _SUFFIX_IDS


def test_prepare_pair_template_ignores_a_leftover_fixed_length_state(tmp_path: Path) -> None:
    """A previous fixed-length encode must not pad or truncate the template's own parts."""
    frozen = _templated_tokenizer(tmp_path)
    tokenize_texts(frozen, ["a"], 12)

    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    assert prepared.prefix_ids == _PREFIX_IDS
    assert prepared.suffix_ids == _SUFFIX_IDS


def test_templated_pair_is_prefix_then_body_then_suffix_padded_on_the_right(
    tmp_path: Path,
) -> None:
    """The row must read prefix + body + suffix, with the padding after it."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    batch = tokenize_pairs(frozen, [("a", "c c")], 10, template=prepared)

    # <s> a | a b c c | c </s> | two pad positions.
    assert batch["input_ids"].tolist() == [[4, 1, 1, 2, 3, 3, 3, 5, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]


def test_templated_pair_truncates_the_body_and_keeps_prefix_and_suffix(tmp_path: Path) -> None:
    """A pair that does not fit must lose body tokens only, never the fixed parts."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    batch = tokenize_pairs(frozen, [("a", "c c")], 6, template=prepared)

    # Budget of 6 - 2 - 2 = 2 body tokens, taken from the left.
    assert batch["input_ids"].tolist() == [[4, 1, 1, 2, 3, 5]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]


def test_templated_pairs_keep_the_fixed_shape_and_dtype(tmp_path: Path) -> None:
    """Every pair must come back as one int32 row of the bucket's width."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    batch = tokenize_pairs(frozen, [("a", "c"), ("a", "c c c c c")], 8, template=prepared)

    assert set(batch.keys()) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["input_ids"].dtype == np.int32
    assert batch["attention_mask"].dtype == np.int32


def test_templated_empty_batch_keeps_the_two_dimensional_shape(tmp_path: Path) -> None:
    """An empty pair list must still yield (0, seq_len) int32 arrays."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    batch = tokenize_pairs(frozen, [], 8, template=prepared)

    assert batch["input_ids"].shape == (0, 8)
    assert batch["attention_mask"].shape == (0, 8)
    assert batch["input_ids"].dtype == np.int32


def test_templated_pairs_ignore_a_leftover_fixed_length_state(tmp_path: Path) -> None:
    """A previous fixed-length encode must not pad or truncate the body either."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )
    tokenize_texts(frozen, ["a"], 12)

    batch = tokenize_pairs(frozen, [("a", "c c")], 10, template=prepared)

    assert batch["input_ids"].tolist() == [[4, 1, 1, 2, 3, 3, 3, 5, 0, 0]]


def test_templated_pair_substitutes_the_texts_verbatim(tmp_path: Path) -> None:
    """A text that reads like a placeholder must be inserted, never substituted into."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix="", body_format=_BODY_FORMAT, suffix="")
    )

    batch = tokenize_pairs(frozen, [("{document}", "a")], 8, template=prepared)

    # "{document} b a", not "a b {document}": the query's own braces are
    # text, not a marker the document could land in.
    assert batch["input_ids"].tolist() == [[6, 9, 7, 2, 1, 0, 0, 0]]


def test_templated_pair_accepts_a_body_format_carrying_other_braces(tmp_path: Path) -> None:
    """Braces that are not one of the two markers must survive into the input."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix="", body_format="{ {query} } {document}", suffix="")
    )

    batch = tokenize_pairs(frozen, [("a", "b")], 8, template=prepared)

    assert batch["input_ids"].tolist() == [[6, 1, 7, 2, 0, 0, 0, 0]]


@pytest.mark.parametrize("seq_len", [4, 3, 1])
def test_templated_pair_rejects_a_bucket_with_no_room_for_the_texts(
    tmp_path: Path, seq_len: int
) -> None:
    """A bucket the template's own tokens fill up leaves nothing to score."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    with pytest.raises(ValueError):
        tokenize_pairs(frozen, [("a", "c")], seq_len, template=prepared)


def test_count_pair_tokens_with_a_template_reports_the_untruncated_length(
    tmp_path: Path,
) -> None:
    """Bucket selection needs the full length: prefix + whole body + suffix."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )

    # 2 prefix + 4 body ("a b c c") + 2 suffix.
    assert count_pair_tokens(frozen, "a", "c c", template=prepared) == 8


def test_count_pair_tokens_with_a_template_ignores_the_bucket_it_would_be_cut_to(
    tmp_path: Path,
) -> None:
    """A pair longer than any bucket must still be counted in full, so it routes as too long."""
    frozen = _templated_tokenizer(tmp_path)
    prepared = prepare_pair_template(
        frozen, PairTemplate(prefix=_PREFIX, body_format=_BODY_FORMAT, suffix=_SUFFIX)
    )
    document = " ".join(["c"] * 20)

    assert count_pair_tokens(frozen, "a", document, template=prepared) == 2 + 22 + 2


def test_pairs_without_a_template_keep_the_tokenizers_own_pair_encoding(tmp_path: Path) -> None:
    """The untemplated path must go on producing the tokenizer's pair encoding, unchanged."""
    frozen = _templated_tokenizer(tmp_path)

    batch = tokenize_pairs(frozen, [("a", "b c")], 8)
    counted = count_pair_tokens(frozen, "a", "b c")

    # <s> a </s> b c </s>, padded to the bucket.
    assert batch["input_ids"].tolist() == [[4, 1, 5, 2, 3, 5, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert counted == 6


# --- real ruri-v3-310m tokenizer -----------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    """Load the frozen ruri-v3-310m tokenizer once for the tests below."""
    return load_frozen_tokenizer(_EMBEDDING_TOKENIZER_PATH)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_tokenize_texts_shape_and_dtype(tokenizer) -> None:
    """tokenize_texts must produce fixed-shape int32 arrays with only the two expected keys."""
    batch = tokenize_texts(tokenizer, ["これはテストです。", "もう一つの短い日本語の文です。"], 128)

    assert set(batch.keys()) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == (2, 128)
    assert batch["attention_mask"].shape == (2, 128)
    assert batch["input_ids"].dtype == np.int32
    assert batch["attention_mask"].dtype == np.int32


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_count_text_tokens_short_text(tokenizer) -> None:
    """count_text_tokens must include the <s>/</s> special tokens around the content."""
    n = count_text_tokens(tokenizer, "テスト")

    assert n >= 3


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_count_pair_tokens_exceeds_either_side_alone(tokenizer) -> None:
    """count_pair_tokens must exceed the token count of either sequence alone."""
    query = "日本の首都はどこですか。"
    document = "東京は日本の首都であり、政治と経済の中心地として発展してきた都市である。"

    pair_count = count_pair_tokens(tokenizer, query, document)

    assert pair_count > count_text_tokens(tokenizer, query)
    assert pair_count > count_text_tokens(tokenizer, document)


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason="frozen ruri-v3-310m tokenizer not found")
def test_long_text_exceeds_1024_tokens_and_fits_after_bucket_truncation(tokenizer) -> None:
    """A long text must exceed 1024 tokens, and tokenize_texts must truncate it to seq_len."""
    long_text = "今日は天気が良いです。" * 300

    n = count_text_tokens(tokenizer, long_text)
    assert n > 1024

    batch = tokenize_texts(tokenizer, [long_text], 128)
    assert batch["input_ids"].shape == (1, 128)
