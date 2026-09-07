"""Tests for poc_qwen.common.

None of these tests download model weights: the attention-mask
equivalence is checked against a small randomly initialized Qwen3 model
built from an in-test config, and the tokenization helper is checked
against a minimal stand-in tokenizer defined below.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from poc_qwen.common import (
    MASK_FILL_VALUE,
    build_causal_padding_mask,
    build_versions_info,
    cosine_rowwise,
    encode_reference,
    last_token_pool_gather,
    last_token_pool_left,
    sanity_text_sets,
    tokenize_batch,
    write_result_json,
)


def _tiny_model(seed: int = 0) -> Qwen3Model:
    """Build a small randomly initialized Qwen3 model for mask tests.

    Args:
        seed: Seed applied before construction so weights are reproducible.

    Returns:
        A ``Qwen3Model`` in eval mode with caching disabled.
    """
    torch.manual_seed(seed)
    config = Qwen3Config(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
    )
    config.use_cache = False
    return Qwen3Model(config).eval()


class _StubTokenizer:
    """Minimal stand-in for a HuggingFace tokenizer call.

    Records the texts and keyword arguments it was called with, and
    returns fixed-shape int64 arrays plus one extra key, so tests can
    check both the delegation and the key/dtype normalization done by
    :func:`poc_qwen.common.tokenize_batch`.
    """

    def __init__(self, padding_side: str = "left") -> None:
        self.padding_side = padding_side
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, np.ndarray]:
        """Record one call and return arrays shaped like the request."""
        self.calls.append((list(texts), dict(kwargs)))
        rows = len(texts)
        cols = int(kwargs["max_length"])
        ids = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
        return {
            "input_ids": ids,
            "attention_mask": np.ones((rows, cols), dtype=np.int64),
            # Extra keys must be dropped by tokenize_batch.
            "token_type_ids": np.zeros((rows, cols), dtype=np.int64),
        }


def test_build_causal_padding_mask_matches_hand_computed_layout() -> None:
    """The 4-D mask must be 0 only on causal, unpadded positions."""
    attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
    fill = -5.0

    mask = build_causal_padding_mask(attention_mask, fill_value=fill)

    expected = torch.tensor(
        [
            [
                [
                    [fill, fill, fill],
                    [fill, 0.0, fill],
                    [fill, 0.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert mask.shape == (1, 1, 3, 3)
    assert mask.dtype == torch.float32
    assert torch.equal(mask, expected)


def test_build_causal_padding_mask_defaults_and_accepts_float_mask() -> None:
    """A float mask must give the same result as an int one, with the default fill."""
    int_mask = torch.tensor([[1, 1], [0, 1]], dtype=torch.long)

    from_int = build_causal_padding_mask(int_mask)
    from_float = build_causal_padding_mask(int_mask.to(torch.float32))

    expected = torch.tensor(
        [
            [[[0.0, MASK_FILL_VALUE], [0.0, 0.0]]],
            [[[MASK_FILL_VALUE, MASK_FILL_VALUE], [MASK_FILL_VALUE, 0.0]]],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(from_int, expected)
    assert torch.equal(from_float, expected)


def test_four_dim_mask_path_matches_two_dim_mask_path() -> None:
    """Feeding the 4-D mask must reproduce the framework's 2-D mask path exactly.

    Only real (unpadded) token positions are compared: the hidden states
    at padded positions are unconstrained because those rows are fully
    masked, and the two paths are free to differ there.
    """
    model = _tiny_model()
    torch.manual_seed(1)
    input_ids = torch.randint(0, model.config.vocab_size, (3, 6))
    # Left padding with a different padded prefix length per row.
    attention_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        from_2d = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        from_4d = model(
            input_ids=input_ids,
            attention_mask=build_causal_padding_mask(attention_mask),
        ).last_hidden_state

    real = attention_mask.to(torch.bool)
    assert from_4d.shape == from_2d.shape
    assert torch.equal(from_4d[real], from_2d[real])


def test_both_pooling_helpers_select_the_same_last_real_token() -> None:
    """Each helper, used with the padding side it is meant for, must pick the same token.

    ``last_token_pool_left`` reads the final position, which only holds a
    real token under left padding; ``last_token_pool_gather`` reads
    ``sum(mask) - 1``, which only points at the last real token under
    right padding. The same rows are therefore laid out both ways, with
    distinct values in the padded slots, and both helpers must return the
    row's last real token.
    """
    torch.manual_seed(2)
    lengths = [3, 4, 5]
    seq_len = 5
    real = torch.randn(3, seq_len, 8)
    left_hidden = torch.full((3, seq_len, 8), -99.0)
    right_hidden = torch.full((3, seq_len, 8), 99.0)
    left_mask = torch.zeros(3, seq_len, dtype=torch.long)
    right_mask = torch.zeros(3, seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        left_hidden[row, seq_len - length :] = real[row, :length]
        right_hidden[row, :length] = real[row, :length]
        left_mask[row, seq_len - length :] = 1
        right_mask[row, :length] = 1
    expected = torch.stack([real[row, length - 1] for row, length in enumerate(lengths)])

    pooled_left = last_token_pool_left(left_hidden)
    pooled_gather = last_token_pool_gather(right_hidden, right_mask)

    assert pooled_left.shape == (3, 8)
    assert torch.equal(pooled_left, left_hidden[:, -1, :])
    assert torch.equal(pooled_left, expected)
    assert torch.equal(pooled_gather, expected)


def test_last_token_pool_gather_selects_last_unpadded_position() -> None:
    """On a right-padded batch the gather pooling must skip the padding."""
    torch.manual_seed(3)
    hidden = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long)

    pooled = last_token_pool_gather(hidden, attention_mask)

    expected = torch.stack([hidden[0, 2], hidden[1, 1]])
    assert torch.equal(pooled, expected)


def test_last_token_pool_gather_clamps_an_all_padding_row() -> None:
    """A row with no real token must read position 0 instead of wrapping to the end."""
    torch.manual_seed(4)
    hidden = torch.randn(1, 4, 3)
    attention_mask = torch.zeros(1, 4, dtype=torch.long)

    pooled = last_token_pool_gather(hidden, attention_mask)

    assert torch.equal(pooled, hidden[:, 0, :])


def test_encode_reference_matches_manual_last_token_pooling() -> None:
    """The reference encoder must equal a manual per-row forward plus pooling."""
    model = _tiny_model()
    tokenizer = _StubTokenizer(padding_side="left")
    texts = ["one", "two"]
    seq_len = 6

    embeddings = encode_reference(model, tokenizer, texts, seq_len)

    batch = tokenize_batch(tokenizer, texts, seq_len)
    with torch.no_grad():
        expected = np.stack(
            [
                model(
                    input_ids=torch.from_numpy(batch["input_ids"][i : i + 1]).long(),
                    attention_mask=torch.from_numpy(batch["attention_mask"][i : i + 1]).long(),
                )[0][:, -1, :]
                .reshape(-1)
                .numpy()
                for i in range(len(texts))
            ]
        )
    assert embeddings.shape == (len(texts), model.config.hidden_size)
    assert embeddings.dtype == np.float32
    np.testing.assert_array_equal(embeddings, expected)


def test_encode_reference_on_empty_input_returns_empty_array() -> None:
    """No texts must give an empty (0, hidden) array without calling the tokenizer."""
    model = _tiny_model()
    tokenizer = _StubTokenizer()

    embeddings = encode_reference(model, tokenizer, [], 6)

    assert embeddings.shape == (0, model.config.hidden_size)
    assert embeddings.dtype == np.float32
    assert tokenizer.calls == []


def test_encode_reference_rejects_unknown_pool() -> None:
    """An unsupported pooling name must raise rather than silently pick one."""
    model = _tiny_model()
    tokenizer = _StubTokenizer()

    with pytest.raises(ValueError):
        encode_reference(model, tokenizer, ["one"], 6, pool="bogus")


def test_tokenize_batch_returns_two_int32_keys() -> None:
    """Only input_ids/attention_mask are kept, cast to int32 at the requested shape."""
    tokenizer = _StubTokenizer()
    texts = ["first text", "second text"]
    seq_len = 7

    batch = tokenize_batch(tokenizer, texts, seq_len)

    assert set(batch) == {"input_ids", "attention_mask"}
    for value in batch.values():
        assert value.dtype == np.int32
        assert value.shape == (len(texts), seq_len)
    called_texts, kwargs = tokenizer.calls[0]
    assert called_texts == texts
    assert kwargs["padding"] == "max_length"
    assert kwargs["truncation"] is True
    assert kwargs["max_length"] == seq_len
    assert kwargs["return_tensors"] == "np"
    # The padding side stays whatever the tokenizer was configured with.
    assert "padding_side" not in kwargs


def test_tokenize_batch_prepends_prompt_to_every_text() -> None:
    """A non-empty prompt is concatenated in front of each text; an empty one is not."""
    tokenizer = _StubTokenizer()
    texts = ["alpha", "beta"]
    prompt = "Instruct: do something\nQuery:"

    tokenize_batch(tokenizer, texts, 4, prompt=prompt)
    tokenize_batch(tokenizer, texts, 4)

    assert tokenizer.calls[0][0] == [prompt + "alpha", prompt + "beta"]
    assert tokenizer.calls[1][0] == texts


def test_cosine_rowwise_known_vectors() -> None:
    """Identical/orthogonal/opposite rows must give 1.0/0.0/-1.0."""
    a = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)

    result = cosine_rowwise(a, b)

    np.testing.assert_allclose(result, [1.0, 0.0, -1.0], atol=1e-6)


def test_cosine_rowwise_zero_row_does_not_divide_by_zero() -> None:
    """A zero row must yield a finite similarity instead of a NaN."""
    a = np.array([[0.0, 0.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0]], dtype=np.float32)

    result = cosine_rowwise(a, b)

    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.0)


def test_sanity_text_sets_cover_three_languages_with_three_texts() -> None:
    """The shared sanity sets must expose en/ja/zh with three texts each."""
    sets = sanity_text_sets()

    assert tuple(language for language, _ in sets) == ("en", "ja", "zh")
    for _, texts in sets:
        assert len(texts) == 3
        assert all(isinstance(text, str) and text for text in texts)


def test_build_versions_info_contains_required_keys() -> None:
    """Every recorded version must be a non-empty string."""
    versions = build_versions_info()

    for key in ("torch", "transformers", "coremltools", "numpy", "python", "platform"):
        assert isinstance(versions[key], str)
        assert versions[key]


def test_write_result_json_creates_parents_and_adds_versions(tmp_path: Path) -> None:
    """Writing a payload must create the directory and record the versions."""
    path = tmp_path / "nested" / "result.json"

    write_result_json(path, {"note": "日本語"})

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["note"] == "日本語"
    assert written["versions"]["torch"]
    # Non-ASCII must be stored verbatim rather than escaped.
    assert "日本語" in path.read_text(encoding="utf-8")


def test_write_result_json_keeps_explicit_versions(tmp_path: Path) -> None:
    """A payload that already carries versions must not be overwritten."""
    path = tmp_path / "result.json"

    write_result_json(path, {"versions": {"torch": "pinned"}})

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["versions"] == {"torch": "pinned"}
