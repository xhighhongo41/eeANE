"""Tests for eeane.compiler.backends.modernbert (v0.6 T3, see 開発資料/v0.6実装計画.md §4.2).

The patch/wrapper logic is ported from the frozen PoC scripts, so these
tests pin both the numerics that the PoC verified (masked mean pooling,
stable sigmoid, patched-eager == unpatched-sdpa forward outputs) and the
provisional backend interface that pipeline.py will drive.

Tests needing the real 310M models skip themselves when
``models/ruri-v3-310m`` / ``models/ruri-v3-reranker-310m`` are absent.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers.models.modernbert import modeling_modernbert

from eeane.compiler.backends import modernbert as mb

_REPO_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_DIR = _REPO_ROOT / "models" / "ruri-v3-310m"
RERANKER_MODEL_DIR = _REPO_ROOT / "models" / "ruri-v3-reranker-310m"
_EMBEDDING_AVAILABLE = (EMBEDDING_MODEL_DIR / "config.json").exists()
_RERANKER_AVAILABLE = (RERANKER_MODEL_DIR / "config.json").exists()

# Sequence length used by the real-model smoke tests: short enough to keep
# the FP32 forward passes quick, long enough to exercise padding.
SMOKE_SEQ_LEN = 128

# Tolerance between the patched eager path and the untouched sdpa path.
# The rank-4 rewrite is bit-exact against upstream eager (v0.3実装記録
# §6-6); what is left here is the eager-vs-sdpa kernel difference in FP32.
PATCH_ABS_TOLERANCE = 1e-4


@pytest.fixture(autouse=True, scope="module")
def _restore_transformers_patches() -> Iterator[None]:
    """Undo the global ModernBert monkeypatches after this module's tests."""
    original_rotate_half = modeling_modernbert.rotate_half
    original_forward = modeling_modernbert.ModernBertAttention.forward
    yield
    modeling_modernbert.rotate_half = original_rotate_half
    modeling_modernbert.ModernBertAttention.forward = original_forward


def _tiny_model(seed: int = 0) -> torch.nn.Module:
    """Build a randomly initialised ModernBertModel small enough for a unit test."""
    config = modeling_modernbert.ModernBertConfig(
        vocab_size=128,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=64,
        local_attention=8,
        pad_token_id=0,
    )
    torch.manual_seed(seed)
    model = modeling_modernbert.ModernBertModel(config)
    return model.eval()


def _tiny_inputs(seq_len: int = 16, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic (input_ids, attention_mask) tensors with padding."""
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(1, 128, (batch_size, seq_len), generator=generator)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    attention_mask[-1, seq_len // 2 :] = 0  # last row is half padding
    return input_ids, attention_mask


# --- pooling / sigmoid (the numerics moved out of poc/common.py) -------------


def test_mean_pool_matches_manual_average() -> None:
    """mean_pool must equal the manual average of unmasked positions only."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 4)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])

    result = mb.mean_pool(hidden, attention_mask)

    expected = torch.stack([hidden[0].mean(dim=0), hidden[1, :3].mean(dim=0)])
    assert torch.allclose(result, expected, atol=1e-6)


def test_mean_pool_fully_masked_row_stays_finite() -> None:
    """A fully masked row must not divide by zero (the clamp guards it)."""
    hidden = torch.ones(1, 4, 3)
    attention_mask = torch.zeros(1, 4, dtype=torch.long)

    result = mb.mean_pool(hidden, attention_mask)

    assert torch.isfinite(result).all()


def test_sigmoid_np_matches_torch_for_moderate_values() -> None:
    """sigmoid_np must agree with torch.sigmoid on ordinary logits."""
    logits = np.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=np.float32)

    result = mb.sigmoid_np(logits)

    expected = torch.sigmoid(torch.from_numpy(logits)).numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_sigmoid_np_is_stable_for_large_magnitudes() -> None:
    """Large-magnitude logits must saturate without overflow warnings or NaN."""
    logits = np.array([-1000.0, -80.0, 80.0, 1000.0], dtype=np.float32)

    with np.errstate(over="raise"):
        result = mb.sigmoid_np(logits)

    assert np.isfinite(result).all()
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)
    assert np.all((result >= 0.0) & (result <= 1.0))


# --- provisional backend interface ------------------------------------------


def test_output_name_per_kind() -> None:
    """Embeddings and rerankers must keep the PoC's graph output names."""
    backend = mb.ModernBertBackend()

    assert backend.output_name("embedding") == "embedding"
    assert backend.output_name("reranker") == "logits"


def test_fixtures_have_the_expected_shape_per_kind() -> None:
    """trace/sanity/padding fixtures must be texts for embeddings, pairs for rerankers."""
    backend = mb.ModernBertBackend()

    assert isinstance(backend.trace_example("embedding"), str)
    assert all(isinstance(text, str) for text in backend.sanity_inputs("embedding"))
    assert backend.padding_input("embedding") == ""

    trace_pair = backend.trace_example("reranker")
    assert isinstance(trace_pair, tuple) and len(trace_pair) == 2
    assert all(len(pair) == 2 for pair in backend.sanity_inputs("reranker"))
    assert backend.padding_input("reranker") == ("", "")


def test_sanity_inputs_are_defensive_copies() -> None:
    """Mutating the returned fixture list must not corrupt the module constants."""
    backend = mb.ModernBertBackend()

    returned = backend.sanity_inputs("embedding")
    returned.append("mutated")

    assert len(backend.sanity_inputs("embedding")) == len(mb.SANITY_TEXTS)


@pytest.mark.parametrize(
    "method",
    ["wrap", "trace_example", "sanity_inputs", "padding_input", "output_name"],
)
def test_unknown_kind_is_rejected(method: str) -> None:
    """Every kind-dispatching method must reject an unsupported kind."""
    backend = mb.ModernBertBackend()
    arguments = {"wrap": (SimpleNamespace(),)}.get(method, ())

    with pytest.raises(ValueError, match="kind"):
        getattr(backend, method)(*arguments, "classifier")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    """load must validate the kind before touching the filesystem."""
    backend = mb.ModernBertBackend()

    with pytest.raises(ValueError, match="kind"):
        backend.load(tmp_path, "classifier")


def test_wrap_selects_the_kind_specific_wrapper() -> None:
    """wrap must return the mean-pooling wrapper vs the raw-logits wrapper."""
    backend = mb.ModernBertBackend()
    model = torch.nn.Identity()

    assert isinstance(backend.wrap(model, "embedding"), mb.EmbeddingWrapper)
    assert isinstance(backend.wrap(model, "reranker"), mb.RerankerWrapper)


def test_apply_patches_rejects_odd_rope_head_dim() -> None:
    """An odd RoPE head dim breaks the rotate_half rewrite and must raise."""
    backend = mb.ModernBertBackend()
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=12, num_attention_heads=4))

    with pytest.raises(ValueError, match="head dim"):
        backend.apply_patches(model)


# --- patch behaviour on a tiny randomly initialised model --------------------


def test_patched_eager_matches_unpatched_sdpa_on_a_tiny_model() -> None:
    """The patched eager path must reproduce the untouched sdpa outputs."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs()

    model.config._attn_implementation = "sdpa"
    with torch.no_grad():
        reference = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    mb.ModernBertBackend().apply_patches(model)
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        patched = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert torch.isfinite(patched).all()
    assert (patched - reference).abs().max().item() < PATCH_ABS_TOLERANCE


def test_patches_are_idempotent() -> None:
    """Applying the patches twice must not recurse or change the outputs."""
    model = _tiny_model()
    input_ids, attention_mask = _tiny_inputs()
    backend = mb.ModernBertBackend()

    backend.apply_patches(model)
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        once = model(input_ids=input_ids, attention_mask=attention_mask)[0]
    backend.apply_patches(model)
    with torch.no_grad():
        twice = model(input_ids=input_ids, attention_mask=attention_mask)[0]

    assert torch.equal(once, twice)


def test_rotate_half_patch_matches_the_upstream_formula() -> None:
    """The static-shape rotate_half must return the same values as the sliced one."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 8)
    expected = torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

    mb.patch_rotate_half()

    assert torch.equal(modeling_modernbert.rotate_half(x), expected)


def test_mask_fill_value_patch_produces_finite_masks() -> None:
    """The mask patch must replace -inf-prone fills with the requested finite value."""
    model = _tiny_model()
    fill_value = -30000.0

    mb.patch_mask_fill_value(model, fill_value)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    global_mask, sliding_window_mask = model._update_attention_mask(attention_mask)

    assert global_mask.shape == (1, 1, 1, 4)
    assert sliding_window_mask.shape == (1, 1, 4, 4)
    assert torch.isfinite(global_mask).all()
    assert torch.isfinite(sliding_window_mask).all()
    # Unmasked positions stay at 0.0, masked ones carry exactly fill_value.
    assert global_mask[0, 0, 0].tolist() == [0.0, 0.0, 0.0, fill_value]
    # local_attention=8 -> a +-4 window, so every position is in-window here.
    assert sliding_window_mask[0, 0, 0].tolist() == [0.0, 0.0, 0.0, fill_value]


def test_mask_fill_value_patch_masks_outside_the_sliding_window() -> None:
    """Positions beyond local_attention//2 must be filled even when unmasked."""
    model = _tiny_model()
    model.config.local_attention = 2  # +-1 window
    fill_value = -30000.0

    mb.patch_mask_fill_value(model, fill_value)
    _, sliding_window_mask = model._update_attention_mask(torch.ones(1, 4, dtype=torch.long))

    assert sliding_window_mask[0, 0, 0].tolist() == [0.0, 0.0, fill_value, fill_value]


# --- real-model smoke tests (local only) ------------------------------------


@pytest.fixture(scope="module")
def embedding_smoke() -> dict[str, Any]:
    """Run the patched embedding wrapper and the FP32 sdpa reference once."""
    if not _EMBEDDING_AVAILABLE:
        pytest.skip("ruri-v3-310m model directory not found")
    return _run_smoke(EMBEDDING_MODEL_DIR, "embedding")


@pytest.fixture(scope="module")
def reranker_smoke() -> dict[str, Any]:
    """Run the patched reranker wrapper and the FP32 sdpa reference once."""
    if not _RERANKER_AVAILABLE:
        pytest.skip("ruri-v3-reranker-310m model directory not found")
    return _run_smoke(RERANKER_MODEL_DIR, "reranker")


def _run_smoke(model_dir: Path, kind: str) -> dict[str, Any]:
    """Compare one patched eager forward pass against the FP32 sdpa reference."""
    backend = mb.ModernBertBackend()
    inputs = backend.sanity_inputs(kind)[:1]
    model, tokenizer = backend.load(model_dir, kind, attn="eager")
    backend.apply_patches(model)
    wrapper = backend.wrap(model, kind)
    tokens = backend.tokenize(tokenizer, kind, inputs, SMOKE_SEQ_LEN)
    with torch.no_grad():
        patched = wrapper(
            torch.from_numpy(tokens["input_ids"]).long(),
            torch.from_numpy(tokens["attention_mask"]).long(),
        )
    reference = backend.reference_outputs(model_dir, kind, inputs, SMOKE_SEQ_LEN)
    return {
        "tokens": tokens,
        "patched": patched.numpy().reshape(len(inputs), -1),
        "reference": np.asarray(reference, dtype=np.float32).reshape(len(inputs), -1),
    }


@pytest.mark.parametrize("smoke_fixture", ["embedding_smoke", "reranker_smoke"])
def test_tokenize_returns_fixed_shape_int32_arrays(
    smoke_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Tokenized Core ML inputs must be (N, S) int32 with a non-empty mask."""
    tokens = request.getfixturevalue(smoke_fixture)["tokens"]

    for key in ("input_ids", "attention_mask"):
        assert tokens[key].dtype == np.int32
        assert tokens[key].shape == (1, SMOKE_SEQ_LEN)
    assert tokens["attention_mask"].sum() > 0
    assert tokens["attention_mask"].sum() < SMOKE_SEQ_LEN  # the row really is padded


@pytest.mark.parametrize("smoke_fixture", ["embedding_smoke", "reranker_smoke"])
def test_patched_forward_matches_fp32_reference(
    smoke_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Patched eager outputs must stay within tolerance of the FP32 sdpa baseline."""
    smoke = request.getfixturevalue(smoke_fixture)

    assert np.isfinite(smoke["patched"]).all()
    assert np.abs(smoke["patched"] - smoke["reference"]).max() < PATCH_ABS_TOLERANCE
