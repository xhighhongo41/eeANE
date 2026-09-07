"""Tests for poc_qwen.patches.

Every patch is checked against the upstream implementation it replaces:
the module-level functions are captured before patching and restored
afterwards by a fixture, so patching cannot leak into other tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RMSNorm

from poc_qwen.patches import (
    REPEAT_KV_MODES,
    apply_patches,
    patch_repeat_kv,
    patch_rmsnorm,
    patch_rotate_half,
)


@pytest.fixture
def upstream() -> Iterator[SimpleNamespace]:
    """Capture the upstream module functions and restore them afterwards.

    Yields:
        Namespace with the original ``rotate_half`` and ``repeat_kv``
        functions, for comparison against the patched ones.
    """
    original = SimpleNamespace(
        rotate_half=modeling_qwen3.rotate_half,
        repeat_kv=modeling_qwen3.repeat_kv,
    )
    try:
        yield original
    finally:
        modeling_qwen3.rotate_half = original.rotate_half
        modeling_qwen3.repeat_kv = original.repeat_kv


def _tiny_model(seed: int = 0) -> Qwen3Model:
    """Build a small randomly initialized Qwen3 model.

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


def _randomize_norm_weights(model: torch.nn.Module, seed: int) -> list[Qwen3RMSNorm]:
    """Give every RMSNorm a non-trivial weight so scaling errors show up.

    Args:
        model: Model whose RMSNorm modules are randomized.
        seed: Seed applied before drawing the weights.

    Returns:
        The RMSNorm modules of ``model``, in ``modules()`` order.
    """
    torch.manual_seed(seed)
    norms = [module for module in model.modules() if isinstance(module, Qwen3RMSNorm)]
    with torch.no_grad():
        for norm in norms:
            norm.weight.copy_(torch.rand(norm.weight.shape) + 0.5)
    return norms


def test_patch_rotate_half_matches_upstream(upstream: SimpleNamespace) -> None:
    """The chunk-based rotate_half must equal the slice-based upstream one."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 16)
    expected = upstream.rotate_half(x)

    record = patch_rotate_half()

    assert record["patch"] == "rotate_half"
    assert record["applied"] is True
    assert modeling_qwen3.rotate_half is not upstream.rotate_half
    assert torch.equal(modeling_qwen3.rotate_half(x), expected)


@pytest.mark.parametrize("mode", list(REPEAT_KV_MODES))
def test_patch_repeat_kv_matches_upstream(upstream: SimpleNamespace, mode: str) -> None:
    """Every replacement mode must reproduce the upstream head order bit for bit."""
    torch.manual_seed(1)
    hidden_states = torch.randn(2, 2, 3, 4)
    n_rep = 2
    expected = upstream.repeat_kv(hidden_states, n_rep)

    record = patch_repeat_kv(mode)
    patched = modeling_qwen3.repeat_kv(hidden_states, n_rep)

    assert record["patch"] == "repeat_kv"
    assert record["applied"] is True
    assert record["mode"] == mode
    assert patched.shape == expected.shape
    assert torch.equal(patched, expected)


@pytest.mark.parametrize("mode", list(REPEAT_KV_MODES))
def test_patch_repeat_kv_returns_input_when_n_rep_is_one(
    upstream: SimpleNamespace, mode: str
) -> None:
    """``n_rep == 1`` must short-circuit exactly like the upstream function."""
    torch.manual_seed(2)
    hidden_states = torch.randn(2, 3, 4, 5)

    patch_repeat_kv(mode)

    assert modeling_qwen3.repeat_kv(hidden_states, 1) is hidden_states


def test_patch_repeat_kv_rejects_unknown_mode(upstream: SimpleNamespace) -> None:
    """An unknown mode must raise without touching the module function."""
    with pytest.raises(ValueError):
        patch_repeat_kv("bogus")

    assert modeling_qwen3.repeat_kv is upstream.repeat_kv


def test_patch_rmsnorm_scaled_matches_upstream_on_small_and_large_inputs() -> None:
    """The scaled RMSNorm must match the upstream one at both input magnitudes."""
    model = _tiny_model()
    norms = _randomize_norm_weights(model, seed=3)
    torch.manual_seed(4)
    inputs = {
        id(norm): [
            torch.randn(2, 3, norm.weight.shape[0]),
            torch.randn(2, 3, norm.weight.shape[0]) * 1e3,
        ]
        for norm in norms
    }
    expected = {id(norm): [norm(x) for x in inputs[id(norm)]] for norm in norms}

    record = patch_rmsnorm(model, mode="scaled", scale=64.0)

    assert record["patch"] == "rmsnorm"
    assert record["applied"] is True
    assert record["mode"] == "scaled"
    assert record["patched_modules"] == len(norms)
    assert len(norms) > 0
    for norm in norms:
        for x, want in zip(inputs[id(norm)], expected[id(norm)], strict=True):
            assert torch.allclose(norm(x), want, rtol=1e-5, atol=1e-6)


def test_patch_rmsnorm_upstream_mode_changes_nothing() -> None:
    """The ``upstream`` mode must leave every forward untouched."""
    model = _tiny_model()
    norms = _randomize_norm_weights(model, seed=5)
    torch.manual_seed(6)
    x = torch.randn(2, 3, model.config.hidden_size)
    expected = model.norm(x)

    record = patch_rmsnorm(model, mode="upstream")

    assert record["applied"] is False
    assert record["patched_modules"] == 0
    assert torch.equal(model.norm(x), expected)
    assert all("forward" not in norm.__dict__ for norm in norms)


def test_patch_rmsnorm_is_scoped_to_the_given_instance() -> None:
    """Patching one model must not change an identically built second model."""
    patched_model = _tiny_model()
    _randomize_norm_weights(patched_model, seed=7)
    reference_model = _tiny_model()
    reference_model.load_state_dict(patched_model.state_dict())
    torch.manual_seed(8)
    x = torch.randn(2, 3, patched_model.config.hidden_size) * 1e3
    expected = reference_model.norm(x)

    patch_rmsnorm(patched_model, mode="scaled", scale=64.0)

    # The reference model keeps the upstream result bit for bit ...
    assert torch.equal(reference_model.norm(x), expected)
    # ... while the patched one takes the replacement path.
    assert "forward" in patched_model.norm.__dict__
    assert torch.allclose(patched_model.norm(x), expected, rtol=1e-5, atol=1e-6)


def test_patch_rmsnorm_rejects_unknown_mode() -> None:
    """An unknown mode must raise without patching any module."""
    model = _tiny_model()
    norms = _randomize_norm_weights(model, seed=9)

    with pytest.raises(ValueError):
        patch_rmsnorm(model, mode="bogus")

    assert all("forward" not in norm.__dict__ for norm in norms)


def test_patch_rmsnorm_rejects_non_positive_scale() -> None:
    """A non-positive scale would divide by zero or flip the sign; it must raise."""
    model = _tiny_model()

    with pytest.raises(ValueError):
        patch_rmsnorm(model, mode="scaled", scale=0.0)


def test_apply_patches_records_every_patch(upstream: SimpleNamespace) -> None:
    """The combined record must describe all three patches."""
    model = _tiny_model()

    record = apply_patches(model, repeat_kv_mode="repeat_interleave", rmsnorm_mode="scaled")

    assert set(record) == {"rotate_half", "repeat_kv", "rmsnorm"}
    assert record["rotate_half"]["applied"] is True
    assert record["repeat_kv"]["applied"] is True
    assert record["repeat_kv"]["mode"] == "repeat_interleave"
    assert record["rmsnorm"]["applied"] is True
    assert record["rmsnorm"]["patched_modules"] > 0
    assert modeling_qwen3.rotate_half is not upstream.rotate_half
    assert modeling_qwen3.repeat_kv is not upstream.repeat_kv


def test_apply_patches_defaults_leave_rmsnorm_upstream(upstream: SimpleNamespace) -> None:
    """By default only the two module functions are replaced."""
    model = _tiny_model()

    record = apply_patches(model)

    assert record["repeat_kv"]["mode"] == "repeat_interleave"
    assert record["rmsnorm"]["applied"] is False
