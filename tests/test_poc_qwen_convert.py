"""Tests for poc_qwen.convert and the wrapper of poc_qwen.convert_embedding.

No model weights are downloaded: every test builds a small randomly
initialized Qwen3 model from an in-test config. One test converts that
tiny model for real with coremltools; it is the only slow test here and
still finishes in a few seconds.

Both modules under test import coremltools at module level, so the whole
file is skipped when coremltools cannot be imported.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from poc_qwen import common
from poc_qwen.patches import patch_repeat_kv, patch_rotate_half

try:
    import coremltools as ct

    COREMLTOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover - only on a machine without coremltools
    COREMLTOOLS_AVAILABLE = False

if COREMLTOOLS_AVAILABLE:
    # Imported inside the guard because both modules import coremltools at
    # module level; a genuinely broken module under test must still raise
    # here instead of being silently skipped.
    from poc_qwen import convert
    from poc_qwen.convert_embedding import LastTokenEmbeddingWrapper, build_stem, parse_args

pytestmark = pytest.mark.skipif(
    not COREMLTOOLS_AVAILABLE, reason="coremltools is required by the modules under test"
)

# Name requested for the single graph output throughout these tests.
OUTPUT_NAME = "embedding"


def _tiny_model(seed: int = 0) -> Qwen3Model:
    """Build a small randomly initialized Qwen3 model.

    Args:
        seed: Seed applied before construction so weights are reproducible.

    Returns:
        A ``Qwen3Model`` in eval mode, configured exactly like the model
        the conversion script builds: eager attention, no cache, tuple
        outputs.
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
        attn_implementation="eager",
    )
    config.use_cache = False
    config.return_dict = False
    return Qwen3Model(config).eval()


def _left_padded_batch(seq_len: int = 12, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a left-padded ``(2, seq_len)`` batch with two different lengths.

    Args:
        seq_len: Fixed sequence length ``S``.
        seed: Seed applied before drawing the token ids.

    Returns:
        Tuple of ``input_ids`` and the 2-D ``attention_mask``, both int64.
    """
    torch.manual_seed(seed)
    input_ids = torch.randint(1, 256, (2, seq_len))
    attention_mask = torch.ones(2, seq_len, dtype=torch.long)
    for row, pad in enumerate((5, 2)):
        attention_mask[row, :pad] = 0
        input_ids[row, :pad] = 0
    return input_ids, attention_mask


def _right_padded_batch(seq_len: int = 12, seed: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a right-padded ``(2, seq_len)`` batch with two different lengths.

    Args:
        seq_len: Fixed sequence length ``S``.
        seed: Seed applied before drawing the token ids.

    Returns:
        Tuple of ``input_ids`` and the 2-D ``attention_mask``, both int64.
    """
    torch.manual_seed(seed)
    input_ids = torch.randint(1, 256, (2, seq_len))
    attention_mask = torch.ones(2, seq_len, dtype=torch.long)
    for row, length in enumerate((7, 10)):
        attention_mask[row, length:] = 0
        input_ids[row, length:] = 0
    return input_ids, attention_mask


@pytest.fixture
def patched_qwen3() -> Iterator[None]:
    """Apply the module-level Qwen3 patches and restore them afterwards.

    ``patch_rotate_half`` and ``patch_repeat_kv`` rebind names in the
    transformers module, which affects every Qwen3 model in the process.
    The originals are captured here and put back on teardown so the
    patches cannot leak into other tests.
    """
    original_rotate_half = modeling_qwen3.rotate_half
    original_repeat_kv = modeling_qwen3.repeat_kv
    patch_rotate_half()
    patch_repeat_kv()
    try:
        yield
    finally:
        modeling_qwen3.rotate_half = original_rotate_half
        modeling_qwen3.repeat_kv = original_repeat_kv


def _fake_model_with_outputs(*names: str) -> Any:
    """Build a stand-in exposing ``get_spec()`` with the given output names.

    Args:
        names: Output names the fake specification declares, in order.

    Returns:
        An object shaped like the part of ``ct.models.MLModel`` that
        :func:`poc_qwen.convert.resolve_output_key` reads.
    """
    spec = SimpleNamespace(
        description=SimpleNamespace(output=[SimpleNamespace(name=name) for name in names])
    )
    return SimpleNamespace(get_spec=lambda: spec)


def test_wrapper_matches_the_plain_two_dimensional_mask_path() -> None:
    """The in-graph 4-D mask must reproduce the framework's own mask path."""
    model = _tiny_model()
    input_ids, attention_mask = _left_padded_batch()
    assert bool((attention_mask == 0).any())  # the batch really exercises padding

    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        expected = common.last_token_pool_left(hidden)
        pooled = LastTokenEmbeddingWrapper(model)(input_ids, attention_mask)

    assert pooled.shape == (input_ids.shape[0], model.config.hidden_size)
    assert torch.allclose(pooled, expected, atol=1e-5)


def test_wrapper_does_not_normalize_its_output() -> None:
    """Pooled vectors are raw: normalization is the caller's responsibility."""
    model = _tiny_model()
    input_ids, attention_mask = _left_padded_batch()

    with torch.no_grad():
        pooled = LastTokenEmbeddingWrapper(model)(input_ids, attention_mask)

    norms = torch.linalg.norm(pooled, dim=1)
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_wrapper_is_traceable_and_the_traced_graph_reproduces_the_output(
    patched_qwen3: None,
) -> None:
    """Tracing must not change what the wrapper computes."""
    model = _tiny_model()
    wrapper = LastTokenEmbeddingWrapper(model).eval()
    input_ids, attention_mask = _left_padded_batch()

    with torch.no_grad():
        expected = wrapper(input_ids, attention_mask)
        traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)
        traced_output = traced(input_ids, attention_mask)

    assert torch.allclose(traced_output, expected, atol=1e-6)


def test_traced_graph_contains_no_aten_int_nodes(patched_qwen3: None) -> None:
    """The traced graph must be free of ``aten::Int`` shape arithmetic.

    An ``aten::Int`` node is what a Python-level ``//`` or index on a
    traced shape lowers to, and the Core ML converter cannot fold it into
    a static slice. Its absence is the mechanical evidence that the
    conversion patches are in effect.
    """
    model = _tiny_model()
    wrapper = LastTokenEmbeddingWrapper(model).eval()
    input_ids, attention_mask = _left_padded_batch()

    traced = convert.trace_model(
        wrapper,
        {
            "input_ids": input_ids.numpy().astype(np.int32),
            "attention_mask": attention_mask.numpy().astype(np.int32),
        },
    )

    assert "aten::Int" not in str(traced.inlined_graph)


def test_gather_right_pooling_selects_the_last_real_token() -> None:
    """Right-padded rows must be pooled at their own last real position."""
    model = _tiny_model()
    input_ids, attention_mask = _right_padded_batch()
    lengths = attention_mask.sum(dim=1)
    assert lengths[0] != lengths[1]  # the two rows must pool at different indices

    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        expected = common.last_token_pool_gather(hidden, attention_mask)
        pooled = LastTokenEmbeddingWrapper(model, pooling="gather-right")(input_ids, attention_mask)

    assert pooled.shape == (input_ids.shape[0], model.config.hidden_size)
    assert torch.allclose(pooled, expected, atol=1e-5)
    # The gathered row must differ from the plain last position, otherwise
    # the test would pass even with left-padding pooling.
    assert not torch.allclose(expected, common.last_token_pool_left(hidden), atol=1e-3)


def test_wrapper_rejects_unknown_pooling() -> None:
    """An unknown pooling mode must fail at construction time."""
    model = _tiny_model()

    with pytest.raises(ValueError):
        LastTokenEmbeddingWrapper(model, pooling="bogus")


@pytest.mark.parametrize(
    ("precision", "target"),
    [("bogus", "macos13"), ("fp16", "bogus")],
)
def test_convert_model_rejects_unknown_precision_or_target(precision: str, target: str) -> None:
    """Unknown conversion settings must raise before any conversion work."""
    traced = torch.jit.trace(torch.nn.Identity(), torch.zeros(1, 2))

    with pytest.raises(ValueError):
        convert.convert_model(traced, 8, precision, target, OUTPUT_NAME)


def test_trace_model_accepts_int32_numpy_inputs(patched_qwen3: None) -> None:
    """int32 example arrays must be widened to the int64 indices tracing needs."""
    model = _tiny_model()
    wrapper = LastTokenEmbeddingWrapper(model).eval()
    input_ids, attention_mask = _left_padded_batch()
    example = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    }

    traced = convert.trace_model(wrapper, example)

    with torch.no_grad():
        expected = wrapper(input_ids, attention_mask)
        traced_output = traced(input_ids, attention_mask)
    assert traced.training is False
    assert torch.allclose(traced_output, expected, atol=1e-6)


def test_resolve_output_key_returns_the_only_declared_output() -> None:
    """A single-output model is indexed by that output whatever was requested."""
    assert convert.resolve_output_key(_fake_model_with_outputs("var_42"), OUTPUT_NAME) == "var_42"


def test_resolve_output_key_prefers_the_expected_name_among_several() -> None:
    """With several outputs the requested name wins, else the first one."""
    model = _fake_model_with_outputs("logits", OUTPUT_NAME)

    assert convert.resolve_output_key(model, OUTPUT_NAME) == OUTPUT_NAME
    assert convert.resolve_output_key(model, "absent") == "logits"


def test_resolve_output_key_rejects_a_model_without_outputs() -> None:
    """A model declaring no outputs cannot be indexed and must raise."""
    with pytest.raises(RuntimeError):
        convert.resolve_output_key(_fake_model_with_outputs(), OUTPUT_NAME)


def test_compile_model_reports_a_missing_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing ``xcrun`` must surface as an explanatory RuntimeError."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError(2, "No such file or directory: 'xcrun'")

    monkeypatch.setattr(convert.subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="xcrun"):
        convert.compile_model(tmp_path / "m.mlpackage", tmp_path / "out" / "m.mlmodelc")


def test_compile_model_reports_a_failing_compiler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero compiler exit code must surface with its stderr."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, "xcrun", stderr="unsupported operation")

    monkeypatch.setattr(convert.subprocess, "run", _raise)

    with pytest.raises(RuntimeError, match="unsupported operation"):
        convert.compile_model(tmp_path / "m.mlpackage", tmp_path / "m.mlmodelc")


def test_compile_model_reports_an_empty_compiler_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful run that produced nothing must not look like a success."""
    monkeypatch.setattr(convert.subprocess, "run", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="no .mlmodelc"):
        convert.compile_model(tmp_path / "m.mlpackage", tmp_path / "m.mlmodelc")

    # The staging directory must not survive a failure.
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_build_stem_uses_the_bare_form_for_default_options() -> None:
    """Default options must yield the plain ``s{S}_b{B}_{precision}_{target}`` stem."""
    args = parse_args(["--seq-len", "128", "--batch", "2"])

    assert build_stem(args) == "s128_b2_fp16_macos13"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--rmsnorm-mode", "scaled"], "s128_b1_fp16_macos13_rms64"),
        (["--rmsnorm-mode", "scaled", "--rmsnorm-scale", "128"], "s128_b1_fp16_macos13_rms128"),
        (["--pooling", "gather-right"], "s128_b1_fp16_macos13_gather"),
        (["--mask-fill-value", "-30000"], "s128_b1_fp16_macos13_fill-30000"),
        (["--precision", "fp32", "--target", "macos15"], "s128_b1_fp32_macos15"),
    ],
)
def test_build_stem_marks_every_non_default_option(argv: list[str], expected: str) -> None:
    """Any option that changes the artifact must change its name."""
    args = parse_args(["--seq-len", "128", *argv])

    assert build_stem(args) == expected


def test_convert_tiny_model_end_to_end(tmp_path: Path, patched_qwen3: None) -> None:
    """Convert a tiny model for real and compare it with the FP32 reference.

    This is the only test that runs coremltools; it exercises
    :func:`poc_qwen.convert.convert_model` and
    :func:`poc_qwen.convert.resolve_output_key` on a genuine artifact and
    checks that the FP16 program still agrees with PyTorch.
    """
    model = _tiny_model()
    wrapper = LastTokenEmbeddingWrapper(model).eval()
    input_ids, attention_mask = _left_padded_batch()
    batch_size, seq_len = input_ids.shape

    with torch.no_grad():
        # Reference through the framework's own 2-D mask path, i.e. not
        # through the 4-D mask the converted graph carries.
        hidden = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        reference = common.last_token_pool_left(hidden).numpy().astype(np.float32)

    example = {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": attention_mask.numpy().astype(np.int32),
    }
    traced = convert.trace_model(wrapper, example)
    mlmodel = convert.convert_model(
        traced, seq_len, "fp16", "macos13", OUTPUT_NAME, batch_size=batch_size
    )
    mlpackage_path = tmp_path / "tiny.mlpackage"
    mlmodel.save(str(mlpackage_path))

    output_key = convert.resolve_output_key(mlpackage_path, OUTPUT_NAME)
    assert output_key == OUTPUT_NAME

    reloaded = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    prediction = reloaded.predict(example)
    embeddings = np.asarray(prediction[output_key], dtype=np.float32).reshape(batch_size, -1)

    assert embeddings.shape == reference.shape
    assert bool(np.isfinite(embeddings).all())
    cosines = common.cosine_rowwise(embeddings, reference)
    assert float(cosines.min()) >= 0.99
