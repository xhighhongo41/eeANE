"""Tests for the weight quantization step of the Core ML conversion.

``eeane.compiler.conversion.quantize_weights_to_int8`` rewrites a saved
``.mlpackage`` so that only its largest weight tensors are stored as
8-bit integers. Everything here runs on a tiny in-test model: one weight
large enough to be quantized next to one that must be left alone. Nothing
is compiled, so the Core ML compiler is not needed.
"""

from __future__ import annotations

from pathlib import Path

import coremltools as ct
import numpy as np
import pytest
import torch

from eeane.compiler import conversion

# Geometry of the tiny model: an embedding table that dominates the
# package next to a projection far below any threshold used here.
_VOCAB = 2000
_HIDDEN = 64
_OUTPUT = 8
_SEQ_LEN = 4

# Element counts of the two weights.
_LARGE_ELEMENTS = _VOCAB * _HIDDEN
_SMALL_ELEMENTS = _HIDDEN * _OUTPUT

# MIL operations a quantized weight is stored as; which of them appears
# depends on the deployment target the package was converted for.
_QUANTIZED_WEIGHT_OPS = ("constexpr_affine_dequantize", "constexpr_blockwise_shift_scale")


class _TwoWeightModel(torch.nn.Module):
    """An embedding table followed by a small bias-free projection."""

    def __init__(self) -> None:
        """Create the two weights from a fixed seed."""
        super().__init__()
        torch.manual_seed(0)
        self.embedding = torch.nn.Embedding(_VOCAB, _HIDDEN)
        self.projection = torch.nn.Linear(_HIDDEN, _OUTPUT, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Project the mean of the looked-up rows."""
        return self.projection(self.embedding(input_ids).mean(dim=1))


def _save_package(directory: Path, target: str = "macos13") -> Path:
    """Convert the tiny model and save it as an ``.mlpackage``.

    Args:
        directory: Directory the package is written into.
        target: Key into ``conversion.TARGETS``.

    Returns:
        Path of the saved package.
    """
    model = _TwoWeightModel().eval()
    example = torch.zeros((1, _SEQ_LEN), dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, _SEQ_LEN), dtype=np.int32)],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",
        compute_precision=conversion.PRECISIONS["fp16"],
        minimum_deployment_target=conversion.TARGETS[target],
    )
    path = directory / "model.mlpackage"
    mlmodel.save(str(path))
    return path


def _weight_bytes(mlpackage_path: Path) -> int:
    """Return the total size of the weight files inside a package."""
    return sum(p.stat().st_size for p in mlpackage_path.rglob("*.bin") if p.is_file())


def _quantized_weight_count(mlpackage_path: Path) -> int:
    """Count the weights a saved package stores in quantized form."""
    spec = ct.models.MLModel(str(mlpackage_path), skip_model_load=True).get_spec()
    main = spec.mlProgram.functions["main"]
    return sum(
        operation.type in _QUANTIZED_WEIGHT_OPS
        for block in main.block_specializations.values()
        for operation in block.operations
    )


@pytest.mark.parametrize("target", sorted(conversion.TARGETS))
def test_quantization_shrinks_the_weight_file(tmp_path: Path, target: str) -> None:
    """Storing the large weight in 8 bits instead of 16 must shrink the package."""
    path = _save_package(tmp_path, target)
    before = _weight_bytes(path)

    conversion.quantize_weights_to_int8(path, _LARGE_ELEMENTS)

    after = _weight_bytes(path)
    assert before >= _LARGE_ELEMENTS * 2
    assert after < before
    # One byte per element plus the per-row scales, instead of two bytes.
    assert after < _LARGE_ELEMENTS * 1.5


@pytest.mark.parametrize("target", sorted(conversion.TARGETS))
def test_quantization_leaves_a_weight_below_the_threshold_alone(
    tmp_path: Path, target: str
) -> None:
    """Only the weight reaching the element count may be quantized."""
    path = _save_package(tmp_path, target)
    assert _quantized_weight_count(path) == 0

    conversion.quantize_weights_to_int8(path, _LARGE_ELEMENTS)

    assert _quantized_weight_count(path) == 1


def test_the_threshold_includes_a_weight_of_exactly_that_many_elements(tmp_path: Path) -> None:
    """``min_elements`` is inclusive: a weight of exactly that size is quantized."""
    path = _save_package(tmp_path)

    conversion.quantize_weights_to_int8(path, _SMALL_ELEMENTS)

    assert _quantized_weight_count(path) == 2


def test_a_threshold_above_every_weight_quantizes_nothing(tmp_path: Path) -> None:
    """A weight one element short of the threshold must stay as it was."""
    path = _save_package(tmp_path)

    conversion.quantize_weights_to_int8(path, _LARGE_ELEMENTS + 1)

    assert _quantized_weight_count(path) == 0


def test_the_quantized_package_replaces_the_original_in_place(tmp_path: Path) -> None:
    """The result must sit at the same path, with nothing left beside it."""
    path = _save_package(tmp_path)

    conversion.quantize_weights_to_int8(path, _LARGE_ELEMENTS)

    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]
    spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
    assert [tensor.name for tensor in spec.description.input] == ["input_ids"]
    assert [tensor.name for tensor in spec.description.output] == ["output"]


def test_a_failed_quantization_keeps_the_original_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure part-way must neither damage the package nor leave debris."""
    path = _save_package(tmp_path)
    before = sorted((p.relative_to(path), p.stat().st_size) for p in path.rglob("*"))

    def _failing_quantize(mlmodel: object, config: object) -> object:
        raise RuntimeError("quantization failed")

    monkeypatch.setattr(conversion.cto_coreml, "linear_quantize_weights", _failing_quantize)

    with pytest.raises(RuntimeError, match="quantization failed"):
        conversion.quantize_weights_to_int8(path, _LARGE_ELEMENTS)

    assert sorted((p.relative_to(path), p.stat().st_size) for p in path.rglob("*")) == before
    assert [entry.name for entry in tmp_path.iterdir()] == [path.name]


@pytest.mark.parametrize("min_elements", [0, -1])
def test_a_non_positive_element_count_is_rejected(tmp_path: Path, min_elements: int) -> None:
    """A threshold below one element would quantize every weight there is."""
    with pytest.raises(ValueError, match="min_elements"):
        conversion.quantize_weights_to_int8(tmp_path / "absent.mlpackage", min_elements)
