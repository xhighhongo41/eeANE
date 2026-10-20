"""Tests for the power-of-two factors that rescale activations into float16.

Three layers:

* The rounding rule every factor goes through, and the two factors
  derived from it: the divisor of the residual stream and the gain of an
  MLP output.
* The divisor of a normalization input, which is not rounded but chosen
  as the smallest power of two that brings a peak to or under its target,
  checked on fixed values, at exact boundaries and over a sweep.
* The acceptance check every applied factor passes, the measurement
  record, and the names the Gemma 3 backend re-exports from this module.

Everything here is plain arithmetic: no model is built.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from eeane.compiler.backends import fp16_ranges
from eeane.compiler.backends import gemma3 as g3

# Peaks of the sweep over which the properties of the normalization input
# divisor are checked: twenty steps per decade from 1e-3 to 1e6.
_SWEEP_PEAKS = tuple(10.0 ** (step / 20) for step in range(-60, 121))

# Names the Gemma 3 backend takes from this module and re-exports.
_REEXPORTED_NAMES = (
    "FP16_RANGE_CONDITIONING_KEY",
    "MAX_GAIN_EXPONENT",
    "MAX_RESIDUAL_DIVISOR_EXPONENT",
    "MLP_OUTPUT_RMS_TARGET",
    "RESIDUAL_PEAK_TARGET",
    "ActivationRanges",
    "mlp_output_gain",
    "residual_divisor",
)

# Measurements nothing can be derived from.
_UNUSABLE = (0.0, -0.0, -3.0, math.nan, math.inf, -math.inf)


def _is_power_of_two(value: float) -> bool:
    """Tell whether ``value`` is a positive power of two."""
    return value > 0 and math.isfinite(value) and math.frexp(value)[0] == 0.5


# --- the rounding rule and the two rounded factors ---------------------------


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (1.0, 1.0),
        (2.0, 2.0),
        (0.25, 0.25),
        # Either side of the rounding point between 2 and 4, at 2**1.5.
        (2.8, 2.0),
        (2.9, 4.0),
        (3.0, 4.0),
        (0.7, 0.5),
        (1 / 3, 0.25),
        (1000.0, 1024.0),
    ],
)
def test_a_ratio_is_rounded_to_the_nearest_power_of_two(ratio: float, expected: float) -> None:
    """The factor is two to the rounded binary logarithm of the ratio."""
    assert fp16_ranges.power_of_two_factor(ratio, -16, 16) == expected


@pytest.mark.parametrize(
    ("ratio", "min_exponent", "max_exponent", "expected"),
    [
        (1e30, 0, 16, 2.0**16),
        (1e30, -16, 4, 16.0),
        (1e-30, 0, 16, 1.0),
        (1e-30, -16, 16, 2.0**-16),
        (1e-30, -3, 16, 0.125),
        (5.0, 3, 16, 8.0),
    ],
)
def test_the_exponent_is_clamped_to_its_bounds(
    ratio: float, min_exponent: int, max_exponent: int, expected: float
) -> None:
    """A ratio beyond the bounds gets the factor of the bound it passed."""
    assert fp16_ranges.power_of_two_factor(ratio, min_exponent, max_exponent) == expected


@pytest.mark.parametrize("ratio", _UNUSABLE)
def test_an_unusable_ratio_yields_no_rescaling(ratio: float) -> None:
    """No factor is derived from a ratio that is not a positive finite number."""
    assert fp16_ranges.power_of_two_factor(ratio, -16, 16) == 1.0
    # Not even when the bounds exclude 1 itself.
    assert fp16_ranges.power_of_two_factor(ratio, 2, 16) == 1.0


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (1.49e5, 128.0),
        (1024.0, 1.0),
        (2048.0, 2.0),
        (1400.0, 1.0),
        (1500.0, 2.0),
        # A stream that already fits is never scaled up.
        (100.0, 1.0),
        (1e-3, 1.0),
        (1e30, 2.0**fp16_ranges.MAX_RESIDUAL_DIVISOR_EXPONENT),
    ],
)
def test_the_residual_divisor_brings_the_peak_near_its_target(peak: float, expected: float) -> None:
    """The divisor is the power of two nearest to peak / target, and never below 1."""
    assert fp16_ranges.residual_divisor(peak) == expected


@pytest.mark.parametrize(
    ("rms", "expected"),
    [
        (0.002, 512.0),
        (0.125, 8.0),
        (1.0, 1.0),
        (0.9, 1.0),
        (4.0, 0.25),
        (1e-30, 2.0**fp16_ranges.MAX_GAIN_EXPONENT),
        (1e30, 2.0**-fp16_ranges.MAX_GAIN_EXPONENT),
    ],
)
def test_the_mlp_output_gain_brings_the_level_near_its_target(rms: float, expected: float) -> None:
    """The gain is the power of two nearest to target / rms, within its bounds."""
    assert fp16_ranges.mlp_output_gain(rms) == expected


@pytest.mark.parametrize("measured", _UNUSABLE)
def test_an_unusable_measurement_yields_no_rounded_factor(measured: float) -> None:
    """Nothing can be derived from a zero, negative or non-finite measurement."""
    assert fp16_ranges.residual_divisor(measured) == 1.0
    assert fp16_ranges.mlp_output_gain(measured) == 1.0


# --- the divisor of a normalization input ------------------------------------


def test_the_norm_input_target_leaves_room_for_the_square() -> None:
    """The square of the target sits a factor of 16 below the float16 maximum."""
    assert fp16_ranges.NORM_INPUT_PEAK_TARGET == 64.0
    assert 65504.0 / fp16_ranges.NORM_INPUT_PEAK_TARGET**2 > 15.99


@pytest.mark.parametrize(
    ("peak", "expected"),
    [
        (469.0, 8.0),
        (167.0, 4.0),
        (153.0, 4.0),
        (712.0, 16.0),
        (1979.0, 32.0),
        # At the target nothing needs dividing; just above it, it does.
        (64.0, 1.0),
        (64.0001, 2.0),
        (128.0, 2.0),
        (128.0001, 4.0),
        (0.5, 1.0),
    ],
)
def test_the_norm_input_divisor_is_the_smallest_that_reaches_the_target(
    peak: float, expected: float
) -> None:
    """The divisor brings the peak to or under the target, and no smaller one would."""
    assert fp16_ranges.norm_input_divisor(peak) == expected


@pytest.mark.parametrize("peak", _UNUSABLE)
def test_an_unusable_peak_yields_no_norm_input_divisor(peak: float) -> None:
    """Nothing can be derived from a zero, negative or non-finite peak."""
    assert fp16_ranges.norm_input_divisor(peak) == 1.0


@pytest.mark.parametrize(
    ("peak", "target", "expected"),
    [
        (469.0, 32.0, 16.0),
        (100.0, 100.0, 1.0),
        (100.5, 100.0, 2.0),
        (1.0, 0.25, 4.0),
        (0.3, 0.1, 4.0),
    ],
)
def test_the_norm_input_divisor_follows_a_given_target(
    peak: float, target: float, expected: float
) -> None:
    """A target handed in is used instead of the default one."""
    assert fp16_ranges.norm_input_divisor(peak, target) == expected
    assert fp16_ranges.norm_input_divisor(peak, target=target) == expected


@pytest.mark.parametrize("exponent", range(fp16_ranges.MAX_RESIDUAL_DIVISOR_EXPONENT))
def test_the_norm_input_divisor_is_exact_at_every_boundary(exponent: int) -> None:
    """A peak of exactly target * 2**e takes 2**e; the next float above it takes twice that."""
    boundary = fp16_ranges.NORM_INPUT_PEAK_TARGET * 2.0**exponent

    assert fp16_ranges.norm_input_divisor(boundary) == 2.0**exponent
    assert fp16_ranges.norm_input_divisor(math.nextafter(boundary, math.inf)) == 2.0 ** (
        exponent + 1
    )
    assert fp16_ranges.norm_input_divisor(math.nextafter(boundary, 0.0)) == 2.0**exponent


@pytest.mark.parametrize(
    ("peak", "target"),
    [
        (1e30, fp16_ranges.NORM_INPUT_PEAK_TARGET),
        (math.nextafter(64.0 * 2.0**16, math.inf), 64.0),
        # A ratio that does not even fit a float must not overflow.
        (1e300, 1e-300),
    ],
)
def test_the_norm_input_divisor_is_clamped_to_its_bound(peak: float, target: float) -> None:
    """Beyond any working model the divisor stops at its largest allowed exponent."""
    expected = 2.0**fp16_ranges.MAX_RESIDUAL_DIVISOR_EXPONENT

    assert fp16_ranges.norm_input_divisor(peak, target) == expected


@pytest.mark.parametrize("target", [0.0, -0.0, -64.0, math.nan, math.inf, -math.inf, True, "64"])
@pytest.mark.parametrize("peak", [469.0, math.nan])
def test_the_norm_input_divisor_refuses_an_unusable_target(peak: float, target: object) -> None:
    """A target that is not a positive finite number is an error, whatever the peak."""
    with pytest.raises(ValueError, match="target"):
        fp16_ranges.norm_input_divisor(peak, target)  # type: ignore[arg-type]


@pytest.mark.parametrize("peak", _SWEEP_PEAKS)
def test_the_norm_input_divisor_holds_its_properties_over_a_sweep(peak: float) -> None:
    """Over six orders of magnitude the divisor is a power of two, sufficient and minimal."""
    target = fp16_ranges.NORM_INPUT_PEAK_TARGET

    divisor = fp16_ranges.norm_input_divisor(peak)

    assert _is_power_of_two(divisor)
    assert divisor >= 1.0
    assert peak / divisor <= target
    if divisor > 1.0:
        assert peak / (divisor / 2.0) > target


# --- accepting a factor, the record, and the re-exported names ---------------


@pytest.mark.parametrize("value", [1.0, 0.5, 2.0**-10, 1024, 2.0**16])
def test_a_power_of_two_is_accepted(value: float) -> None:
    """A positive power of two, as a float or an integer, passes the check."""
    fp16_ranges.check_power_of_two("a factor", value)


@pytest.mark.parametrize(
    "value",
    [3.0, 6, 0, 0.0, -2.0, -0.5, math.nan, math.inf, -math.inf, True, "2", None],
    ids=repr,
)
def test_anything_else_is_refused(value: object) -> None:
    """Anything but a positive power of two is refused, naming the factor."""
    with pytest.raises(ValueError, match="the divisor must be a positive power of two"):
        fp16_ranges.check_power_of_two("the divisor", value)  # type: ignore[arg-type]


def test_the_ranges_record_is_frozen() -> None:
    """A measurement, once taken, is not edited in place."""
    ranges = fp16_ranges.ActivationRanges(
        residual_peak=1.5e5, mlp_output_rms=(0.1, 0.002), inputs=3
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        ranges.residual_peak = 0.0  # type: ignore[misc]
    assert [field.name for field in dataclasses.fields(ranges)] == [
        "residual_peak",
        "mlp_output_rms",
        "inputs",
    ]


def test_the_ranges_record_can_be_extended() -> None:
    """A frozen subclass may add fields of its own after the shared ones."""

    @dataclasses.dataclass(frozen=True)
    class WiderRanges(fp16_ranges.ActivationRanges):
        norm_input_peaks: tuple[float, ...]

    ranges = WiderRanges(
        residual_peak=469.0, mlp_output_rms=(1.0,), inputs=2, norm_input_peaks=(167.0,)
    )

    assert isinstance(ranges, fp16_ranges.ActivationRanges)
    assert ranges.residual_peak == 469.0
    assert ranges.norm_input_peaks == (167.0,)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ranges.norm_input_peaks = ()  # type: ignore[misc]


def test_the_module_lists_its_public_surface() -> None:
    """Every name the module declares public exists."""
    assert set(fp16_ranges.__all__) == {
        "FP16_RANGE_CONDITIONING_KEY",
        "MAX_GAIN_EXPONENT",
        "MAX_RESIDUAL_DIVISOR_EXPONENT",
        "MLP_OUTPUT_RMS_TARGET",
        "NORM_INPUT_PEAK_TARGET",
        "RESIDUAL_PEAK_TARGET",
        "ActivationRanges",
        "check_power_of_two",
        "mlp_output_gain",
        "norm_input_divisor",
        "power_of_two_factor",
        "residual_divisor",
    }
    assert all(hasattr(fp16_ranges, name) for name in fp16_ranges.__all__)


@pytest.mark.parametrize("name", _REEXPORTED_NAMES)
def test_the_gemma3_backend_re_exports_the_shared_objects(name: str) -> None:
    """The backend hands out the very objects of this module, not copies of them."""
    assert name in g3.__all__
    assert getattr(g3, name) is getattr(fp16_ranges, name)


def test_the_shared_constants_keep_their_values() -> None:
    """Moving the constants must not have changed what they are."""
    assert fp16_ranges.FP16_RANGE_CONDITIONING_KEY == "fp16_range_conditioning"
    assert fp16_ranges.RESIDUAL_PEAK_TARGET == 1024.0
    assert fp16_ranges.MLP_OUTPUT_RMS_TARGET == 1.0
    assert fp16_ranges.MAX_RESIDUAL_DIVISOR_EXPONENT == 16
    assert fp16_ranges.MAX_GAIN_EXPONENT == 16
