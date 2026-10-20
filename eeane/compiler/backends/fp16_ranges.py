"""Power-of-two factors that rescale a model's activations into float16.

A compiled program runs in float16, which ends at 65504 and loses normal
precision below 2**-14, and some models produce activations outside that
range: a residual stream that grows past the maximum, or a branch output
so small that its square underflows. Such a model can be rescaled by
rewriting weights and normalization constants only, because a
normalization undoes a constant factor in its input once its ``eps`` is
scaled by the square of that factor: the function the model computes is
unchanged, while the activations between the rewritten weights and the
normalization move into the float16 range.

Every factor derived here is a power of two. Multiplying or dividing a
binary floating-point number by one changes its exponent only -- neither
float32 nor float16 rounds the mantissa -- so the rescaled weights carry
exactly the precision of the original ones.

This module holds the rules that turn a measured range into such a
factor, the record a measurement is reported in, and the check every
applied factor passes. Where a model is measured and which of its
weights a factor is written into depend on the architecture, and are
left to its backend.

This module is plain arithmetic and imports neither ``torch`` nor
``transformers``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Public surface of this module.
__all__ = [
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
]

# Metadata key under which the FP16 range conditioning is recorded.
FP16_RANGE_CONDITIONING_KEY = "fp16_range_conditioning"

# Peak magnitude the residual stream is brought near by
# :func:`residual_divisor`. Float16 ends at 65504, so a peak around 2**10
# leaves a factor of about 64 for inputs that drive the stream harder than
# the calibration sentences did, while keeping the stream itself far above
# the float16 resolution.
RESIDUAL_PEAK_TARGET = 1024.0

# Root mean square each MLP output is brought near by
# :func:`mlp_output_gain`. The normalization that reads the MLP output
# squares it first; around 1 the squares sit in the middle of the float16
# range, as far from its underflow as from its overflow.
MLP_OUTPUT_RMS_TARGET = 1.0

# Peak magnitude the input of an RMS normalization is brought to or under
# by :func:`norm_input_divisor`. The normalization squares its input
# first; the square of 64 is 4096, which leaves a factor of 16 to the
# float16 maximum of 65504.
NORM_INPUT_PEAK_TARGET = 64.0

# Bounds on the exponents of the power-of-two factors. A divisor below 1
# is never used: an activation that already fits float16 is left as it
# is. The upper bounds are far beyond anything a working model calls for
# (they allow factors of 65536); they only keep a degenerate measurement
# from producing a normalization constant that float32 itself can no
# longer hold.
MAX_RESIDUAL_DIVISOR_EXPONENT = 16
MAX_GAIN_EXPONENT = 16


@dataclass(frozen=True)
class ActivationRanges:
    """What a measurement of activation ranges observed on one model.

    Attributes:
        residual_peak: Largest absolute value the residual stream took,
            over every layer, input and position; ``nan`` when any of it
            was not finite, ``0.0`` when nothing was measured.
        mlp_output_rms: Root mean square of each layer's MLP output, in
            layer order, over every input, position and dimension; ``0.0``
            for a layer nothing was measured on.
        inputs: Number of inputs the measurement ran over.
    """

    residual_peak: float
    mlp_output_rms: tuple[float, ...]
    inputs: int


def power_of_two_factor(ratio: float, min_exponent: int, max_exponent: int) -> float:
    """Round ``ratio`` to the nearest power of two within exponent bounds.

    Every factor of the range conditioning is a power of two because
    multiplying or dividing a binary floating-point number by one changes
    its exponent only: neither float32 nor float16 rounds the mantissa,
    so the rescaled weights carry exactly the precision of the original
    ones.

    Args:
        ratio: The factor that would be exact for the measured value.
        min_exponent: Smallest exponent allowed.
        max_exponent: Largest exponent allowed.

    Returns:
        ``2.0 ** e`` with ``e`` the rounded binary logarithm of ``ratio``
        clamped to the bounds; ``1.0`` -- no rescaling -- when ``ratio``
        is not a positive finite number, since nothing can be derived from
        a measurement that is zero, negative, infinite or NaN.
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        return 1.0
    exponent = round(math.log2(ratio))
    return 2.0 ** max(min_exponent, min(max_exponent, exponent))


def residual_divisor(residual_peak: float) -> float:
    """Return the power of two the residual stream is divided by.

    The divisor brings the measured peak of the stream to about
    :data:`RESIDUAL_PEAK_TARGET`. It is never below 1: a stream that is
    already at or under the target is left alone.

    Args:
        residual_peak: Measured peak magnitude of the residual stream.

    Returns:
        The divisor, a power of two of at least 1; ``1.0`` when the peak
        is not a positive finite number.
    """
    if not math.isfinite(residual_peak) or residual_peak <= 0.0:
        return 1.0
    return power_of_two_factor(
        residual_peak / RESIDUAL_PEAK_TARGET, 0, MAX_RESIDUAL_DIVISOR_EXPONENT
    )


def mlp_output_gain(mlp_output_rms: float) -> float:
    """Return the power of two one layer's MLP output is multiplied by.

    The gain brings the measured root mean square of the MLP output to
    about :data:`MLP_OUTPUT_RMS_TARGET`.

    Args:
        mlp_output_rms: Measured root mean square of the layer's MLP
            output.

    Returns:
        The gain, a power of two; ``1.0`` when the measurement is not a
        positive finite number.
    """
    if not math.isfinite(mlp_output_rms) or mlp_output_rms <= 0.0:
        return 1.0
    return power_of_two_factor(
        MLP_OUTPUT_RMS_TARGET / mlp_output_rms, -MAX_GAIN_EXPONENT, MAX_GAIN_EXPONENT
    )


def norm_input_divisor(peak: float, target: float = NORM_INPUT_PEAK_TARGET) -> float:
    """Return the power of two the input of an RMS normalization is divided by.

    An RMS normalization computes ``x * (mean(x**2) + eps) ** -0.5``. Fed
    ``x / d`` with its ``eps`` divided by ``d**2``, it computes exactly
    the same, so its input can be divided by ``d`` -- keeping the squares
    it takes within float16 -- without changing its output.

    Unlike the residual divisor, this one is not rounded to the nearest
    power of two: the target bounds a square, so it is a ceiling rather
    than a level to aim near. The divisor is the smallest power of two
    that brings the peak to or under the target.

    Args:
        peak: Measured peak magnitude of the normalization input.
        target: Peak magnitude the input must not exceed once divided.

    Returns:
        The smallest power of two ``d`` of at least 1 with
        ``peak / d <= target``, or ``2.0 ** MAX_RESIDUAL_DIVISOR_EXPONENT``
        when even that is not enough; ``1.0`` when the peak is not a
        positive finite number.

    Raises:
        ValueError: If ``target`` is not a positive finite number.
    """
    if (
        isinstance(target, bool)
        or not isinstance(target, int | float)
        or not math.isfinite(target)
        or target <= 0.0
    ):
        raise ValueError(f"the target must be a positive finite number (got {target!r})")
    if not math.isfinite(peak) or peak <= 0.0:
        return 1.0
    # The difference of the two logarithms stays finite where the ratio of
    # the two numbers might not.
    exponent = math.ceil(math.log2(peak) - math.log2(target))
    exponent = max(0, min(MAX_RESIDUAL_DIVISOR_EXPONENT, exponent))
    # The logarithms are rounded, so the exponent may be one off at an
    # exact boundary. Dividing by a power of two rounds nothing, so the
    # comparisons below settle it exactly.
    while exponent > 0 and math.ldexp(peak, -(exponent - 1)) <= target:
        exponent -= 1
    while exponent < MAX_RESIDUAL_DIVISOR_EXPONENT and math.ldexp(peak, -exponent) > target:
        exponent += 1
    return 2.0**exponent


def check_power_of_two(name: str, value: float) -> None:
    """Refuse a conditioning factor that is not a positive power of two.

    Args:
        name: How the factor is named in the error message.
        value: The factor to check.

    Raises:
        ValueError: If ``value`` is not finite, not positive, or has a
            mantissa other than that of a power of two.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0.0
        or math.frexp(value)[0] != 0.5
    ):
        raise ValueError(f"{name} must be a positive power of two (got {value!r})")
