"""Numerically equivalent monkeypatches that make Qwen3 convertible to Core ML.

The upstream Qwen3 implementation shipped with transformers is written
for eager PyTorch execution and uses three constructs that either fail to
convert or degrade the resulting Core ML model. Each patch below swaps
one of them for a form that computes the same values:

* ``rotate_half`` splits the last dimension with ``x[..., : x.shape[-1] //
  2]``. The Python-level ``//`` on a traced shape becomes an
  ``aten::Int`` node in the TorchScript graph, which the Core ML
  converter cannot fold into a static slice. ``torch.chunk`` splits into
  the same two halves without consulting the shape at graph level, and
  the head dimension is always even so the two halves are identical.
* ``repeat_kv`` expands grouped-query key/value heads through a rank-5
  intermediate (``x[:, :, None].expand(...).reshape(...)``). Rank-5
  tensors are a known trigger for the Neural Engine compiler to reject a
  subgraph and fall back to CPU, so the replacement stays rank-4 while
  keeping the same head order (``head = kv_index * n_rep + rep_index``).
* ``Qwen3RMSNorm.forward`` computes ``x.pow(2).mean(-1)``. Even though it
  casts to float32 first, the converted graph runs in FP16, where any
  element above ~256 squares past the 65504 FP16 maximum and overflows to
  infinity. Dividing the input by a constant before squaring moves the
  whole computation into a safe range.

Applying a patch is recorded in a small dict so the conversion scripts
can store exactly which variants produced a given artifact.

The two function patches replace module attributes and therefore affect
every model in the process, while the RMSNorm patch is applied per module
instance so an unpatched reference model can be kept alongside a patched
one.
"""

from __future__ import annotations

import types
from collections.abc import Callable

import torch
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

# Accepted values for the ``mode`` argument of :func:`patch_repeat_kv`.
REPEAT_KV_MODES: tuple[str, ...] = ("repeat_interleave", "repeat_reshape")

# Accepted values for the ``mode`` argument of :func:`patch_rmsnorm`.
RMSNORM_MODES: tuple[str, ...] = ("upstream", "scaled")


def _rotate_half_chunked(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims, splitting with ``chunk`` instead of slices.

    Args:
        x: Query or key tensor whose last dimension is even.

    Returns:
        ``cat(-x2, x1)`` over the last dimension, identical to the
        upstream result.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv_interleave(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads with ``repeat_interleave``.

    Args:
        hidden_states: Tensor of shape ``(B, kv_heads, S, head_dim)``.
        n_rep: Number of query heads sharing one key/value head.

    Returns:
        Tensor of shape ``(B, kv_heads * n_rep, S, head_dim)``; the input
        itself when ``n_rep == 1``.
    """
    if n_rep == 1:
        return hidden_states
    return hidden_states.repeat_interleave(n_rep, dim=1)


def _repeat_kv_reshape(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads by repeating along the sequence axis.

    ``repeat`` along dim 2 lays the copies out as ``[rep, seq]``, so the
    following reshape moves ``rep`` next to the head axis and yields the
    upstream head order ``kv_index * n_rep + rep_index``.

    Args:
        hidden_states: Tensor of shape ``(B, kv_heads, S, head_dim)``.
        n_rep: Number of query heads sharing one key/value head.

    Returns:
        Tensor of shape ``(B, kv_heads * n_rep, S, head_dim)``; the input
        itself when ``n_rep == 1``.
    """
    if n_rep == 1:
        return hidden_states
    batch, kv_heads, slen, head_dim = hidden_states.shape
    repeated = hidden_states.repeat(1, 1, n_rep, 1)
    return repeated.reshape(batch, kv_heads * n_rep, slen, head_dim)


_REPEAT_KV_IMPLEMENTATIONS: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] = {
    "repeat_interleave": _repeat_kv_interleave,
    "repeat_reshape": _repeat_kv_reshape,
}


def _make_scaled_rmsnorm_forward(scale: float) -> Callable[..., torch.Tensor]:
    """Build an overflow-safe RMSNorm forward for a given scale.

    Normalization by the root mean square is invariant to a constant
    factor -- ``(x / s) / rms(x / s) == x / rms(x)`` -- so dividing the
    input by ``s`` before squaring changes nothing mathematically while
    shrinking the squared values by ``s**2``. Only the epsilon breaks
    that invariance, because it is added to the *scaled* variance; the
    patched forward therefore divides it by ``s**2`` as well, which makes
    the result exactly equivalent to the upstream formula rather than
    merely close to it.

    Args:
        scale: Positive constant the input is divided by.

    Returns:
        A function usable as an ``Qwen3RMSNorm.forward`` bound method.
    """
    eps_scale = scale * scale

    def forward(self: Qwen3RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``hidden_states`` without overflowing FP16."""
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32) / scale
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon / eps_scale)
        return self.weight * x.to(input_dtype)

    return forward


def patch_rotate_half() -> dict[str, object]:
    """Replace the rotary half-split with a constant chunk split.

    Rebinds ``transformers.models.qwen3.modeling_qwen3.rotate_half``;
    the rotary embedding helper looks the name up at call time, so the
    replacement takes effect immediately for every Qwen3 model in the
    process.

    Returns:
        Record of the applied patch.
    """
    modeling_qwen3.rotate_half = _rotate_half_chunked
    return {"patch": "rotate_half", "applied": True}


def patch_repeat_kv(mode: str = "repeat_interleave") -> dict[str, object]:
    """Replace the grouped-query key/value expansion with a rank-4 form.

    Rebinds ``transformers.models.qwen3.modeling_qwen3.repeat_kv`` with
    an implementation that never materializes a rank-5 tensor. Both modes
    produce the upstream head order and, for ``n_rep == 1``, return the
    input unchanged exactly as upstream does.

    Args:
        mode: ``"repeat_interleave"`` (default) or ``"repeat_reshape"``.
            Both are numerically identical; they differ in which
            operators the converter has to lower.

    Returns:
        Record of the applied patch, including the chosen mode.

    Raises:
        ValueError: If ``mode`` is unknown. Nothing is patched in that
            case.
    """
    if mode not in _REPEAT_KV_IMPLEMENTATIONS:
        raise ValueError(f"unknown repeat_kv mode: {mode!r} (expected one of {REPEAT_KV_MODES})")
    modeling_qwen3.repeat_kv = _REPEAT_KV_IMPLEMENTATIONS[mode]
    return {"patch": "repeat_kv", "applied": True, "mode": mode}


def patch_rmsnorm(
    model: torch.nn.Module, mode: str = "upstream", scale: float = 64.0
) -> dict[str, object]:
    """Replace every RMSNorm forward of a model with an overflow-safe form.

    The replacement is bound to each ``Qwen3RMSNorm`` **instance** rather
    than to the class: the accuracy scripts keep an unpatched reference
    model in the same process, and rewriting the class would silently
    change that reference too.

    Args:
        model: Model whose RMSNorm modules are patched (the per-layer
            input/post-attention norms, the query/key norms inside
            attention, and the final norm).
        mode: ``"upstream"`` to leave the model untouched, or
            ``"scaled"`` to divide the input by ``scale`` before
            squaring.
        scale: Positive constant used by the ``"scaled"`` mode. The
            epsilon is divided by ``scale ** 2`` so the result stays
            mathematically identical to the upstream formula.

    Returns:
        Record of the applied patch, including the number of patched
        modules.

    Raises:
        ValueError: If ``mode`` is unknown, or if ``scale`` is not
            positive (zero would divide by zero, a negative value would
            flip the sign of the normalized output). Nothing is patched
            in either case.
    """
    if mode not in RMSNORM_MODES:
        raise ValueError(f"unknown rmsnorm mode: {mode!r} (expected one of {RMSNORM_MODES})")
    if mode == "upstream":
        return {"patch": "rmsnorm", "applied": False, "mode": mode, "patched_modules": 0}
    if not scale > 0.0:
        raise ValueError(f"rmsnorm scale must be positive, got {scale!r}")

    forward = _make_scaled_rmsnorm_forward(scale)
    patched = 0
    for module in model.modules():
        if isinstance(module, Qwen3RMSNorm):
            module.forward = types.MethodType(forward, module)
            patched += 1
    return {
        "patch": "rmsnorm",
        "applied": True,
        "mode": mode,
        "scale": scale,
        "patched_modules": patched,
    }


def apply_patches(
    model: torch.nn.Module,
    *,
    repeat_kv_mode: str = "repeat_interleave",
    rmsnorm_mode: str = "upstream",
    rmsnorm_scale: float = 64.0,
) -> dict[str, object]:
    """Apply every conversion patch and return the combined record.

    ``rotate_half`` and ``repeat_kv`` are always replaced -- without them
    the model does not convert at all -- while the RMSNorm treatment is
    opt-in through ``rmsnorm_mode``.

    Args:
        model: Model to patch; only its RMSNorm modules are touched, the
            other two patches are process-wide module rebinds.
        repeat_kv_mode: Mode forwarded to :func:`patch_repeat_kv`.
        rmsnorm_mode: Mode forwarded to :func:`patch_rmsnorm`.
        rmsnorm_scale: Scale forwarded to :func:`patch_rmsnorm`.

    Returns:
        Records of all three patches, keyed by patch name, ready to be
        stored in the metadata of the converted artifact.

    Raises:
        ValueError: If either mode is unknown.
    """
    return {
        "rotate_half": patch_rotate_half(),
        "repeat_kv": patch_repeat_kv(repeat_kv_mode),
        "rmsnorm": patch_rmsnorm(model, mode=rmsnorm_mode, scale=rmsnorm_scale),
    }
