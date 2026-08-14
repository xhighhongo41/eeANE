"""Shared Core ML conversion pipeline for the eeANE PoC scripts.

Both ``poc/convert_embedding.py`` and ``poc/convert_reranker.py`` share the
same HF-to-Core-ML pipeline: patch -> trace -> convert -> compile -> resolve
output key. This module is the single source of truth for that pipeline so
the two model-specific scripts cannot drift apart (see 開発資料/v0.2実装計画.md
§4.2).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import transformers
from transformers.models.modernbert import modeling_modernbert

TARGETS = {"macos13": ct.target.macOS13, "macos15": ct.target.macOS15}
PRECISIONS = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}

# Captured at import time so patch_eager_attention_rank4() can delegate the
# non-eager attention paths to the untouched upstream implementation, and so
# that applying the patch twice cannot recurse.
_ORIGINAL_ATTENTION_FORWARD = modeling_modernbert.ModernBertAttention.forward


def patch_rotate_half() -> None:
    """Replace ModernBert's ``rotate_half`` with a static-shape equivalent.

    The upstream implementation slices with ``x.shape[-1] // 2``, which
    traces to ``aten::size -> floor_divide -> aten::Int``. coremltools 9.0
    cannot convert that ``aten::Int`` under numpy 2.x and raises
    "only 0-dimensional arrays can be converted to Python scalars"
    (§4.8 C1). ``torch.chunk`` with a constant chunk count yields the
    identical result without any dynamic shape arithmetic; this is exact
    because RoPE head dimensions are always even.
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    modeling_modernbert.rotate_half = rotate_half


def patch_eager_attention_rank4() -> None:
    """Rewrite ModernBert's eager attention without rank-5 intermediates.

    Upstream splits the fused projection with
    ``qkv.view(bs, -1, 3, heads, head_dim).transpose(3, 1).unbind(dim=2)``,
    which materializes rank-5 tensors. At B=1 the leading batch axis is
    degenerate and the ANE compiler copes, but for B>1 loading the compiled
    model on CPU_AND_NE fails with
    "MILCompilerForANE error: failed to compile ANE model using ANEF.
    Error=_ANECompiler : ANECCompile() FAILED." and the whole model silently
    falls back to CPU (where the ``finfo.min`` mask makes the outputs NaN,
    the v0.1 known limitation).

    Because ``Wqkv`` emits the three projections contiguously along the last
    axis with the 3 as the outermost sub-index, ``chunk(3, dim=-1)``
    followed by a rank-4 ``view``/``transpose`` selects exactly the same
    elements as the upstream rank-5 path. The rewrite is therefore
    bit-exact (verified: max |patched - upstream| = 0.0 on ruri-v3-310m)
    and only changes tensor rank, never semantics.

    Only the ``eager`` attention path is rewritten; ``sdpa`` and
    ``flash_attention_2`` keep the upstream implementation, so the FP32
    baselines used by the sanity checks are unaffected.

    Raises:
        ValueError: If a patched module's head geometry contradicts the
            ``heads * head_dim == all_head_size`` layout assumption.
    """

    def eager_attention_rank4(
        module: modeling_modernbert.ModernBertAttention,
        qkv: torch.Tensor,
        attention_mask: torch.Tensor,
        sliding_window_mask: torch.Tensor,
        position_ids: torch.Tensor | None,
        local_attention: tuple[int, int],
        bs: int,
        dim: int,
        output_attentions: bool = False,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Eager attention over a rank-3 ``qkv`` of shape (B, S, 3*H*D)."""
        cos, sin = module.rotary_emb(qkv, position_ids=position_ids)
        heads, head_dim = module.num_heads, module.head_dim
        # (B, S, 3*H*D) -> three (B, H, S, D) tensors, all rank 4.
        query, key, value = (
            part.view(bs, -1, heads, head_dim).transpose(1, 2) for part in qkv.chunk(3, dim=-1)
        )
        query, key = modeling_modernbert.apply_rotary_pos_emb(query, key, cos, sin)

        scale = head_dim**-0.5
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale
        if local_attention != (-1, -1):
            attention_mask = sliding_window_mask
        attn_weights = attn_weights + attention_mask
        # Upstream upcasts the softmax to fp32 before casting back.
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=module.attention_dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, dim)
        if output_attentions:
            return (attn_output, attn_weights)
        return (attn_output,)

    def attention_forward(
        self: modeling_modernbert.ModernBertAttention,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        """Replacement for ``ModernBertAttention.forward``."""
        if self.config._attn_implementation != "eager":
            return _ORIGINAL_ATTENTION_FORWARD(
                self, hidden_states, output_attentions=output_attentions, **kwargs
            )
        if self.all_head_size != self.num_heads * self.head_dim:
            raise ValueError(
                "unexpected ModernBert head geometry "
                f"({self.num_heads} heads x {self.head_dim} != {self.all_head_size}); "
                "the rank-4 qkv split assumption does not hold"
            )
        # The rank-5 view of the upstream forward is skipped entirely.
        attn_outputs = eager_attention_rank4(
            self,
            qkv=self.Wqkv(hidden_states),
            local_attention=self.local_attention,
            bs=hidden_states.shape[0],
            dim=self.all_head_size,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = self.out_drop(self.Wo(attn_outputs[0]))
        return (hidden_states,) + attn_outputs[1:]

    modeling_modernbert.ModernBertAttention.forward = attention_forward


def patch_mask_fill_value(model: torch.nn.Module, fill_value: float) -> None:
    """Override ModernBert attention mask generation with a finite fill value.

    ``ModernBertModel._update_attention_mask`` fills masked positions with
    ``torch.finfo(float32).min``, which becomes ``-inf`` once the graph is
    cast to FP16 and can make softmax produce NaN for fully masked rows.
    This monkeypatch reproduces the same masks with a finite fill value.

    Args:
        model: The loaded ModernBertModel instance (pass the ``.model``
            attribute when the loaded object is a SequenceClassification
            model).
        fill_value: Finite value used for masked positions (e.g. -30000.0).
    """
    local_attention = model.config.local_attention

    def _update(attention_mask: torch.Tensor, output_attentions: bool = False) -> tuple:
        seq_len = attention_mask.shape[-1]
        global_mask = (1.0 - attention_mask[:, None, None, :].float()) * fill_value  # (B,1,1,S)
        rows = torch.arange(seq_len).unsqueeze(0)
        distance = (rows - rows.T).abs()  # (S, S)
        window = (distance <= local_attention // 2)[None, None, :, :]  # (1,1,S,S)
        sliding_window_mask = global_mask.masked_fill(~window, fill_value)  # (B,1,S,S)
        return global_mask, sliding_window_mask

    model._update_attention_mask = _update


def trace_model(wrapper: torch.nn.Module, example: dict[str, np.ndarray]) -> torch.jit.ScriptModule:
    """Trace the wrapper with a tokenized fixed-shape example input."""
    # nn.Embedding requires int64 indices at trace time; Core ML inputs are
    # declared as int32 separately in convert_model().
    input_ids = torch.from_numpy(example["input_ids"]).long()
    attention_mask = torch.from_numpy(example["attention_mask"]).long()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (input_ids, attention_mask), strict=False)
    return traced.eval()


def convert_model(
    traced: torch.jit.ScriptModule,
    seq_len: int,
    precision: str,
    target: str,
    output_name: str,
    batch_size: int = 1,
) -> ct.models.MLModel:
    """Convert the traced module to an in-memory ML program (§4.4).

    Args:
        traced: Module produced by :func:`trace_model`.
        seq_len: Fixed sequence length S.
        precision: Key into :data:`PRECISIONS`.
        target: Key into :data:`TARGETS`.
        output_name: Name given to the single graph output.
        batch_size: Fixed batch size B (v0.3実装計画.md §4.2). The default
            of 1 reproduces the v0.1/v0.2 artifacts bit-for-bit.
    """
    return ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(batch_size, seq_len), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(batch_size, seq_len), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name=output_name)],
        convert_to="mlprogram",
        compute_precision=PRECISIONS[precision],
        minimum_deployment_target=TARGETS[target],
    )


def compile_model(mlpackage_path: Path, mlmodelc_path: Path) -> None:
    """Compile an .mlpackage into an .mlmodelc directory.

    The compiler names its output after the input package, so compilation
    runs in a staging directory and the result is moved to the requested
    path to keep the naming under our control.

    Args:
        mlpackage_path: Existing .mlpackage path.
        mlmodelc_path: Destination .mlmodelc path (replaced if present).

    Raises:
        RuntimeError: If the compiler failed or produced no .mlmodelc directory.
    """
    staging = mlmodelc_path.with_suffix(".compile_tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(staging)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"coremlcompiler failed ({result.returncode}): {result.stderr}")
        produced = sorted(staging.glob("*.mlmodelc"))
        if not produced:
            raise RuntimeError(f"coremlcompiler produced no .mlmodelc in {staging}")
        if mlmodelc_path.exists():
            shutil.rmtree(mlmodelc_path)
        shutil.move(str(produced[0]), str(mlmodelc_path))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def resolve_output_key(prediction: dict[str, Any], preferred: str) -> str:
    """Pick the model output key from a ``predict`` result dict.

    Args:
        prediction: The dict returned by ``CompiledMLModel.predict``.
        preferred: The output name requested at conversion time.

    Raises:
        RuntimeError: If the model returned no outputs.
    """
    keys = list(prediction)
    if not keys:
        raise RuntimeError("Core ML model returned no outputs")
    # Prefer the requested name but tolerate a renamed output.
    return preferred if preferred in keys else keys[0]


def build_versions_info() -> dict[str, str]:
    """Collect the library/runtime versions recorded in conversion metadata."""
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
