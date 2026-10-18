"""Core ML conversion primitives (ported from poc/).

The trace -> convert -> compile sequence proven by the PoC scripts
(``poc/convert_common.py``) lives here, unchanged in behaviour: the frozen
PoC tree stays a historical record while this module becomes the single
source of truth for the pipeline steps that talk to ``coremltools`` and
``xcrun coremlcompiler``.

The only functional difference against the PoC is
:func:`build_versions_info`, which additionally records the eeANE version
so that compiled artifacts can be invalidated when eeANE itself changes.

Importing this module pulls in ``torch``/``transformers``/``coremltools``;
it therefore requires the ``[compile]`` extra and must never be imported
from the ``eeane serve`` code path (see :mod:`eeane.compiler`).
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

from eeane import __version__

# Minimum deployment targets / compute precisions selectable from the CLI.
TARGETS = {"macos13": ct.target.macOS13, "macos15": ct.target.macOS15}
PRECISIONS = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}


def trace_model(wrapper: torch.nn.Module, example: dict[str, np.ndarray]) -> torch.jit.ScriptModule:
    """Trace the wrapper with a tokenized fixed-shape example input.

    Args:
        wrapper: Traceable module returned by a backend's ``wrap``.
        example: Tokenized example with int32 ``input_ids`` and
            ``attention_mask`` of shape (B, S); its shapes become the
            fixed shapes of the traced graph.

    Returns:
        The traced module, in eval mode.
    """
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
    """Convert the traced module to an in-memory ML program.

    Args:
        traced: Module produced by :func:`trace_model`.
        seq_len: Fixed sequence length S.
        precision: Key into :data:`PRECISIONS`.
        target: Key into :data:`TARGETS`.
        output_name: Name given to the single graph output.
        batch_size: Fixed batch size B. The default of 1 reproduces the
            v0.1-v0.5 artifacts bit-for-bit.

    Returns:
        The converted (not yet saved) ML program.

    Raises:
        KeyError: If ``precision`` or ``target`` is not a known key.
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
        RuntimeError: If the compiler is unavailable, failed, or produced
            no .mlmodelc directory.
    """
    staging = mlmodelc_path.with_suffix(".compile_tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        try:
            result = subprocess.run(
                ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(staging)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            # xcrun missing entirely: Xcode command line tools not installed.
            raise RuntimeError(
                f"cannot run 'xcrun coremlcompiler' ({exc}); the Xcode command line tools "
                "are required to compile Core ML models"
            ) from exc
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

    Returns:
        ``preferred`` when present, otherwise the first returned key.

    Raises:
        RuntimeError: If the model returned no outputs.
    """
    keys = list(prediction)
    if not keys:
        raise RuntimeError("Core ML model returned no outputs")
    # Prefer the requested name but tolerate a renamed output.
    return preferred if preferred in keys else keys[0]


def build_versions_info() -> dict[str, str]:
    """Collect the library/runtime versions recorded in conversion metadata.

    Returns:
        Version strings for eeANE itself plus every library whose change
        can alter a compiled artifact. The ``eeane`` entry is what the
        PoC's version block lacked; it participates in the idempotent-skip
        comparison (:func:`eeane.compiler.pipeline.needs_conversion`).
    """
    return {
        "eeane": __version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
