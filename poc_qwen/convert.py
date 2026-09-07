"""Core ML conversion primitives for the decoder-style embedding experiment.

Every artifact this package produces is built by the same four steps, and
they live here so the driver scripts cannot drift apart in how they run
them:

1. :func:`trace_model` records the PyTorch wrapper into a TorchScript
   graph with a fixed-shape example input.
2. :func:`convert_model` turns that graph into a Core ML ``mlprogram``
   whose ``input_ids`` and ``attention_mask`` inputs have a fixed
   ``(B, S)`` shape and int32 dtype.
3. :func:`compile_model` runs the Xcode command line compiler to turn the
   saved ``.mlpackage`` into the ``.mlmodelc`` directory the runtime
   loads.
4. :func:`resolve_output_key` reports the name the converter actually
   gave the single graph output, which is not always the requested one.

Nothing here is specific to a particular model architecture: the
architecture-dependent parts (which pooling to apply, which patches to
install) belong to the wrapper handed to :func:`trace_model`.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

# Minimum deployment targets selectable from the command line. macOS 13 is
# the oldest target that supports the ``mlprogram`` model type together
# with the CPU_AND_NE compute unit.
TARGETS: dict[str, ct.target] = {"macos13": ct.target.macOS13, "macos15": ct.target.macOS15}

# Compute precisions selectable from the command line. FP16 is what the
# Neural Engine executes natively; FP32 exists to separate precision
# problems from conversion problems when a result looks wrong.
PRECISIONS: dict[str, ct.precision] = {
    "fp16": ct.precision.FLOAT16,
    "fp32": ct.precision.FLOAT32,
}


def trace_model(wrapper: torch.nn.Module, example: dict[str, np.ndarray]) -> torch.jit.ScriptModule:
    """Trace a wrapper module with a tokenized fixed-shape example input.

    Args:
        wrapper: Module taking ``(input_ids, attention_mask)`` positionally
            and returning a single tensor.
        example: Tokenized example as returned by
            ``poc_qwen.common.tokenize_batch``; its ``input_ids`` and
            ``attention_mask`` entries fix the ``(B, S)`` shape of the
            traced graph.

    Returns:
        The traced module in eval mode.
    """
    # The example arrays are int32 because that is the dtype the Core ML
    # inputs are declared with in convert_model(), but nn.Embedding needs
    # int64 indices to run at trace time, so they are widened here. The
    # converter inserts the int32-to-int64 cast on its side.
    input_ids = torch.from_numpy(example["input_ids"]).long()
    attention_mask = torch.from_numpy(example["attention_mask"]).long()
    with torch.no_grad():
        # strict=False: the wrapped model returns tuples whose length the
        # tracer cannot prove is constant, which is fine for a graph with
        # a single tensor output.
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
    """Convert a traced module into an in-memory Core ML program.

    Args:
        traced: Module produced by :func:`trace_model`.
        seq_len: Fixed sequence length ``S``.
        precision: Key into :data:`PRECISIONS`.
        target: Key into :data:`TARGETS`.
        output_name: Name requested for the single graph output.
        batch_size: Fixed batch size ``B``. Both the batch and the
            sequence length are baked into the artifact, because a Neural
            Engine model is compiled for one concrete input shape.

    Returns:
        The converted model, not yet written to disk.

    Raises:
        ValueError: If ``precision`` or ``target`` is unknown.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision: {precision!r} (expected one of {tuple(PRECISIONS)})")
    if target not in TARGETS:
        raise ValueError(f"unknown target: {target!r} (expected one of {tuple(TARGETS)})")
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


def compile_model(mlpackage_path: Path, mlmodelc_path: Path) -> Path:
    """Compile an ``.mlpackage`` into an ``.mlmodelc`` directory.

    The compiler names its output after the input package, so compilation
    runs in a staging directory and the result is moved to the requested
    path; that keeps the artifact naming under the caller's control.

    Args:
        mlpackage_path: Existing ``.mlpackage`` path.
        mlmodelc_path: Destination ``.mlmodelc`` path; an existing
            directory at that path is replaced.

    Returns:
        ``mlmodelc_path``.

    Raises:
        RuntimeError: If the Xcode command line tools are unavailable, if
            the compiler exited with an error, or if it produced no
            ``.mlmodelc`` directory.
    """
    mlmodelc_path.parent.mkdir(parents=True, exist_ok=True)
    # The staging directory is created next to the destination so that
    # moving the compiled directory stays a rename within one filesystem.
    staging = Path(tempfile.mkdtemp(prefix=f".{mlmodelc_path.name}.", dir=mlmodelc_path.parent))
    command = ["xcrun", "coremlcompiler", "compile", str(mlpackage_path), str(staging)]
    try:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"coremlcompiler failed with exit code {exc.returncode} "
                f"while compiling {mlpackage_path}: {(exc.stderr or '').strip()}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "could not run 'xcrun coremlcompiler'; the Xcode command line tools are "
                f"required to compile {mlpackage_path}: {exc}"
            ) from exc
        produced = sorted(staging.glob("*.mlmodelc"))
        if not produced:
            raise RuntimeError(
                f"coremlcompiler produced no .mlmodelc directory in {staging} "
                f"while compiling {mlpackage_path}"
            )
        if mlmodelc_path.exists():
            shutil.rmtree(mlmodelc_path)
        shutil.move(str(produced[0]), str(mlmodelc_path))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return mlmodelc_path


def resolve_output_key(mlmodel_or_path: ct.models.MLModel | Path | str, expected: str) -> str:
    """Report the name Core ML actually gave the graph output.

    The name requested at conversion time is honoured in practice, but the
    converter is free to rename an output (for example when the requested
    name collides with an internal one), and a prediction dict must be
    indexed with the real name. Reading it from the model specification
    keeps the callers from guessing.

    Args:
        mlmodel_or_path: A model object exposing ``get_spec()``, or a path
            to an ``.mlmodel`` file or ``.mlpackage`` directory. A
            compiled ``.mlmodelc`` directory carries no specification and
            cannot be inspected here.
        expected: Output name requested at conversion time.

    Returns:
        The single output name when the model has exactly one output,
        ``expected`` when the model has several outputs and one of them
        carries that name, and otherwise the first declared output.

    Raises:
        RuntimeError: If the model declares no outputs at all.
    """
    get_spec = getattr(mlmodel_or_path, "get_spec", None)
    spec = get_spec() if callable(get_spec) else ct.utils.load_spec(str(mlmodel_or_path))
    names = [output.name for output in spec.description.output]
    if not names:
        raise RuntimeError("the Core ML model declares no outputs")
    if len(names) == 1:
        return names[0]
    return expected if expected in names else names[0]
