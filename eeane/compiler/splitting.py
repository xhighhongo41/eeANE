"""Splitting a compiled model the Neural Engine would refuse into parts that fit.

Core ML does not place a compiled model above a size limit on the Neural
Engine at all (see :mod:`eeane.ne_limits`): every operation silently runs
on the CPU instead, with neither an error nor a warning. Since the limit
applies to each compiled model on its own, a model that is too large is
cut into several models that each fit, stored side by side as a parts
directory (``<stem>.parts``, see :mod:`eeane.segmented`) and run one
after another, the outputs of one part feeding the next.

The cut is made by ``coremltools.models.utils.bisect_model``, which
splits a converted ML program where the cumulative weight size reaches
half, after an operation with a single output and a single consumer, so
exactly the tensors crossing the cut become the second half's inputs. A
half that still compiles above the limit is cut again, up to
:data:`MAX_SPLIT_DEPTH` times. Two choices in how it is called matter:

* It is handed the *path* of the saved package, not a loaded model, and
  its own correctness check is off. Given a path it only reads the
  model's description and weights, so the model is never compiled or
  loaded for execution; the check would run the full model and both
  halves on the CPU on random inputs, which is slow and holds all three
  in memory at once. Whether the parts reproduce the model is measured
  instead by the self-check, which runs them on real inputs afterwards.
* Its halves are kept as separate compiled models rather than merged
  into one Core ML pipeline model. A pipeline model has been reported to
  put the later of its chunks on the CPU, whereas a separate model is
  placed on a compute device on its own, so each part gets the Neural
  Engine if it fits.

Importing this module pulls in ``coremltools`` (and, through
:mod:`eeane.compiler.conversion`, the rest of the conversion stack); it
belongs to the ``[compile]`` extra like the rest of the compiler.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coremltools as ct

from eeane import ne_limits, segmented
from eeane.compiler import artifacts, conversion
from eeane.compiler.artifacts import CompileError

# Name recorded for how a split model was made.
SPLIT_METHOD = "coremltools_bisect"

# How many times a part may be halved again. Only a half that is still
# too large is split further, so a model ends up with the fewest parts
# that fit, but never more than 2**MAX_SPLIT_DEPTH of them.
MAX_SPLIT_DEPTH = 4

# Base name of the n-th part (1-based) inside a parts directory; the
# compiled model gets ``.mlmodelc`` appended, a kept package ``.mlpackage``.
PART_NAME_FORMAT = "part{index:02d}"

# Where the limit a split was made for came from: the operating system
# that compiled the model, or an explicit ``--ne-size-limit``.
LIMIT_SOURCE_OS = "os"
LIMIT_SOURCE_OVERRIDE = "override"

# Largest number of parts a model may be split into.
_MAX_PARTS = 2**MAX_SPLIT_DEPTH

# Suffix of the work area next to the artifact. The parts are assembled
# there and only moved into place once all of them exist, so a failure
# part-way never leaves a parts directory that looks complete.
_STAGING_SUFFIX = ".split_tmp"

# Sub-directories of the work area: the assembled parts directory, and
# the chunks of every bisection on the way to it.
_STAGED_PARTS_DIRNAME = "parts"
_WORK_DIRNAME = "work"

# Bytes per GiB, for the sizes quoted in messages.
_BYTES_PER_GIB = 1 << 30


@dataclass(frozen=True)
class _Leaf:
    """A chunk that fits the limit and becomes one part.

    Attributes:
        package: The chunk's ``.mlpackage``.
        mlmodelc: The chunk compiled.
        size_bytes: Size of the compiled chunk.
    """

    package: Path
    mlmodelc: Path
    size_bytes: int


def bisect_package(mlpackage_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Cut a saved package into two halves of about equal weight size.

    Args:
        mlpackage_path: The ``.mlpackage`` to cut; it is only read.
        output_dir: Directory the two halves are written into (created
            if needed).

    Returns:
        Paths of the first and the second half,
        ``<output_dir>/<package stem>_chunk1.mlpackage`` and
        ``..._chunk2.mlpackage``.

    Raises:
        CompileError: If coremltools finds no point to cut the model at,
            or did not write both halves.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Looked up at call time, so a test can stand in for coremltools.
        ct.models.utils.bisect_model(
            str(mlpackage_path),
            str(output_dir),
            merge_chunks_to_pipeline=False,
            check_output_correctness=False,
        )
    except ValueError as exc:
        # coremltools' way of saying it found no operation to cut after
        # (or that the model is not a single-function program).
        raise CompileError(
            f"the model '{mlpackage_path.name}' could not be split into two parts: {exc}"
        ) from exc
    chunks = tuple(output_dir / f"{mlpackage_path.stem}_chunk{index}.mlpackage" for index in (1, 2))
    for chunk in chunks:
        if not chunk.is_dir():
            raise CompileError(
                f"splitting '{mlpackage_path.name}' did not produce '{chunk.name}' "
                f"in '{output_dir}'"
            )
    return chunks[0], chunks[1]


def package_io_names(mlpackage_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the names of the tensors a saved package reads and writes.

    The halves of a bisection read the tensors crossing the cut, under
    names coremltools derives from the operations that produce them, so
    they are read from each package rather than predicted.

    Args:
        mlpackage_path: A saved ``.mlpackage``.

    Returns:
        ``(input names, output names)``, each in declaration order.

    Raises:
        CompileError: If the package cannot be read, or declares no input
            or no output.
    """
    try:
        description = ct.utils.load_spec(str(mlpackage_path)).description
        inputs = tuple(str(tensor.name) for tensor in description.input)
        outputs = tuple(str(tensor.name) for tensor in description.output)
    except Exception as exc:  # noqa: BLE001 - any unreadable package is reported alike
        raise CompileError(
            f"cannot read the inputs and outputs of '{mlpackage_path}': {exc}"
        ) from exc
    if not inputs or not outputs:
        raise CompileError(
            f"'{mlpackage_path}' declares no {'inputs' if not inputs else 'outputs'}, "
            "so it cannot be run as a part of a split model"
        )
    return inputs, outputs


def split_oversized_variant(
    mlpackage_path: Path,
    mlmodelc_path: Path,
    *,
    output_name: str,
    limit_bytes: int | None,
    limit_source: str,
    keep_part_packages: bool,
    progress: Callable[[str], None],
) -> dict[str, Any] | None:
    """Replace a freshly compiled variant above the size limit by parts that fit.

    A variant within the limit is left exactly as it was compiled; only a
    parts directory an earlier run left under the same name is removed,
    since it would otherwise be served instead of the fresh artifact.

    A variant above the limit is cut in two, both halves are compiled, and
    every half still above the limit is cut again, until each fits. The
    halves that fit become the parts, in running order: every part of a
    first half runs before any part of the second. They are assembled into
    ``<stem>.parts`` next to the artifact, together with a manifest
    describing which tensors each part reads and writes, and the
    oversized ``<stem>.mlmodelc`` is deleted.

    Args:
        mlpackage_path: The converted ``.mlpackage``; it is only read, and
            left for the caller to keep or delete.
        mlmodelc_path: The variant just compiled from it.
        output_name: Name of the model's output tensor, which the last
            part produces.
        limit_bytes: Largest compiled model the Neural Engine accepts, or
            ``None`` when no limit applies (nothing is ever split then).
        limit_source: Where ``limit_bytes`` came from
            (:data:`LIMIT_SOURCE_OS` or :data:`LIMIT_SOURCE_OVERRIDE`),
            recorded as given.
        keep_part_packages: Keep each part's ``.mlpackage`` in the parts
            directory, next to its compiled model.
        progress: Receives one line announcing the split and one
            describing its result.

    Returns:
        ``None`` when nothing was split. Otherwise a JSON-serializable
        record: the method, the limit and its source, the compiled size
        before, and per part its directory name, compiled size, inputs
        and outputs.

    Raises:
        CompileError: If the model cannot be split into parts that fit --
            no point to cut at, a part still too large after
            :data:`MAX_SPLIT_DEPTH` rounds, a chunk that does not compile,
            or parts that do not chain into the model's output. Neither
            the parts directory nor the oversized artifact survives it.
    """
    parts_dir = artifacts.parts_path_for(mlmodelc_path)
    size_before = ne_limits.compiled_model_size_bytes(mlmodelc_path)
    if limit_bytes is None or size_before <= limit_bytes:
        _remove(parts_dir)
        return None

    progress(
        f"the compiled model ({_gib(size_before)}) is above the Neural Engine size limit "
        f"({ne_limits.describe_size_limit(limit_bytes)}); splitting it into parts that each fit"
    )
    started = time.perf_counter()
    staging = mlmodelc_path.with_suffix(_STAGING_SUFFIX)
    try:
        _remove(staging)
        # Read first: an unreadable package should fail before the work.
        model_inputs, _ = package_io_names(mlpackage_path)
        leaves = _split_until_fit(mlpackage_path, staging / _WORK_DIRNAME, "root", 0, limit_bytes)
        staged_parts, records = _assemble_parts(
            leaves,
            staging / _STAGED_PARTS_DIRNAME,
            output_name=output_name,
            model_inputs=model_inputs,
            keep_part_packages=keep_part_packages,
        )
        _remove(parts_dir)
        shutil.move(str(staged_parts), str(parts_dir))
        _remove(mlmodelc_path)
        _remove(staging)
    except BaseException as exc:
        # Neither a half-made parts directory nor the oversized artifact
        # may be served, or reused by the next run.
        for path in (staging, parts_dir, mlmodelc_path):
            _remove(path, ignore_errors=True)
        if isinstance(exc, CompileError) or not isinstance(exc, Exception):
            raise
        raise CompileError(
            f"failed to split '{mlmodelc_path.name}' into parts that fit the Neural Engine: {exc}"
        ) from exc

    progress(
        f"split into {len(records)} parts "
        f"({', '.join(_gib(record['size_bytes']) for record in records)}) "
        f"in {time.perf_counter() - started:.1f} s"
    )
    return {
        "method": SPLIT_METHOD,
        "limit_bytes": limit_bytes,
        "limit_source": limit_source,
        "size_before_bytes": size_before,
        "parts": records,
    }


def _split_until_fit(
    mlpackage_path: Path, work_dir: Path, label: str, depth: int, limit_bytes: int
) -> list[_Leaf]:
    """Bisect a package and, recursively, every half still above the limit.

    Args:
        mlpackage_path: A package whose compiled form is above the limit.
        work_dir: Directory every bisection writes below.
        label: Name of this package's place in the split tree, unique
            among all bisections of one split.
        depth: How many bisections lie between the converted model and
            this package (0 for the converted model itself).
        limit_bytes: The size limit every part has to fit.

    Returns:
        The halves that fit, depth first: all parts of the first half
        before those of the second.

    Raises:
        CompileError: If a half is still above the limit after
            :data:`MAX_SPLIT_DEPTH` bisections, or cannot be bisected.
        RuntimeError: If a half does not compile.
    """
    node_dir = work_dir / label
    halves = bisect_package(mlpackage_path, node_dir)
    if depth > 0:
        # An intermediate half is not a part: once cut, only its own halves
        # are needed, so it is dropped now rather than holding as much disk
        # space as they do until the split is done.
        shutil.rmtree(mlpackage_path, ignore_errors=True)

    leaves: list[_Leaf] = []
    for index, half in enumerate(halves, start=1):
        compiled = node_dir / f"chunk{index}.mlmodelc"
        conversion.compile_model(half, compiled)
        size = ne_limits.compiled_model_size_bytes(compiled)
        if size <= limit_bytes:
            leaves.append(_Leaf(package=half, mlmodelc=compiled, size_bytes=size))
            continue
        if depth + 1 >= MAX_SPLIT_DEPTH:
            raise CompileError(
                f"a part of the model still compiles to {_gib(size)}, above the Neural Engine "
                f"size limit ({ne_limits.describe_size_limit(limit_bytes)}): {_MAX_PARTS} parts "
                "were not enough. Pass '--ne-size-limit none' to compile it in one piece, "
                "which then runs on the CPU"
            )
        shutil.rmtree(compiled)
        leaves.extend(_split_until_fit(half, work_dir, f"{label}-{index}", depth + 1, limit_bytes))
    return leaves


def _assemble_parts(
    leaves: list[_Leaf],
    parts_dir: Path,
    *,
    output_name: str,
    model_inputs: tuple[str, ...],
    keep_part_packages: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    """Move the parts into one directory, named in running order, and describe them.

    Args:
        leaves: The parts, in running order.
        parts_dir: Directory to create and fill.
        output_name: The model's output tensor.
        model_inputs: The inputs of the converted model as a whole.
        keep_part_packages: Move each part's package along too.

    Returns:
        ``parts_dir`` and the per-part records.

    Raises:
        CompileError: If the parts' declared tensors do not chain from the
            model's inputs to its output.
    """
    parts_dir.mkdir(parents=True)
    segments: list[segmented.Segment] = []
    records: list[dict[str, Any]] = []
    for index, leaf in enumerate(leaves, start=1):
        name = PART_NAME_FORMAT.format(index=index)
        mlmodelc_name = f"{name}.mlmodelc"
        inputs, outputs = package_io_names(leaf.package)
        shutil.move(str(leaf.mlmodelc), str(parts_dir / mlmodelc_name))
        if keep_part_packages:
            shutil.move(str(leaf.package), str(parts_dir / f"{name}.mlpackage"))
        segments.append(segmented.Segment(mlmodelc=mlmodelc_name, inputs=inputs, outputs=outputs))
        records.append(
            {
                "name": mlmodelc_name,
                "size_bytes": leaf.size_bytes,
                "inputs": list(inputs),
                "outputs": list(outputs),
            }
        )

    manifest = segmented.SegmentsManifest(
        output_name=output_name, inputs=model_inputs, segments=tuple(segments)
    )
    try:
        segmented.write_segments(parts_dir, manifest)
    except segmented.SegmentsError as exc:
        raise CompileError(f"the parts of the split model do not chain together: {exc}") from exc
    return parts_dir, records


def _remove(path: Path, *, ignore_errors: bool = False) -> None:
    """Delete a directory tree, a file or a link if anything is at ``path``.

    Args:
        path: What to delete.
        ignore_errors: Swallow a failure to delete (used while already
            handling one).
    """
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError:
        if not ignore_errors:
            raise


def _gib(size_bytes: int) -> str:
    """Format a byte count as GiB for a message."""
    return f"{size_bytes / _BYTES_PER_GIB:.3f} GiB"
