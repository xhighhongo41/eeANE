"""A compiled model stored as several parts, run one after another.

The Neural Engine refuses a compiled model above a size limit (see
:mod:`eeane.ne_limits`) and silently runs it on the CPU instead. A model
that is too large is therefore stored as several compiled parts, each of
which fits, in one directory (``<stem>.parts``) next to a manifest
(``segments.json``). The parts are run in order: the model's own inputs
and the outputs of the earlier parts feed the later parts, matched by
tensor name, and the last part yields the model's output. The manifest
records, for every part, which tensors it reads and which it produces, so
a reader needs no knowledge of what the model is.

The loader of one part is injected (:meth:`SegmentedModel.load`), so this
module depends on neither Core ML nor the conversion stack: it imports
only the standard library, and the serving path and the compiler can both
use it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Name of the manifest inside a parts directory.
SEGMENTS_FILENAME = "segments.json"

# Layout version of the manifest. A reader refuses a manifest with a
# higher number: it may give the keys another meaning.
SEGMENTS_FORMAT_VERSION = 1


class SegmentsError(ValueError):
    """A parts directory or its manifest cannot be used."""


@dataclass(frozen=True)
class Segment:
    """One compiled part of a segmented model.

    Attributes:
        mlmodelc: Name of the part's ``.mlmodelc`` directory inside the
            parts directory (a single path component).
        inputs: Names of the tensors the part reads; each is an input of
            the whole model or an output of an earlier part.
        outputs: Names of the tensors the part produces.
    """

    mlmodelc: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class SegmentsManifest:
    """Description of a segmented model: its parts and how data flows.

    Attributes:
        output_name: The tensor that is the model's output; one of the
            last part's outputs.
        inputs: Names of the inputs of the whole model.
        segments: The parts, in the order they run.
    """

    output_name: str
    inputs: tuple[str, ...]
    segments: tuple[Segment, ...]

    def to_json(self) -> dict[str, Any]:
        """Return the manifest as plain JSON types.

        Returns:
            A dict holding the format version, the output name, the model
            inputs and, per part, its directory name, inputs and outputs.
        """
        return {
            "format_version": SEGMENTS_FORMAT_VERSION,
            "output_name": self.output_name,
            "inputs": list(self.inputs),
            "segments": [
                {
                    "mlmodelc": segment.mlmodelc,
                    "inputs": list(segment.inputs),
                    "outputs": list(segment.outputs),
                }
                for segment in self.segments
            ],
        }

    @classmethod
    def from_json(cls, payload: Any, *, source: str) -> SegmentsManifest:
        """Validate a decoded manifest and build the dataclasses from it.

        Besides the shape, the data flow is checked: every input of a
        part must be an input of the model or an output of an earlier
        part, and the model's output must be among the last part's
        outputs.

        Args:
            payload: The decoded JSON document.
            source: Where the payload came from (a file path), named in
                every error message.

        Returns:
            The validated manifest.

        Raises:
            SegmentsError: If the payload is not a usable manifest, or
                was written by a newer eeANE.
        """
        if not isinstance(payload, dict):
            raise _fail(source, "the parts manifest must be a JSON object")

        version = payload.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise _fail(source, "has no usable 'format_version'")
        if version > SEGMENTS_FORMAT_VERSION:
            raise _fail(
                source,
                f"has format_version {version}: these parts were written by a newer eeANE "
                f"(this release reads up to {SEGMENTS_FORMAT_VERSION}); upgrade eeANE or "
                "recompile the model with this release",
            )

        output_name = _name(payload.get("output_name"), "'output_name'", source)
        model_inputs = _names(payload.get("inputs"), "'inputs'", source)

        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise _fail(source, "'segments' must be a non-empty list")

        segments: list[Segment] = []
        available = set(model_inputs)
        for position, raw in enumerate(raw_segments, start=1):
            where = f"segment {position}"
            if not isinstance(raw, dict):
                raise _fail(source, f"{where} must be an object")
            mlmodelc = _part_directory_name(raw.get("mlmodelc"), where, source)
            if any(mlmodelc == earlier.mlmodelc for earlier in segments):
                raise _fail(source, f"{where}: '{mlmodelc}' is used by more than one part")
            inputs = _names(raw.get("inputs"), f"{where}: 'inputs'", source)
            outputs = _names(raw.get("outputs"), f"{where}: 'outputs'", source)
            for name in inputs:
                if name not in available:
                    raise _fail(
                        source,
                        f"{where} ('{mlmodelc}') reads '{name}', which is neither an input "
                        "of the model nor an output of an earlier part",
                    )
            available.update(outputs)
            segments.append(Segment(mlmodelc=mlmodelc, inputs=inputs, outputs=outputs))

        if output_name not in segments[-1].outputs:
            raise _fail(
                source,
                f"the model output '{output_name}' is not an output of the last part "
                f"('{segments[-1].mlmodelc}')",
            )
        return cls(output_name=output_name, inputs=model_inputs, segments=tuple(segments))


def _fail(source: str, detail: str) -> SegmentsError:
    """Build the error for an unusable manifest, naming where it came from."""
    return SegmentsError(f"{source}: {detail}")


def _name(value: Any, what: str, source: str) -> str:
    """Return ``value`` if it is a non-empty string, else raise."""
    if not isinstance(value, str) or not value:
        raise _fail(source, f"{what} must be a non-empty string")
    return value


def _names(value: Any, what: str, source: str) -> tuple[str, ...]:
    """Return a non-empty list of distinct non-empty strings as a tuple."""
    if not isinstance(value, list) or not value:
        raise _fail(source, f"{what} must be a non-empty list of names")
    names = tuple(_name(item, f"every entry of {what}", source) for item in value)
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        raise _fail(source, f"{what} names {', '.join(repr(n) for n in repeated)} more than once")
    return names


def _part_directory_name(value: Any, where: str, source: str) -> str:
    """Return ``value`` if it names a single directory inside the parts directory.

    A manifest is data read from disk, so a name that could point outside
    the parts directory (a separator, ``..``) is refused rather than
    joined onto a path.
    """
    name = _name(value, f"{where}: 'mlmodelc'", source)
    if "/" in name or "\\" in name or "\0" in name or name in (".", ".."):
        raise _fail(source, f"{where}: 'mlmodelc' must be a plain directory name, not {name!r}")
    return name


def is_segmented(path: Path) -> bool:
    """Tell whether ``path`` is a parts directory.

    Args:
        path: A compiled model (``.mlmodelc``) or a parts directory.

    Returns:
        ``True`` if it is a directory holding a manifest file. Neither the
        manifest nor the parts are read.
    """
    try:
        return path.is_dir() and (path / SEGMENTS_FILENAME).is_file()
    except OSError:
        return False


def load_segments(parts_dir: Path) -> SegmentsManifest:
    """Read and validate the manifest of a parts directory.

    Args:
        parts_dir: Directory holding ``segments.json`` and the parts.

    Returns:
        The manifest, once every part it lists exists as a directory.

    Raises:
        SegmentsError: If the manifest cannot be read, is not valid UTF-8
            JSON, fails validation, or lists a part that is not there.
    """
    path = parts_dir / SEGMENTS_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise _fail(str(path), f"the parts manifest cannot be read ({exc})") from exc
    except ValueError as exc:  # not UTF-8
        raise _fail(str(path), f"the parts manifest is not valid UTF-8 text ({exc})") from exc
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise _fail(str(path), f"the parts manifest is not valid JSON ({exc})") from exc

    manifest = SegmentsManifest.from_json(payload, source=str(path))
    for part in part_paths(parts_dir, manifest):
        if not part.is_dir():
            raise _fail(str(path), f"lists the part '{part.name}', which is not in '{parts_dir}'")
    return manifest


def write_segments(parts_dir: Path, manifest: SegmentsManifest) -> Path:
    """Write the manifest of a parts directory.

    Args:
        parts_dir: Existing directory to write ``segments.json`` into.
        manifest: The manifest. It is validated the way a manifest read
            from disk is, so an unusable one is never written.

    Returns:
        Path of the written file.

    Raises:
        SegmentsError: If the manifest is not usable.
        OSError: If the file cannot be written.
    """
    path = parts_dir / SEGMENTS_FILENAME
    payload = manifest.to_json()
    SegmentsManifest.from_json(payload, source=str(path))
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def part_paths(parts_dir: Path, manifest: SegmentsManifest) -> list[Path]:
    """Return the compiled-part directories of a manifest, in running order.

    Args:
        parts_dir: Directory the parts live in.
        manifest: The parts' manifest.

    Returns:
        ``parts_dir / segment.mlmodelc`` for every segment.
    """
    return [parts_dir / segment.mlmodelc for segment in manifest.segments]


class SegmentedModel:
    """Loaded parts of a segmented model, run one after another.

    Each part is an object with a ``predict(feed) -> dict`` method, such
    as a loaded Core ML model; this class neither loads nor knows what
    they are.
    """

    def __init__(self, parts: Sequence[Any], manifest: SegmentsManifest) -> None:
        """Pair loaded parts with the manifest that describes them.

        Args:
            parts: One loaded part per segment, in the manifest's order.
            manifest: The manifest of the parts.

        Raises:
            ValueError: If the number of parts differs from the number of
                segments, or there is none.
        """
        if len(parts) != len(manifest.segments):
            raise ValueError(
                f"the manifest lists {len(manifest.segments)} parts but {len(parts)} were given"
            )
        if not parts:
            raise ValueError("a segmented model needs at least one part")
        self._parts = tuple(parts)
        self._manifest = manifest

    @classmethod
    def load(cls, parts_dir: Path, load_part: Callable[[Path], Any]) -> SegmentedModel:
        """Load every part of a parts directory.

        Args:
            parts_dir: Directory holding ``segments.json`` and the parts.
            load_part: Loads one compiled part from its ``.mlmodelc``
                path. Called once per part, in running order, and only
                after the manifest has been read and validated.

        Returns:
            The segmented model.

        Raises:
            SegmentsError: If the manifest is not usable.
        """
        manifest = load_segments(parts_dir)
        parts = [load_part(path) for path in part_paths(parts_dir, manifest)]
        return cls(parts, manifest)

    @property
    def parts(self) -> tuple[Any, ...]:
        """The loaded parts, in running order."""
        return self._parts

    @property
    def manifest(self) -> SegmentsManifest:
        """The manifest describing the parts."""
        return self._manifest

    def predict(
        self,
        inputs: Mapping[str, Any],
        retain: Callable[[Any, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run the parts one after another.

        Every part is handed only the tensors its segment declares,
        picked from the model's inputs and the outputs of the earlier
        parts.

        Core ML releases the arrays it was handed from a thread of its
        own after a call returns, so the caller has to keep them alive
        until then. ``retain`` is how: it is called after every part's
        call -- also when that call raises -- with the part and the exact
        feed it got, and therefore sees every array handed to every part,
        the intermediate outputs of earlier parts included. Keeping only
        the model's own inputs alive would leave those intermediates to
        be freed while Core ML may still touch them.

        Args:
            inputs: The model's inputs by name.
            retain: Optional callback receiving ``(part, feed)`` after
                each part's call.

        Returns:
            The last part's prediction.

        Raises:
            ValueError: If a part needs a tensor that neither the caller
                gave nor an earlier part produced.
        """
        environment: dict[str, Any] = dict(inputs)
        prediction: Mapping[str, Any] = {}
        for part, segment in zip(self._parts, self._manifest.segments, strict=True):
            missing = [name for name in segment.inputs if name not in environment]
            if missing:
                raise ValueError(
                    f"part '{segment.mlmodelc}' needs {', '.join(repr(n) for n in missing)}, "
                    "which neither the caller gave nor an earlier part produced"
                )
            feed = {name: environment[name] for name in segment.inputs}
            try:
                prediction = part.predict(feed)
            finally:
                if retain is not None:
                    retain(part, feed)
            environment.update(prediction)
        return dict(prediction)
