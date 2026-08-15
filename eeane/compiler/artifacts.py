"""Compile artifact layout, naming and records (v0.6実装計画.md §4.3, §4.4).

Everything ``eeane compile`` decides *about* its outputs -- where the
cache lives, what a model directory and a variant are called, which
buckets are compiled, whether an existing variant can be reused, and what
the generated ``[[models]]`` snippet looks like -- is collected here,
separately from the conversion driver (:mod:`eeane.compiler.pipeline`).

These are pure decisions: nothing here loads a model, so the whole module
is unit-testable without ``torch`` or an ANE.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eeane.compiler import sources

# Schema versions of the JSON files written next to the artifacts. Bumped
# whenever a consumer (v0.7 cache auto-resolution) would need to tell old
# and new layouts apart.
METADATA_FORMAT_VERSION = 1
MODEL_INFO_FORMAT_VERSION = 1

# Default sequence-length buckets per model kind (v0.6実装計画.md §4.1);
# they reproduce the v0.4/v0.5 deployed configuration.
DEFAULT_BUCKETS: dict[str, tuple[int, ...]] = {
    "embedding": (128, 512, 1024),
    "reranker": (512, 1024),
}

# Cache layout: <out-dir>/compiled/<model-name>/{tokenizer.json,model_info.json,<stem>.*}
CACHE_SUBDIR = "compiled"
TOKENIZER_FILENAME = "tokenizer.json"
MODEL_INFO_FILENAME = "model_info.json"

# Version keys compared against the recorded metadata when deciding
# whether an existing variant can be reused (v0.6実装計画.md §4.3).
SKIP_VERSION_KEYS: tuple[str, ...] = (
    "python",
    "torch",
    "transformers",
    "coremltools",
    "eeane",
)

# Self-check statuses the pipeline itself acts on. Any other status a
# self-check implementation returns is recorded verbatim and treated as
# non-fatal.
SELFCHECK_STATUS_FAILED = "failed"
SELFCHECK_STATUS_SKIPPED = "skipped"


class CompileError(RuntimeError):
    """Raised when a compile step fails with a user-actionable message."""


@dataclass(frozen=True)
class VariantPlan:
    """One bucket's artifact paths plus whether it must be converted.

    Attributes:
        seq_len: Fixed sequence length S of the variant.
        stem: Artifact base name (e.g. ``s512_b1_eager_macos13``).
        mlpackage_path: Intermediate ``.mlpackage`` path.
        mlmodelc_path: Compiled ``.mlmodelc`` path.
        metadata_path: Variant metadata JSON path.
        convert: ``False`` when an up-to-date artifact already exists.
    """

    seq_len: int
    stem: str
    mlpackage_path: Path
    mlmodelc_path: Path
    metadata_path: Path
    convert: bool


def model_identifier(source: str, model_dir: Path) -> str:
    """Return the model id used in the snippet and ``model_info.json``.

    Args:
        source: The raw ``eeane compile <source>`` argument.
        model_dir: Directory :func:`eeane.compiler.sources.resolve_source`
            resolved it to.

    Returns:
        The Hub id for a downloaded model (``org/name``), otherwise the
        resolved directory name.

    Raises:
        CompileError: If a local source resolves to a directory without a
            usable name (e.g. a filesystem root).
    """
    if _is_hub_source(source):
        return source
    return _local_directory_name(model_dir)


def model_cache_name(source: str, model_dir: Path) -> str:
    """Return the cache directory name for a compile source.

    Hub ids are normalised the way the HuggingFace cache does it
    (``org/name`` -> ``org--name``) so the name stays a single path
    component; local sources keep their directory name.

    Args:
        source: The raw ``eeane compile <source>`` argument.
        model_dir: Directory the source resolved to.

    Returns:
        A single-component directory name.

    Raises:
        CompileError: If a local source resolves to a directory without a
            usable name (e.g. a filesystem root).
    """
    if _is_hub_source(source):
        return source.replace("/", "--")
    return _local_directory_name(model_dir)


def resolve_out_root(out_dir: Path | None, env: Mapping[str, str] | None = None) -> Path:
    """Resolve the cache root directory (``--out-dir`` or the default).

    Args:
        out_dir: Explicit ``--out-dir`` value, or ``None`` for the
            default ``$XDG_CACHE_HOME/eeane`` (``~/.cache/eeane`` when
            unset). A relative ``XDG_CACHE_HOME`` is ignored, as the XDG
            base directory specification requires.
        env: Environment mapping to read ``XDG_CACHE_HOME`` from;
            defaults to ``os.environ`` (a parameter for testability).

    Returns:
        An absolute path (the directory itself need not exist yet).
    """
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve()

    environment = env if env is not None else os.environ
    xdg_cache_home = environment.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache_home) if xdg_cache_home else Path()
    if not xdg_cache_home or not base.is_absolute():
        base = Path.home() / ".cache"
    return (base / "eeane").resolve()


def resolve_buckets(buckets: Sequence[int] | None, kind: str) -> list[int]:
    """Resolve the sequence-length buckets to compile for ``kind``.

    Args:
        buckets: Explicit ``--buckets`` value, or ``None`` for the
            kind-specific default.
        kind: Resolved model kind.

    Returns:
        Ascending, deduplicated bucket lengths.

    Raises:
        CompileError: If ``kind`` has no default bucket set, or the
            explicit list is empty or holds a non-positive length.
    """
    if buckets is None:
        if kind not in DEFAULT_BUCKETS:
            supported = ", ".join(sorted(DEFAULT_BUCKETS))
            raise CompileError(
                f"no default buckets are defined for kind '{kind}' "
                f"(known kinds: {supported}); pass --buckets explicitly"
            )
        return list(DEFAULT_BUCKETS[kind])

    resolved = [int(bucket) for bucket in buckets]
    if not resolved:
        raise CompileError("--buckets must list at least one sequence length")
    non_positive = [bucket for bucket in resolved if bucket <= 0]
    if non_positive:
        raise CompileError(f"bucket lengths must be positive, got {non_positive}")
    # Deduplicate: compiling the same bucket twice would only overwrite
    # the same artifact with itself.
    return sorted(set(resolved))


def variant_stem(seq_len: int, batch_size: int, attn: str, target: str, precision: str) -> str:
    """Build the artifact base name of one variant.

    Follows the PoC naming (v0.6実装計画.md §4.3) so that artifacts
    produced by ``eeane compile`` and by the frozen PoC scripts stay
    recognisably the same: ``s{S}_b{B}_{attn}_{target}`` plus an
    ``_fp32`` suffix, which keeps an fp32 experiment from overwriting the
    fp16 baseline.

    Args:
        seq_len: Fixed sequence length S.
        batch_size: Fixed batch size B.
        attn: Attention implementation that was traced.
        target: Minimum deployment target key.
        precision: Compute precision key.

    Returns:
        The artifact base name, without an extension.
    """
    stem = f"s{seq_len}_b{batch_size}_{attn}_{target}"
    if precision == "fp32":
        stem += "_fp32"
    return stem


def needs_conversion(
    mlmodelc_path: Path,
    metadata_path: Path,
    versions: Mapping[str, str],
    *,
    force: bool = False,
) -> bool:
    """Tell whether a variant must be (re)converted.

    A variant is reusable only when the compiled artifact and its
    metadata both exist, every :data:`SKIP_VERSION_KEYS` entry matches the
    current environment, and the recorded self-check did not fail (a
    failed variant must never be silently reused, v0.6実装計画.md §4.5).

    Args:
        mlmodelc_path: Compiled artifact directory.
        metadata_path: Variant metadata JSON path.
        versions: Version block of the current environment.
        force: ``--force``; short-circuits to ``True``.

    Returns:
        ``True`` when the variant must be converted.
    """
    if force or not mlmodelc_path.is_dir():
        return True
    try:
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # Unreadable metadata: treat the artifact as unknown, not as valid.
        return True
    if not isinstance(recorded, dict):
        return True
    recorded_versions = recorded.get("versions")
    if not isinstance(recorded_versions, dict):
        return True
    if any(recorded_versions.get(key) != versions.get(key) for key in SKIP_VERSION_KEYS):
        return True
    selfcheck = recorded.get("selfcheck")
    return isinstance(selfcheck, dict) and selfcheck.get("status") == SELFCHECK_STATUS_FAILED


def build_config_snippet(
    *,
    model_id: str,
    kind: str,
    tokenizer_path: Path,
    artifacts: Mapping[int, Path],
    normalize: bool = True,
) -> str:
    """Build the ``[[models]]`` TOML snippet for a compiled model.

    Paths are absolutized because the snippet is meant to be pasted into
    a config file living somewhere else entirely (eeANE resolves relative
    config paths against the config file's directory).

    Args:
        model_id: Value of the ``id`` key.
        kind: ``"embedding"`` or ``"reranker"``.
        tokenizer_path: Frozen ``tokenizer.json``.
        artifacts: Bucket length -> compiled ``.mlmodelc`` path.
        normalize: ``normalize`` value for an embedding entry. Never
            emitted for a reranker: the config schema rejects it there.

    Returns:
        A TOML fragment ending in a newline.
    """
    lines = [
        "[[models]]",
        f"id = {_toml_string(model_id)}",
        f"kind = {_toml_string(kind)}",
        f"tokenizer = {_toml_string(str(Path(tokenizer_path).resolve()))}",
    ]
    if kind == "embedding":
        lines.append(f"normalize = {'true' if normalize else 'false'}")
    lines += ["", "[models.artifacts]"]
    lines += [
        f"{bucket} = {_toml_string(str(Path(artifacts[bucket]).resolve()))}"
        for bucket in sorted(artifacts)
    ]
    return "\n".join(lines) + "\n"


def write_config_snippet(path: Path, snippet: str) -> None:
    """Write a config snippet to ``path`` (``--emit-config``).

    Args:
        path: Destination file; parent directories are created.
        snippet: Text produced by :func:`build_config_snippet`.

    Raises:
        CompileError: If the destination cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(snippet, encoding="utf-8")
    except OSError as exc:
        raise CompileError(f"cannot write the config snippet to '{path}': {exc}") from exc


def write_json_record(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON record with the project's formatting conventions.

    Args:
        path: Destination file.
        payload: JSON-serializable mapping.

    Raises:
        CompileError: If the file cannot be written, or the payload (e.g.
            a self-check report) is not JSON-serializable.
    """
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError as exc:
        raise CompileError(f"cannot serialize the record for '{path}': {exc}") from exc
    try:
        path.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        raise CompileError(f"cannot write '{path}': {exc}") from exc


def ensure_writable_directory(path: Path) -> None:
    """Create ``path`` if needed and check that it can be written to.

    Args:
        path: Directory to create.

    Raises:
        CompileError: If the directory cannot be created or is read-only.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CompileError(f"cannot create the output directory '{path}': {exc}") from exc
    if not os.access(path, os.W_OK):
        raise CompileError(f"the output directory '{path}' is not writable")


def _is_hub_source(source: str) -> bool:
    """Tell whether a compile source was resolved from the HuggingFace Hub.

    Mirrors :func:`eeane.compiler.sources.resolve_source`: an existing
    path always wins over the Hub interpretation.

    Args:
        source: The raw ``eeane compile <source>`` argument.

    Returns:
        ``True`` when the source is a Hub id rather than a local path.
    """
    candidate = Path(source).expanduser() if source.strip() else None
    if candidate is not None and candidate.exists():
        return False
    return sources.looks_like_hf_repo_id(source)


def _local_directory_name(model_dir: Path) -> str:
    """Return the resolved directory name of a local model source.

    Args:
        model_dir: Local model directory.

    Returns:
        The directory's own name.

    Raises:
        CompileError: If the directory has no name (a filesystem root),
            which would collapse the cache layout.
    """
    name = Path(model_dir).resolve().name
    if not name:
        raise CompileError(
            f"cannot derive a model name from '{model_dir}'; "
            "pass a model directory rather than a filesystem root"
        )
    return name


def _toml_string(value: str) -> str:
    """Quote a string as a TOML basic string.

    Args:
        value: Raw string (an id or a path).

    Returns:
        The quoted, escaped TOML literal.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'
