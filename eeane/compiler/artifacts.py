"""Compile artifact layout, naming and records.

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
# whenever a consumer (the cache auto-resolution in eeane.config) would
# need to tell old and new layouts apart. model_info.json is at 2 since it
# now also records embedding_dim, recommended_buckets and calibration.
METADATA_FORMAT_VERSION = 1
MODEL_INFO_FORMAT_VERSION = 2

# Default sequence-length buckets per model kind; they reproduce the
# v0.4/v0.5 deployed configuration.
DEFAULT_BUCKETS: dict[str, tuple[int, ...]] = {
    "embedding": (128, 512, 1024),
    "reranker": (512, 1024),
}

# Cache layout: <out-dir>/compiled/<model-name>/{tokenizer.json,model_info.json,<stem>.*}
CACHE_SUBDIR = "compiled"
TOKENIZER_FILENAME = "tokenizer.json"
MODEL_INFO_FILENAME = "model_info.json"

# Version keys compared against the recorded metadata when deciding
# whether an existing variant can be reused.
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

    Follows the PoC naming so that artifacts produced by ``eeane compile``
    and by the frozen PoC scripts stay recognisably the same:
    ``s{S}_b{B}_{attn}_{target}`` plus an ``_fp32`` suffix, which keeps an
    fp32 experiment from overwriting the fp16 baseline.

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
    failed variant must never be silently reused).

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


def discover_variants(
    model_root: Path,
    *,
    batch_size: int,
    attn: str,
    target: str,
    precision: str,
) -> dict[int, Path]:
    """Find previously compiled same-family variants in a model's cache dir.

    An ``eeane compile`` run only converts the buckets it was asked for,
    but the emitted config snippet and ``model_info.json`` describe the
    cache as a whole: adding one bucket (e.g. ``--buckets 2048``) must not
    silently drop the buckets compiled by earlier runs. A variant belongs
    to the family when its recorded batch size, attention implementation,
    deployment target, and precision all match the current invocation and
    its ``.mlmodelc`` still exists on disk.

    Args:
        model_root: The model's cache directory (holding ``s*.json``
            variant metadata next to the ``.mlmodelc`` directories).
        batch_size: Batch size of the current invocation.
        attn: Attention implementation of the current invocation.
        target: Deployment target of the current invocation.
        precision: Compute precision of the current invocation.

    Returns:
        Mapping of bucket length to the existing ``.mlmodelc`` path.
        Unreadable or incomplete metadata simply leaves its variant
        unlisted (fail toward "not present", never an error).
    """
    found: dict[int, Path] = {}
    if not model_root.is_dir():
        return found
    for metadata_path in sorted(model_root.glob("s*.json")):
        try:
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
            variant = recorded["variant"]
            recorded_args = recorded["args"]
            seq_len = int(variant["seq_len"])
            stem = str(variant["stem"])
            matches = (
                int(variant["batch_size"]) == batch_size
                and recorded_args["attn"] == attn
                and recorded_args["target"] == target
                and recorded_args["precision"] == precision
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if not matches:
            continue
        mlmodelc_path = model_root / f"{stem}.mlmodelc"
        if mlmodelc_path.is_dir():
            found[seq_len] = mlmodelc_path
    return found


def aggregate_calibration(
    kind: str,
    cache_artifacts: Mapping[int, Path],
    run_reports: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[int], int | None]:
    """Aggregate every cached bucket's self-check into the model-level summary.

    Reads back the ``selfcheck`` block every same-family bucket already
    carries (this run's own buckets from ``run_reports``, the rest from
    their metadata JSON file next to ``cache_artifacts[seq_len]``) and
    turns it into the three pieces of ``model_info.json`` that describe
    the cache as a whole rather than one bucket.

    Args:
        kind: Resolved model kind (``embedding_dim`` is only ever derived
            for ``"embedding"``).
        cache_artifacts: Bucket -> ``.mlmodelc`` path of every same-family
            variant now present in the cache.
        run_reports: Self-check reports this invocation itself produced,
            keyed by bucket. A bucket this run reused or skipped is not
            in here and is read back from disk instead.

    Returns:
        Tuple of ``(calibration, recommended_buckets, embedding_dim)``:
        the ``calibration`` record, the ascending buckets to recommend
        loading (every cached bucket whose self-check did not report
        ``status="failed"``), and the embedding width (``None`` for a
        reranker, or when no cached bucket recorded one).

    Raises:
        CompileError: If cached embedding buckets disagree on
            ``embedding_dim`` -- a corrupt or hand-edited cache.
    """
    reports: dict[int, Mapping[str, Any] | None] = {
        seq_len: run_reports[seq_len] if seq_len in run_reports else _read_selfcheck(mlmodelc_path)
        for seq_len, mlmodelc_path in cache_artifacts.items()
    }
    buckets = {
        str(seq_len): _calibration_bucket_entry(reports[seq_len]) for seq_len in sorted(reports)
    }
    calibration = {"machine": _pick_machine(reports, run_reports), "buckets": buckets}
    recommended_buckets = sorted(
        seq_len for seq_len in reports if buckets[str(seq_len)]["status"] != SELFCHECK_STATUS_FAILED
    )
    embedding_dim = _resolve_embedding_dim(kind, reports)
    return calibration, recommended_buckets, embedding_dim


def _read_selfcheck(mlmodelc_path: Path) -> Mapping[str, Any] | None:
    """Read back the ``selfcheck`` block of a variant from its metadata file.

    Args:
        mlmodelc_path: Compiled ``.mlmodelc`` path; its metadata JSON is
            the same stem with a ``.json`` extension.

    Returns:
        The ``selfcheck`` dict, or ``None`` when the metadata is missing,
        unreadable, or carries no usable ``selfcheck`` block.
    """
    metadata_path = mlmodelc_path.with_suffix(".json")
    try:
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(recorded, dict):
        return None
    selfcheck = recorded.get("selfcheck")
    return selfcheck if isinstance(selfcheck, dict) else None


def _calibration_bucket_entry(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build one bucket's entry of ``calibration.buckets``.

    Args:
        report: That bucket's ``selfcheck`` block, or ``None``.

    Returns:
        ``{"status": None, ..., "measured": False}`` when the metadata
        could not be read, the self-check was skipped, or no self-check
        ran at all; otherwise every measured field the report carries
        (``None`` for a field the report itself does not have, e.g. the
        internal-exception report of ``eeane.compiler.selfcheck``, which
        has no ``sanity``/``compute_plan``/``latency`` section) plus
        ``"measured": True``.
    """
    status = report.get("status") if isinstance(report, Mapping) else None
    if status is None or status == SELFCHECK_STATUS_SKIPPED:
        return {
            "status": None,
            "sanity_passed": None,
            "ne_placement_pct": None,
            "latency_median_ms": None,
            "latency_p95_ms": None,
            "measured": False,
        }
    sanity = report.get("sanity") if isinstance(report, Mapping) else None
    compute_plan = report.get("compute_plan") if isinstance(report, Mapping) else None
    latency = report.get("latency") if isinstance(report, Mapping) else None
    return {
        "status": status,
        "sanity_passed": sanity.get("passed") if isinstance(sanity, Mapping) else None,
        "ne_placement_pct": (
            compute_plan.get("ne_placement_pct") if isinstance(compute_plan, Mapping) else None
        ),
        "latency_median_ms": latency.get("median_ms") if isinstance(latency, Mapping) else None,
        "latency_p95_ms": latency.get("p95_ms") if isinstance(latency, Mapping) else None,
        "measured": True,
    }


def _pick_machine(
    reports: Mapping[int, Mapping[str, Any] | None],
    run_reports: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Pick the ``calibration.machine`` block, preferring this run's own.

    Args:
        reports: Every cached bucket's ``selfcheck`` block (or ``None``),
            keyed by bucket.
        run_reports: The subset ``reports`` this invocation itself
            produced; a machine block from here always wins over one read
            back from an earlier run's metadata.

    Returns:
        The first ``machine`` block found, preferring this run's own
        buckets (ascending) and falling back to the rest (ascending), or
        ``None`` when no bucket carries one (e.g. every self-check was
        skipped, always and previously).
    """
    ordered = sorted(reports, key=lambda seq_len: (seq_len not in run_reports, seq_len))
    for seq_len in ordered:
        report = reports[seq_len]
        machine = report.get("machine") if isinstance(report, Mapping) else None
        if isinstance(machine, Mapping):
            return dict(machine)
    return None


def _resolve_embedding_dim(
    kind: str, reports: Mapping[int, Mapping[str, Any] | None]
) -> int | None:
    """Derive the single ``embedding_dim`` recorded across every cached bucket.

    Args:
        kind: Resolved model kind; a reranker never records a width.
        reports: Every cached bucket's ``selfcheck`` block (or ``None``).

    Returns:
        The shared ``embedding_dim``, or ``None`` for a reranker or when
        no cached bucket recorded one (e.g. every self-check was skipped).

    Raises:
        CompileError: If cached buckets disagree on ``embedding_dim``.
    """
    if kind != "embedding":
        return None
    dims: set[int] = set()
    for report in reports.values():
        sanity = report.get("sanity") if isinstance(report, Mapping) else None
        dim = sanity.get("embedding_dim") if isinstance(sanity, Mapping) else None
        if isinstance(dim, int) and not isinstance(dim, bool):
            dims.add(dim)
    if not dims:
        return None
    if len(dims) > 1:
        raise CompileError(
            f"the compiled-model cache records inconsistent embedding_dim values "
            f"{sorted(dims)} across its buckets; it looks corrupt -- recompile with --force"
        )
    return next(iter(dims))


def build_config_snippet(
    *,
    model_id: str,
    kind: str,
    tokenizer_path: Path,
    artifacts: Mapping[int, Path],
    normalize: bool = True,
    cache_root_hint: Path | None = None,
) -> str:
    """Build the minimal ``[[models]]`` TOML snippet for a compiled model.

    The minimal form sets only ``id`` (plus ``normalize`` for an embedding
    entry): ``kind``, ``tokenizer`` and ``artifacts`` are then resolved
    automatically from the compiled-model cache at server start, so the
    snippet keeps working across recompiles that add or drop buckets. The
    explicit equivalent -- what a user would write to pin those fields
    instead, e.g. to point at artifacts moved out of the cache -- is
    included as a comment.

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
        cache_root_hint: The cache root the artifacts were compiled into,
            when it is not the default eeANE resolves on its own; a
            reminder comment naming the ``[server] cache_root`` the
            entry needs is prepended. ``None`` when the default cache
            root was used, which needs no such reminder.

    Returns:
        A TOML fragment ending in a newline.
    """
    lines: list[str] = []
    if cache_root_hint is not None:
        lines += [
            "# This model was compiled into a non-default cache root; the server",
            "# needs to be pointed at the same one to resolve this entry automatically:",
            "# [server]",
            f"# cache_root = {_toml_string(str(Path(cache_root_hint).resolve()))}",
            "",
        ]
    lines += ["[[models]]", f"id = {_toml_string(model_id)}"]
    if kind == "embedding":
        lines.append(f"normalize = {'true' if normalize else 'false'}")
    lines += [
        "# kind / tokenizer / artifacts are resolved from the compiled-model cache.",
        "# To pin them explicitly instead:",
        f"# kind = {_toml_string(kind)}",
        f"# tokenizer = {_toml_string(str(Path(tokenizer_path).resolve()))}",
        "# [models.artifacts]",
    ]
    lines += [
        f"# {bucket} = {_toml_string(str(Path(artifacts[bucket]).resolve()))}"
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
