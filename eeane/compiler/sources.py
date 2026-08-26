"""Compile input resolution.

``eeane compile <source>`` accepts either a local HuggingFace-format model
directory or a Hub repo id (``org/name``). This module turns both into a
local directory path, downloading the latter into the *standard* Hugging
Face cache (shared with every other HF tool) via
``huggingface_hub.snapshot_download``.

Input model directories are strictly read-only: nothing here writes to,
modifies, or deletes anything inside a resolved model directory.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("eeane.compiler.sources")

# Files fetched for a Hub repo id. Weights are restricted to safetensors by
# default: a ``.bin`` checkpoint needs ``torch.load`` (a pickle format), so
# it is only fetched when the caller explicitly opts in via
# ``allow_pickle`` -- see :data:`HF_BIN_PATTERNS` and :func:`resolve_source`.
HF_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer*",
    "*.model",
    "special_tokens_map.json",
    "*config*.json",
)

# Extra patterns added to a Hub download when ``allow_pickle=True`` and no
# safetensors weights were found with :data:`HF_ALLOW_PATTERNS` alone.
HF_BIN_PATTERNS: tuple[str, ...] = (
    "*.bin",
    "*.bin.index.json",
)

# ``org/name`` with the character set the Hub allows; a single slash only,
# so local paths (``./x/y``, ``/a/b``) never match.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class SourceError(RuntimeError):
    """Base class for every compile-source resolution failure."""


class SourceNotFoundError(SourceError):
    """Raised when the source is neither a local directory nor a repo id."""


class MissingSafetensorsError(SourceError):
    """Raised when a source carries no safetensors weights and no opt-in.

    Applies to both a downloaded Hub snapshot and a local model directory:
    either one may fall back to pickle-based ``.bin`` weights, but only
    when ``allow_pickle=True`` was passed to :func:`resolve_source`.
    """


def looks_like_hf_repo_id(source: str) -> bool:
    """Report whether ``source`` has the shape of a Hub repo id.

    This is a purely syntactic check (``org/name``); a repo-id-shaped
    string may still be a relative directory path, which is why
    :func:`resolve_source` looks at the filesystem first.

    Args:
        source: The raw ``eeane compile <source>`` argument.

    Returns:
        True if ``source`` matches the ``org/name`` shape.
    """
    return bool(_HF_REPO_ID_RE.match(source))


def resolve_source(source: str, revision: str | None = None, *, allow_pickle: bool = False) -> Path:
    """Resolve a compile source to a local model directory.

    An existing path wins over the Hub interpretation, so a local
    ``models/ruri-v3-310m`` is never mistaken for a repo id. Nothing is
    written into the resolved directory; Hub downloads only add files to
    the standard Hugging Face cache.

    Both a local directory and a Hub download go through the same
    safetensors-first policy: safetensors weights are always accepted;
    pickle-based ``.bin`` weights are accepted only when
    ``allow_pickle=True`` (and a WARNING is logged when they are used). A
    source with neither is passed through unchanged and left to
    ``transformers`` to report.

    Args:
        source: Local directory path or Hub repo id (``org/name``).
        revision: Optional Hub revision (branch, tag, or commit) forwarded
            to ``snapshot_download``; ignored for local directories.
        allow_pickle: Opt into pickle-based ``.bin`` weights when no
            safetensors weights are available. Only intended for models
            from publishers the caller trusts.

    Returns:
        Absolute path to the model directory.

    Raises:
        SourceError: If an existing path is not a directory, or the
            download failed.
        SourceNotFoundError: If ``source`` is neither an existing path nor
            a repo-id-shaped string.
        MissingSafetensorsError: If a source has no safetensors weights,
            has pickle-based ``.bin`` weights instead, and
            ``allow_pickle`` is ``False``.
    """
    candidate = Path(source).expanduser() if source.strip() else None
    if candidate is not None and candidate.exists():
        if not candidate.is_dir():
            raise SourceError(
                f"'{source}' is not a directory; eeane compile expects a "
                "HuggingFace-format model directory or a Hub id (org/name)"
            )
        resolved = candidate.resolve()
        _check_pickle_gate(resolved, str(resolved), allow_pickle=allow_pickle)
        return resolved
    if looks_like_hf_repo_id(source):
        return _download_snapshot(source, revision, allow_pickle=allow_pickle)
    raise SourceNotFoundError(
        f"'{source}' is neither an existing model directory nor a Hugging Face "
        "id of the form org/name"
    )


def _download_snapshot(repo_id: str, revision: str | None, *, allow_pickle: bool) -> Path:
    """Fetch a repo's model files into the standard Hugging Face cache.

    Authentication is left entirely to huggingface_hub's own mechanisms
    (``HF_TOKEN``, ``huggingface-cli login``); eeANE never handles tokens.

    Only :data:`HF_ALLOW_PATTERNS` (safetensors) is requested at first. If
    the resulting snapshot has no safetensors weights and
    ``allow_pickle=True``, a second download is made with
    :data:`HF_BIN_PATTERNS` added, so a repo that already has safetensors
    never pays for a ``.bin`` download it does not need.

    Args:
        repo_id: Hub repo id (``org/name``).
        revision: Optional revision passed through to the Hub.
        allow_pickle: Opt into a ``.bin`` fallback download when no
            safetensors weights were found.

    Returns:
        Absolute path to the downloaded snapshot directory.

    Raises:
        SourceError: If huggingface_hub is unavailable, a download
            failed, or a snapshot path does not exist afterwards.
        MissingSafetensorsError: If the snapshot has no safetensors
            weights and ``allow_pickle`` is ``False``.
    """
    # Deferred import: eeane.compiler.sources must stay importable (and
    # unit-testable) without the Hub client being loaded up front.
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - transformers pulls it in
        raise SourceError(
            f"downloading '{repo_id}' needs huggingface_hub, which is not installed"
        ) from exc

    snapshot_dir = _fetch_snapshot(huggingface_hub, repo_id, revision, HF_ALLOW_PATTERNS)
    if _has_safetensors(snapshot_dir):
        return snapshot_dir.resolve()
    if not allow_pickle:
        raise _missing_safetensors_error(repo_id)

    # The safetensors-only download above never requested .bin files, so a
    # second, larger download is needed to actually fetch them.
    _warn_pickle_checkpoint(repo_id)
    snapshot_dir = _fetch_snapshot(
        huggingface_hub, repo_id, revision, (*HF_ALLOW_PATTERNS, *HF_BIN_PATTERNS)
    )
    return snapshot_dir.resolve()


def _fetch_snapshot(
    huggingface_hub_module: Any,
    repo_id: str,
    revision: str | None,
    allow_patterns: Sequence[str],
) -> Path:
    """Download one snapshot via ``snapshot_download`` and validate the result.

    Shared by :func:`_download_snapshot`'s first (safetensors-only) and
    optional second (safetensors + ``.bin``) download.

    Args:
        huggingface_hub_module: The deferred-imported ``huggingface_hub``
            module (passed in so the import stays deferred and mockable).
        repo_id: Hub repo id (``org/name``).
        revision: Optional revision passed through to the Hub.
        allow_patterns: File patterns requested for this download.

    Returns:
        The downloaded snapshot directory (not yet resolved to an
        absolute path).

    Raises:
        SourceError: If the download failed, or the returned path is not
            a directory.
    """
    try:
        snapshot = huggingface_hub_module.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(allow_patterns),
        )
    except Exception as exc:
        raise SourceError(
            f"failed to download '{repo_id}' from the Hugging Face Hub: {exc}"
        ) from exc

    snapshot_dir = Path(snapshot)
    if not snapshot_dir.is_dir():
        raise SourceError(
            f"the snapshot downloaded for '{repo_id}' is not a directory: {snapshot_dir}"
        )
    return snapshot_dir


def _check_pickle_gate(directory: Path, location: str, *, allow_pickle: bool) -> None:
    """Enforce the safetensors-first / opt-in-pickle policy on one directory.

    Applied identically to a downloaded snapshot and a local model
    directory. A directory with no weight files at all (neither
    safetensors nor ``.bin``) is left untouched: ``transformers`` reports
    that case with its own, already clear, error.

    Args:
        directory: Directory to inspect (a downloaded snapshot or a local
            model directory).
        location: Human-readable identifier of ``directory`` for log/error
            messages (a repo id or a filesystem path).
        allow_pickle: Whether pickle-based ``.bin`` weights are accepted
            as a fallback.

    Raises:
        MissingSafetensorsError: If ``directory`` has ``.bin`` weights but
            no safetensors weights, and ``allow_pickle`` is ``False``.
    """
    if _has_safetensors(directory):
        return
    if not _has_bin_weights(directory):
        # No weights of either kind: nothing for this policy to enforce.
        # transformers itself reports the absence of any checkpoint.
        return
    if not allow_pickle:
        raise _missing_safetensors_error(location)
    _warn_pickle_checkpoint(location)


def _has_safetensors(directory: Path) -> bool:
    """Report whether ``directory`` carries safetensors weights.

    Args:
        directory: Directory to inspect.

    Returns:
        True if a ``*.safetensors`` file or a ``*.safetensors.index.json``
        shard index is present.
    """
    return any(directory.glob("*.safetensors")) or any(directory.glob("*.safetensors.index.json"))


def _has_bin_weights(directory: Path) -> bool:
    """Report whether ``directory`` carries pickle-based ``.bin`` weights.

    Args:
        directory: Directory to inspect.

    Returns:
        True if a ``*.bin`` file or a ``*.bin.index.json`` shard index is
        present.
    """
    return any(directory.glob("*.bin")) or any(directory.glob("*.bin.index.json"))


def _missing_safetensors_error(location: str) -> MissingSafetensorsError:
    """Build the error raised when safetensors weights are required but absent.

    Args:
        location: Human-readable identifier of the source that lacks
            safetensors weights (a repo id or a filesystem path).

    Returns:
        A :class:`MissingSafetensorsError` whose message points at
        ``--allow-pickle`` as the opt-in for pickle-based weights.
    """
    return MissingSafetensorsError(
        f"no .safetensors weights are available for '{location}'. eeANE requires "
        "safetensors checkpoints by default; pass --allow-pickle to opt into "
        "pickle-based weights (pytorch_model.bin) instead -- only do this for "
        "models from publishers you trust"
    )


def _warn_pickle_checkpoint(location: str) -> None:
    """Log the WARNING emitted when pickle-based ``.bin`` weights are used.

    Args:
        location: Human-readable identifier of the source being loaded
            from ``.bin`` weights (a repo id or a filesystem path).
    """
    logger.warning(
        "loading a pickle-based checkpoint (.bin) from '%s'. transformers enforces "
        "torch.load(weights_only=True), which mitigates but does not eliminate the "
        "risks of the pickle format. Only use checkpoints from publishers you trust.",
        location,
    )
