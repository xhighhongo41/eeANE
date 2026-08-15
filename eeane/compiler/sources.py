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

import re
from pathlib import Path

# Files fetched for a Hub repo id. Weights are restricted to safetensors on
# purpose: ``.bin`` checkpoints need ``torch.load``, whose security
# implications are out of scope here.
HF_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer*",
    "*.model",
    "special_tokens_map.json",
    "*config*.json",
)

# ``org/name`` with the character set the Hub allows; a single slash only,
# so local paths (``./x/y``, ``/a/b``) never match.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class SourceError(RuntimeError):
    """Base class for every compile-source resolution failure."""


class SourceNotFoundError(SourceError):
    """Raised when the source is neither a local directory nor a repo id."""


class MissingSafetensorsError(SourceError):
    """Raised when a downloaded snapshot carries no safetensors weights."""


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


def resolve_source(source: str, revision: str | None = None) -> Path:
    """Resolve a compile source to a local model directory.

    An existing path wins over the Hub interpretation, so a local
    ``models/ruri-v3-310m`` is never mistaken for a repo id. Nothing is
    written into the resolved directory; Hub downloads only add files to
    the standard Hugging Face cache.

    Args:
        source: Local directory path or Hub repo id (``org/name``).
        revision: Optional Hub revision (branch, tag, or commit) forwarded
            to ``snapshot_download``; ignored for local directories.

    Returns:
        Absolute path to the model directory.

    Raises:
        SourceError: If an existing path is not a directory, or the
            download failed.
        SourceNotFoundError: If ``source`` is neither an existing path nor
            a repo-id-shaped string.
        MissingSafetensorsError: If a downloaded snapshot has no
            safetensors weights.
    """
    candidate = Path(source).expanduser() if source.strip() else None
    if candidate is not None and candidate.exists():
        if not candidate.is_dir():
            raise SourceError(
                f"'{source}' is not a directory; eeane compile expects a "
                "HuggingFace-format model directory or a Hub id (org/name)"
            )
        return candidate.resolve()
    if looks_like_hf_repo_id(source):
        return _download_snapshot(source, revision)
    raise SourceNotFoundError(
        f"'{source}' is neither an existing model directory nor a Hugging Face "
        "id of the form org/name"
    )


def _download_snapshot(repo_id: str, revision: str | None) -> Path:
    """Fetch a repo's model files into the standard Hugging Face cache.

    Authentication is left entirely to huggingface_hub's own mechanisms
    (``HF_TOKEN``, ``huggingface-cli login``); eeANE never handles tokens.

    Args:
        repo_id: Hub repo id (``org/name``).
        revision: Optional revision passed through to the Hub.

    Returns:
        Absolute path to the downloaded snapshot directory.

    Raises:
        SourceError: If huggingface_hub is unavailable, the download
            failed, or the snapshot path does not exist afterwards.
        MissingSafetensorsError: If the snapshot has no safetensors
            weights.
    """
    # Deferred import: eeane.compiler.sources must stay importable (and
    # unit-testable) without the Hub client being loaded up front.
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - transformers pulls it in
        raise SourceError(
            f"downloading '{repo_id}' needs huggingface_hub, which is not installed"
        ) from exc

    try:
        snapshot = huggingface_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(HF_ALLOW_PATTERNS),
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
    _require_safetensors(snapshot_dir, repo_id)
    return snapshot_dir.resolve()


def _require_safetensors(snapshot_dir: Path, repo_id: str) -> None:
    """Verify that a downloaded snapshot carries safetensors weights.

    Applied to downloads only: :data:`HF_ALLOW_PATTERNS` deliberately
    excludes ``.bin`` checkpoints, so a bin-only repo would otherwise
    arrive without any weights at all and fail much later with a confusing
    transformers error. A local directory is the user's own tree and is
    left to transformers to interpret.

    Args:
        snapshot_dir: Downloaded snapshot directory.
        repo_id: Hub repo id, for the error message.

    Raises:
        MissingSafetensorsError: If neither a safetensors file nor a shard
            index is present.
    """
    if any(snapshot_dir.glob("*.safetensors")) or any(
        snapshot_dir.glob("*.safetensors.index.json")
    ):
        return
    raise MissingSafetensorsError(
        f"no .safetensors weights were downloaded for '{repo_id}'. eeANE only "
        "supports safetensors checkpoints; models distributed as .bin only are not "
        "supported"
    )
