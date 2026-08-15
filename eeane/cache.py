"""Reader for the compiled-model cache that ``eeane compile`` writes.

Layout of the cache (one directory per compiled model)::

    <cache root>/compiled/<model name>/
        tokenizer.json
        model_info.json
        s512_b1_eager_macos13.mlmodelc/

This module implements the *runtime* half of that convention -- where the
cache root is, which directory a model id maps to, and how to read a
model's ``model_info.json`` -- so that a served model can be configured
by id alone. It stays import-light on purpose: the serving process is
installed without the ``[compile]`` extra, so nothing here may reach for
the compiler package or its heavyweight dependencies.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Cache layout constants, shared with the writer side by convention. The
# tokenizer and artifact file names are not fixed here: they are read from
# each model's own record, so that a differently named artifact is an
# error rather than a silently guessed path.
CACHE_SUBDIR = "compiled"
MODEL_INFO_FILENAME = "model_info.json"

# Directory the cache root gets by default, below the XDG cache home.
CACHE_DIRNAME = "eeane"
XDG_CACHE_HOME_ENV = "XDG_CACHE_HOME"


class CacheError(Exception):
    """Raised when the compiled-model cache is missing, unusable, or malformed."""


def resolve_cache_root(
    override: Path | None = None, *, env: Mapping[str, str] | None = None
) -> Path:
    """Resolve the root directory of the compiled-model cache.

    Args:
        override: Explicitly configured cache root. ``~`` is expanded; the
            path is otherwise returned as given, so callers stay in
            control of how a relative path is anchored.
        env: Environment mapping to read ``XDG_CACHE_HOME`` from; defaults
            to ``os.environ`` (a parameter for testability).

    Returns:
        The cache root (which need not exist yet). Without an override
        this is ``$XDG_CACHE_HOME/eeane``, falling back to
        ``~/.cache/eeane`` when ``XDG_CACHE_HOME`` is unset or relative,
        as the XDG base directory specification requires.
    """
    if override is not None:
        return Path(override).expanduser()

    environment = env if env is not None else os.environ
    xdg_cache_home = environment.get(XDG_CACHE_HOME_ENV, "").strip()
    base = Path(xdg_cache_home) if xdg_cache_home else None
    if base is None or not base.is_absolute():
        base = Path.home() / ".cache"
    return base / CACHE_DIRNAME


def model_cache_dir(cache_root: Path, model_id: str) -> Path:
    """Return the cache directory holding one model's compiled artifacts.

    Args:
        cache_root: Cache root as returned by :func:`resolve_cache_root`.
        model_id: Model id as configured. A Hub id (``org/name``) is
            normalised to a single path component (``org--name``), the
            same way the writer side names the directory.

    Returns:
        ``<cache_root>/compiled/<normalised name>`` (which need not exist).

    Raises:
        CacheError: If ``model_id`` does not name a single directory
            component (empty, ``.``/``..``, or still path-like after
            normalisation).
    """
    name = model_id.strip().replace("/", "--")
    # Path().name differs from the input for anything still holding a
    # separator; the dot names have to be spelled out, as they are
    # ordinary path components that would climb out of the cache root.
    if not name or name in {".", ".."} or name != Path(name).name:
        raise CacheError(f"model id '{model_id}' does not name a single cache directory component")
    return cache_root / CACHE_SUBDIR / name


def model_info_path(model_dir: Path) -> Path:
    """Return the ``model_info.json`` path inside a model's cache directory.

    Args:
        model_dir: Directory returned by :func:`model_cache_dir`.

    Returns:
        The record's path (which need not exist).
    """
    return model_dir / MODEL_INFO_FILENAME


def load_model_info(model_dir: Path) -> dict[str, Any]:
    """Read and parse a model's ``model_info.json``.

    The record is returned as-is: interpreting its fields (and tolerating
    older ``format_version`` values) is the caller's job.

    Args:
        model_dir: Directory returned by :func:`model_cache_dir`.

    Returns:
        The parsed JSON object.

    Raises:
        CacheError: If the record is missing, unreadable, not valid UTF-8
            or JSON, or not a JSON object.
    """
    path = model_info_path(model_dir)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise CacheError(f"no compiled-model record at '{path}'") from exc
    except UnicodeDecodeError as exc:
        raise CacheError(f"'{path}' is not valid UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise CacheError(f"cannot read '{path}': {exc}") from exc

    try:
        info = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CacheError(f"'{path}' is not valid JSON: {exc}") from exc
    if not isinstance(info, dict):
        raise CacheError(f"'{path}' must hold a JSON object, found {type(info).__name__}")
    return info
