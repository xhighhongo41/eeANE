"""Neural Engine placement limits shared by the compiler and the server.

Core ML decides on its own which compute device runs a model, and it
reports neither an error nor a warning when it keeps a whole model off
the Neural Engine. The facts needed to notice that live here, in a module
that imports nothing beyond the standard library so both ``eeane compile``
and ``eeane serve`` can use it.

Two measured facts are encoded:

* From macOS 27 on, a compiled model larger than 1 GiB is not placed on
  the Neural Engine at all: every operation falls back to the CPU. Up to
  macOS 26 the same cliff sat at 2 GiB.
* A model whose share of Neural Engine operations is below the warning
  threshold is, in practice, not being accelerated.
"""

from __future__ import annotations

import platform
from pathlib import Path

# Largest compiled model the Neural Engine accepts from macOS 27 on.
NE_MODEL_SIZE_LIMIT_BYTES = 1 << 30

# First macOS major version that enforces NE_MODEL_SIZE_LIMIT_BYTES.
NE_SIZE_LIMIT_FIRST_MACOS_MAJOR = 27

# Share of dispatched operations (percent) below which a compiled model is
# reported as not running on the Neural Engine.
NE_PLACEMENT_WARN_THRESHOLD = 90.0


def macos_major_version(mac_version: str | None = None) -> int | None:
    """Return the major version of a macOS version string.

    Args:
        mac_version: Version such as ``"27.0.1"``. ``None`` reads the
            running system's version.

    Returns:
        The major version, or ``None`` when the string carries none (as on
        a platform other than macOS).
    """
    if mac_version is None:
        mac_version = platform.mac_ver()[0]
    head = mac_version.strip().split(".", 1)[0]
    return int(head) if head.isdigit() else None


def ne_size_limit_applies(mac_version: str | None = None) -> bool:
    """Tell whether the 1 GiB Neural Engine size limit is in force.

    Args:
        mac_version: macOS version to judge; ``None`` means the running
            system.

    Returns:
        ``True`` from macOS 27 on, ``False`` for earlier versions and for
        a version that cannot be determined.
    """
    major = macos_major_version(mac_version)
    return major is not None and major >= NE_SIZE_LIMIT_FIRST_MACOS_MAJOR


def compiled_model_size_bytes(mlmodelc_path: Path) -> int:
    """Return the total size of a compiled model directory.

    Args:
        mlmodelc_path: ``.mlmodelc`` directory.

    Returns:
        Sum of the sizes of every regular file below it, in bytes; ``0``
        when the directory does not exist.
    """
    if not mlmodelc_path.is_dir():
        return 0
    return sum(p.stat().st_size for p in mlmodelc_path.rglob("*") if p.is_file())
