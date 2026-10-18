"""Neural Engine placement limits shared by the compiler and the server.

Core ML decides on its own which compute device runs a model, and it
reports neither an error nor a warning when it keeps a whole model off
the Neural Engine. The facts needed to notice that live here, in a module
that imports nothing beyond the standard library so both ``eeane compile``
and ``eeane serve`` can use it.

Two measured facts are encoded:

* A compiled model above a size limit is not placed on the Neural Engine
  at all: every operation falls back to the CPU. The limit is 2 GiB up to
  macOS 26 and 1 GiB from macOS 27 on. It applies to each compiled model
  on its own, so a model stored as several compiled parts is judged part
  by part. ``eeane compile --ne-size-limit`` replaces the limit derived
  from the operating system, e.g. to compile for another machine.
* A model whose share of Neural Engine operations is below the warning
  threshold is, in practice, not being accelerated.
"""

from __future__ import annotations

import platform
import re
from fractions import Fraction
from pathlib import Path

# Largest compiled model the Neural Engine accepts from macOS 27 on, and
# up to macOS 26.
NE_MODEL_SIZE_LIMIT_BYTES_MACOS27 = 1 << 30
NE_MODEL_SIZE_LIMIT_BYTES_LEGACY = 2 << 30

# First macOS major version that enforces NE_MODEL_SIZE_LIMIT_BYTES_MACOS27
# instead of NE_MODEL_SIZE_LIMIT_BYTES_LEGACY.
NE_SIZE_LIMIT_FIRST_MACOS_MAJOR = 27

# Value of a size-limit option that turns the limit (and with it the
# splitting of an oversized model) off.
SIZE_LIMIT_DISABLED = "none"

# Share of dispatched operations (percent) below which a compiled model is
# reported as not running on the Neural Engine.
NE_PLACEMENT_WARN_THRESHOLD = 90.0

_BYTES_PER_UNIT = {"kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30}

# A plain number optionally followed by a binary unit. ASCII only, so a
# digit of another script is not mistaken for a number.
_SIZE_PATTERN = re.compile(r"(?P<number>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>kib|mib|gib)?", re.ASCII)


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


def parse_size_limit(text: str) -> int | None:
    """Parse the value of a Neural Engine size-limit option.

    Args:
        text: ``"none"`` (case-insensitive), a positive whole number of
            bytes such as ``"1073741824"``, or a number -- decimals
            allowed -- followed by ``KiB``, ``MiB`` or ``GiB``, such as
            ``"800MiB"`` or ``"1.5 GiB"``. Case and surrounding
            whitespace are ignored. Decimal units (``GB``) are refused on
            purpose: the limit is a binary size, and reading ``1GB`` as
            10**9 bytes would silently be a different limit than the one
            meant.

    Returns:
        The limit in bytes (a fractional result is truncated), or
        ``None`` for ``"none"``, meaning no limit.

    Raises:
        ValueError: If ``text`` has none of those forms or names a size
            below one byte.
    """
    stripped = text.strip()
    normalized = stripped.lower()
    if normalized == SIZE_LIMIT_DISABLED:
        return None

    match = _SIZE_PATTERN.fullmatch(normalized)
    limit = 0
    if match is not None:
        number = match["number"]
        unit = match["unit"]
        if unit is not None:
            # Exact arithmetic: "1.5GiB" must come out as 1610612736, not
            # as whatever a binary float happens to round to.
            limit = int(Fraction(number) * _BYTES_PER_UNIT[unit])
        elif "." not in number:
            limit = int(number)
    if limit < 1:
        raise ValueError(
            f"invalid Neural Engine size limit {stripped!r}: give a size such as 1GiB, "
            f"800MiB or 1073741824 (bytes), or '{SIZE_LIMIT_DISABLED}'"
        )
    return limit


def ne_size_limit_bytes(mac_version: str | None = None, override: str | None = None) -> int | None:
    """Return the largest compiled model the Neural Engine is assumed to accept.

    Args:
        mac_version: macOS version to judge; ``None`` means the running
            system. Not consulted when ``override`` is given.
        override: A user-given limit in the form
            :func:`parse_size_limit` reads, taking precedence over the
            operating system.

    Returns:
        The limit in bytes, or ``None`` when there is none: the override
        says ``"none"``, or the version cannot be determined (as on a
        platform other than macOS).

    Raises:
        ValueError: If ``override`` is not a valid limit.
    """
    if override is not None:
        return parse_size_limit(override)
    major = macos_major_version(mac_version)
    if major is None:
        return None
    if major >= NE_SIZE_LIMIT_FIRST_MACOS_MAJOR:
        return NE_MODEL_SIZE_LIMIT_BYTES_MACOS27
    return NE_MODEL_SIZE_LIMIT_BYTES_LEGACY


def describe_size_limit(limit_bytes: int) -> str:
    """Format a size limit for a message.

    Args:
        limit_bytes: The limit in bytes.

    Returns:
        ``"1 GiB"`` or ``"800 MiB"`` when the limit is an exact multiple
        of that unit, otherwise ``"<n> bytes"``.
    """
    for unit_bytes, name in ((1 << 30, "GiB"), (1 << 20, "MiB")):
        if limit_bytes % unit_bytes == 0:
            return f"{limit_bytes // unit_bytes} {name}"
    return f"{limit_bytes} bytes"


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
