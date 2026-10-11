"""Start-up checks for compiled models that will not run on the Neural Engine.

Core ML reports neither an error nor a warning when it keeps a model off
the Neural Engine; the model simply runs on the CPU, slower and, for some
models, with invalid results. The checks here read facts that are already
on disk (the placement recorded when a model was compiled, and its size)
so the server can say so at start-up.

Only the standard library and :mod:`eeane.ne_limits` are imported: the
serving path must not depend on the conversion stack.
"""

from __future__ import annotations

import json
from pathlib import Path

from eeane import ne_limits

_BYTES_PER_GIB = 1 << 30


def _recorded_placement_pct(artifact_path: Path) -> float | None:
    """Read the Neural Engine placement recorded next to a compiled model.

    Args:
        artifact_path: ``.mlmodelc`` directory. Its metadata is the JSON
            file with the same name and a ``.json`` suffix.

    Returns:
        The recorded percentage, or ``None`` when the metadata is absent,
        unreadable, or holds no measured numeric value.
    """
    try:
        record = json.loads(artifact_path.with_suffix(".json").read_text(encoding="utf-8"))
        plan = record["selfcheck"]["compute_plan"]
        if plan["status"] != "measured":
            return None
        pct = plan["ne_placement_pct"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if isinstance(pct, bool) or not isinstance(pct, int | float):
        return None
    return float(pct)


def artifact_placement_reasons(artifact_path: Path, *, mac_version: str | None = None) -> list[str]:
    """Explain why a compiled model will not run on the Neural Engine.

    Args:
        artifact_path: ``.mlmodelc`` directory of one compiled model.
        mac_version: macOS version to judge the size limit by; ``None``
            means the running system.

    Returns:
        Short English phrases, empty when nothing is wrong. The recorded
        placement comes first, the size limit second; both can apply.
    """
    reasons: list[str] = []

    pct = _recorded_placement_pct(artifact_path)
    if pct is not None and pct < ne_limits.NE_PLACEMENT_WARN_THRESHOLD:
        reasons.append(
            f"only {pct:.1f}% of its operations were placed on the Neural Engine "
            "when it was compiled"
        )

    if ne_limits.ne_size_limit_applies(mac_version):
        try:
            size = ne_limits.compiled_model_size_bytes(artifact_path)
        except OSError:
            size = 0
        if size > ne_limits.NE_MODEL_SIZE_LIMIT_BYTES:
            reasons.append(
                f"it is {size / _BYTES_PER_GIB:.2f} GiB, above the 1 GiB "
                "the Neural Engine accepts on macOS 27 and later"
            )
    return reasons
