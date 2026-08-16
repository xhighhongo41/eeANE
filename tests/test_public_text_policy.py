"""Guard test: public files must not reference eeANE's private internal documents.

This repository keeps a directory and a top-level file, private to the
development team, for planning and progress-tracking material; neither is
part of the OSS distribution and both are meaningless to a reader who
only has the public repository. This test scans every publicly
distributed text file for references to that material -- by name, or by
an internal task/record label naming it -- and fails, listing every
offending line, if any are found. See :data:`_FORBIDDEN_PATTERNS` for the
exact names being guarded against.

The forbidden pattern strings are built by concatenation below so that
this test file's own source never contains them verbatim; otherwise the
scan (which also covers ``tests/**/*.py``) would flag itself.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Glob patterns (relative to the repo root) for every publicly distributed
# text file this policy applies to. ``poc/`` is deliberately excluded: it
# is a frozen historical record of the original PoC scripts, not material
# eeANE distributes as its current implementation or documentation.
_TARGET_GLOBS: tuple[str, ...] = (
    "eeane/**/*.py",
    "tools/**/*.py",
    "tools/**/*.sh",
    "tests/**/*.py",
    "docs/**/*",
    ".github/**/*.yml",
)

# Individual files (relative to the repo root) that fall outside the glob
# patterns above but are still part of the public distribution.
_TARGET_FILES: tuple[str, ...] = (
    "README.md",
    "README_ja.md",
    "eeane.example.toml",
    "pyproject.toml",
)

# Names of eeANE's non-public internal documents/labels. Each is built by
# concatenating two halves so the literal substring never appears in this
# file's own source (see the module docstring).
_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "開発" + "資料",
    "実装" + "計画",
    "実装" + "記録",
    "ロード" + "マップ",
    "PROJECT" + ".md",
    "申し" + "送り",
)

# Allowlist of (path relative to the repo root, forbidden pattern) pairs
# that are permitted despite matching a forbidden pattern above. Empty for
# now: add an entry here only together with a comment explaining why that
# specific match is not actually a reference to non-public material.
_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


def _target_files() -> list[Path]:
    """Collect every public text file this policy applies to.

    Returns:
        Sorted list of absolute paths to check, deduplicated across the
        overlapping glob patterns.
    """
    files: set[Path] = set()
    for pattern in _TARGET_GLOBS:
        files.update(path for path in _REPO_ROOT.glob(pattern) if path.is_file())
    for name in _TARGET_FILES:
        path = _REPO_ROOT / name
        if path.is_file():
            files.add(path)
    return sorted(files)


def _find_violations() -> list[str]:
    """Scan every target file for forbidden internal-document references.

    Args:
        None.

    Returns:
        One ``"<relative path>:<line number>: <pattern>"`` string per
        match, excluding matches covered by :data:`_ALLOWLIST`. Files that
        cannot be decoded as UTF-8 text are skipped (none are expected
        among the target files).
    """
    violations: list[str] = []
    for path in _target_files():
        rel_path = path.relative_to(_REPO_ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in _FORBIDDEN_PATTERNS:
                if pattern in line and (rel_path, pattern) not in _ALLOWLIST:
                    violations.append(f"{rel_path}:{lineno}: {pattern!r}")
    return violations


def test_no_internal_document_references_in_public_files() -> None:
    """Public files must contain none of eeANE's internal-document names/labels."""
    violations = _find_violations()
    assert not violations, "internal-document references found in public files:\n" + "\n".join(
        violations
    )
