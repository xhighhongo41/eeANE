"""eeANE model compiler subpackage.

This subpackage implements ``eeane compile``: converting a HuggingFace
distribution-format model into ANE-ready ``.mlmodelc`` artifacts. It
requires the ``[compile]`` extra (``torch``/``transformers``/
``sentencepiece``), which is *not* part of the runtime dependency set
used by ``eeane serve`` / ``eeane check-config``.

Runtime modules (``eeane.config``/``eeane.schemas``/``eeane.runtime``/
``eeane.engine``/``eeane.server``/``eeane.cli``) must never import this
subpackage (or ``torch``/``transformers``) at module load time; see
:func:`eeane.cli._cmd_compile` for the deferred-import pattern that
keeps ``eeane serve`` usable without the ``[compile]`` extra installed.
"""

from __future__ import annotations

import argparse
import importlib.util

# Command shown to the user to install the missing dependencies. Kept as
# a module-level constant so tests can assert on it without duplicating
# the string used in the error message below.
_INSTALL_HINT = "uv sync --extra compile"


class MissingCompileDependencyError(RuntimeError):
    """Raised when a ``[compile]``-only dependency is not installed."""


def require_compile_dependencies() -> None:
    """Check that the ``[compile]`` extra's packages are importable.

    Only checks for the presence of ``torch`` and ``transformers`` (via
    :func:`importlib.util.find_spec`, so nothing is actually imported
    here) since those are the two packages whose absence is most
    consequential (large, slow to install, and the source of most of
    the runtime/compile dependency split).

    Raises:
        MissingCompileDependencyError: If ``torch`` and/or
            ``transformers`` is not installed. The message lists every
            missing package and how to install them.
    """
    missing = [name for name in ("torch", "transformers") if importlib.util.find_spec(name) is None]
    if not missing:
        return
    joined = ", ".join(missing)
    raise MissingCompileDependencyError(
        f"eeane compile requires the [compile] extra; missing package(s): {joined}. "
        f"Install with: {_INSTALL_HINT}"
    )


def run_compile(args: argparse.Namespace) -> int:
    """Run ``eeane compile``.

    Args:
        args: Parsed ``compile`` subcommand arguments (see
            :func:`eeane.cli.build_parser`).

    Returns:
        The pipeline's exit code: ``0`` on success, ``1`` when a step
        failed (the reason is printed to stderr).
    """
    # Deferred: importing the pipeline (and the self-check hook) pulls in
    # torch/transformers/coremltools, which must not happen before
    # require_compile_dependencies() had a chance to report them missing.
    from eeane.compiler import pipeline, selfcheck

    return pipeline.run(args, selfcheck_fn=selfcheck.run_selfcheck)
