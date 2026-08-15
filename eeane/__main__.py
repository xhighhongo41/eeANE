"""``python -m eeane`` entry point; delegates to :mod:`eeane.cli`."""

from eeane.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
