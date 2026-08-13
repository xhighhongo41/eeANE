#!/bin/bash
# Lint and test gate. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
