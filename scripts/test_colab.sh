#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cargo test --workspace
uv run --project py-worker python -m pytest py-worker/tests
