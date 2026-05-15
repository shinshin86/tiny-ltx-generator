#!/usr/bin/env bash
set -euo pipefail
nvidia-smi || true
python3 --version
uv --version || true
cargo --version || true
ffmpeg -version | head -n 1 || true
