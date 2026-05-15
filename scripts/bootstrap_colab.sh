#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --download-models) DOWNLOAD_MODELS=1 ;;
  esac
done

cd "$(dirname "$0")/.."
export COLAB_MODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ffmpeg
fi

mkdir -p /content/outputs /content/ltx_tmp /content/models
cp -n configs/colab_engine.example.toml configs/colab_engine.toml || true
cp -n configs/model_registry.example.toml configs/model_registry.toml || true

uv sync --project py-worker
cargo build --release -p ltx-runner

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  bash scripts/download_models_colab.sh
fi

./target/release/ltx-runner check
