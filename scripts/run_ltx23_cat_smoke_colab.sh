#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMFY_ROOT="${LTX_COMFYUI_ROOT:-/content/ComfyUI}"
OUT_DIR="${LTX_OUTPUT_DIR:-/content/outputs}"
JOB_ID="${LTX_JOB_ID:-cat-smoke-ltx23}"

cd "$REPO_ROOT"

if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
fi
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi

scripts/setup_comfy_ltx23_colab.sh --comfy-root "$COMFY_ROOT"

cargo test --workspace
cargo build --release -p ltx-runner

./target/release/ltx-runner prepare-run \
  --workflow fixtures/ltx23_comfy_template.json \
  --manifest fixtures/ltx23_template_manifest.example.json \
  --out-dir "$OUT_DIR" \
  --job-id "$JOB_ID" \
  --comfy-root "$COMFY_ROOT" \
  --prompt "${LTX_PROMPT:-a small orange cat walking through a sunlit room, cinematic, natural motion, detailed fur}" \
  --negative-prompt "${LTX_NEGATIVE_PROMPT:-low quality, noisy, distorted, flickering, abstract}" \
  --width "${LTX_WIDTH:-768}" \
  --height "${LTX_HEIGHT:-512}" \
  --duration-seconds "${LTX_DURATION_SECONDS:-4}" \
  --fps "${LTX_FPS:-12}" \
  --seed "${LTX_SEED:-12345}" \
  --checkpoint "${LTX_CHECKPOINT:-ltx-2.3-22b-dev-fp8.safetensors}" \
  --text-encoder "${LTX_TEXT_ENCODER:-gemma_3_12B_it_fp4_mixed.safetensors}" \
  --distilled-lora "${LTX_DISTILLED_LORA:-ltx-2.3-22b-distilled-lora-384.safetensors}" \
  --lora-strength "${LTX_LORA_STRENGTH:-0.5}" \
  --spatial-upscaler "${LTX_SPATIAL_UPSCALER:-ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"

python3 scripts/comfy_headless_adapter.py \
  --request "$OUT_DIR/jobs/$JOB_ID/adapter_request.json"

test -s "$OUT_DIR/jobs/$JOB_ID/output.mp4"
echo "$OUT_DIR/jobs/$JOB_ID/output.mp4"
