#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${LTX_COMFYUI_ROOT:-/content/ComfyUI}"
COMFY_LTX_ROOT="$COMFY_ROOT/custom_nodes/ComfyUI-LTXVideo"

if [[ ! -d "$COMFY_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"
else
  git -C "$COMFY_ROOT" pull --ff-only
fi

if [[ ! -d "$COMFY_LTX_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git "$COMFY_LTX_ROOT"
else
  git -C "$COMFY_LTX_ROOT" pull --ff-only
fi

uv pip install --python python3 -r "$COMFY_ROOT/requirements.txt"
if [[ -f "$COMFY_LTX_ROOT/requirements.txt" ]]; then
  uv pip install --python python3 -r "$COMFY_LTX_ROOT/requirements.txt"
fi

mkdir -p \
  "$COMFY_ROOT/models/checkpoints" \
  "$COMFY_ROOT/models/loras" \
  "$COMFY_ROOT/models/text_encoders" \
  "$COMFY_ROOT/models/latent_upscale_models"

echo "ComfyUI headless LTX backend is ready at $COMFY_ROOT" >&2
