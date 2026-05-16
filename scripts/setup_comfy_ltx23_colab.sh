#!/usr/bin/env bash
set -euo pipefail

COMFY_ROOT="${LTX_COMFYUI_ROOT:-/content/ComfyUI}"
DOWNLOAD_MODELS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --comfy-root)
      COMFY_ROOT="$2"
      shift 2
      ;;
    --skip-models)
      DOWNLOAD_MODELS=0
      shift
      ;;
    --help)
      cat <<'EOF'
Usage: scripts/setup_comfy_ltx23_colab.sh [--comfy-root /content/ComfyUI] [--skip-models]

Sets up ComfyUI plus the LTX-2.3 lightweight template dependencies for Colab.
By default it downloads the model artifacts required by fixtures/ltx23_comfy_template.json.
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

COMFY_LTX_ROOT="$COMFY_ROOT/custom_nodes/ComfyUI-LTXVideo"

clone_or_update() {
  local repo="$1"
  local dest="$2"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" pull --ff-only
  else
    mkdir -p "$(dirname "$dest")"
    git clone --depth 1 "$repo" "$dest"
  fi
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ffmpeg
fi

clone_or_update https://github.com/comfyanonymous/ComfyUI.git "$COMFY_ROOT"
clone_or_update https://github.com/Lightricks/ComfyUI-LTXVideo.git "$COMFY_LTX_ROOT"

python3 -m pip install --upgrade pip
python3 -m pip install -r "$COMFY_ROOT/requirements.txt"
if [[ -f "$COMFY_LTX_ROOT/requirements.txt" ]]; then
  python3 -m pip install -r "$COMFY_LTX_ROOT/requirements.txt"
fi
python3 -m pip install huggingface_hub safetensors accelerate

mkdir -p \
  "$COMFY_ROOT/models/checkpoints" \
  "$COMFY_ROOT/models/loras" \
  "$COMFY_ROOT/models/text_encoders" \
  "$COMFY_ROOT/models/latent_upscale_models"

download_hf_file() {
  local repo="$1"
  local filename="$2"
  local dest_dir="$3"
  local target="$dest_dir/$(basename "$filename")"
  if [[ -s "$target" ]]; then
    echo "model already exists: $target" >&2
    return
  fi
  python3 - "$repo" "$filename" "$dest_dir" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id, filename, dest_dir = sys.argv[1:4]
downloaded = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    token=os.environ.get("HF_TOKEN"),
)
target = Path(dest_dir) / Path(filename).name
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(downloaded, target)
print(target)
PY
}

if [[ "$DOWNLOAD_MODELS" == "1" ]]; then
  download_hf_file Lightricks/LTX-2.3-fp8 \
    ltx-2.3-22b-dev-fp8.safetensors \
    "$COMFY_ROOT/models/checkpoints"
  download_hf_file Lightricks/LTX-2.3 \
    ltx-2.3-22b-distilled-lora-384.safetensors \
    "$COMFY_ROOT/models/loras"
  download_hf_file Lightricks/LTX-2.3 \
    ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
    "$COMFY_ROOT/models/latent_upscale_models"
  download_hf_file Comfy-Org/ltx-2 \
    split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors \
    "$COMFY_ROOT/models/text_encoders"
fi

echo "ComfyUI LTX-2.3 Colab setup is ready at $COMFY_ROOT" >&2
