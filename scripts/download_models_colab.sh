#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_DIR="${LTX_MODEL_DIR:-/content/models}"
VARIANT="${LTX_DOWNLOAD_VARIANT:-ltx2_3_dev_fp8_distilled_lora}"
GEMMA_REPO="${LTX_GEMMA_REPO:-Lightricks/gemma-3-12b-it-qat-q4_0-unquantized}"
TEXT_ENCODER_REPO="${LTX_TEXT_ENCODER_REPO:-Comfy-Org/ltx-2}"
TEXT_ENCODER_FILE="${LTX_TEXT_ENCODER_FILE:-split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors}"
SPATIAL_UPSCALER_FILE="${LTX_SPATIAL_UPSCALER_FILE:-ltx-2.3-spatial-upscaler-x2-1.0.safetensors}"
mkdir -p "$MODEL_DIR"

HF_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ARGS+=(--token "$HF_TOKEN")
fi

hf_download() {
  local repo="$1"
  local include="$2"
  local dest="$3"
  mkdir -p "$dest"
  uv run --project py-worker hf download "$repo" \
    --include "$include" \
    --local-dir "$dest" \
    "${HF_ARGS[@]}"
}

hf_download_exact() {
  local repo="$1"
  local filename="$2"
  local dest="$3"
  mkdir -p "$dest"
  uv run --project py-worker hf download "$repo" "$filename" \
    --local-dir "$dest" \
    "${HF_ARGS[@]}"
  local nested="$dest/$filename"
  local flat="$dest/$(basename "$filename")"
  if [[ -f "$nested" && "$nested" != "$flat" ]]; then
    mv "$nested" "$flat"
  fi
}

download_comfy_text_encoder() {
  local dest="$MODEL_DIR"
  local flat="$dest/$(basename "$TEXT_ENCODER_FILE")"
  if [[ -f "$flat" ]]; then
    echo "Comfy text encoder already exists: $flat" >&2
    return
  fi
  hf_download_exact "$TEXT_ENCODER_REPO" "$TEXT_ENCODER_FILE" "$dest"
}

download_gemma() {
  local dest="$MODEL_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
  if [[ -f "$dest/config.json" ]]; then
    echo "Gemma text encoder already exists: $dest" >&2
    return
  fi
  uv run --project py-worker hf download "$GEMMA_REPO" \
    --local-dir "$dest" \
    "${HF_ARGS[@]}"
}

case "$VARIANT" in
  ltx2_3_distilled)
    hf_download Lightricks/LTX-2.3 "ltx-2.3-22b-distilled.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "$SPATIAL_UPSCALER_FILE" "$MODEL_DIR"
    download_gemma
    ;;
  ltx2_3_full)
    hf_download Lightricks/LTX-2.3 "ltx-2.3-22b-dev.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "$SPATIAL_UPSCALER_FILE" "$MODEL_DIR"
    download_gemma
    ;;
  ltx2_3_fp8)
    hf_download Lightricks/LTX-2.3-fp8 "ltx-2.3-22b-dev-fp8.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "$SPATIAL_UPSCALER_FILE" "$MODEL_DIR"
    download_gemma
    ;;
  ltx2_3_dev_fp8_distilled_lora)
    hf_download Lightricks/LTX-2.3-fp8 "ltx-2.3-22b-dev-fp8.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "ltx-2.3-22b-distilled-lora-384.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "$SPATIAL_UPSCALER_FILE" "$MODEL_DIR"
    download_comfy_text_encoder
    ;;
  ltx2_3_distilled_fp8)
    hf_download Lightricks/LTX-2.3-fp8 "ltx-2.3-22b-distilled-fp8.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "$SPATIAL_UPSCALER_FILE" "$MODEL_DIR"
    download_gemma
    ;;
  sulphur_2_dev_bf16)
    hf_download SulphurAI/Sulphur-2-base "sulphur_dev_bf16.safetensors" "$MODEL_DIR"
    download_gemma
    ;;
  sulphur_2_dev_fp8mixed)
    hf_download SulphurAI/Sulphur-2-base "sulphur_dev_fp8mixed.safetensors" "$MODEL_DIR"
    download_gemma
    ;;
  sulphur_2_distil_bf16)
    hf_download SulphurAI/Sulphur-2-base "sulphur_distil_bf16.safetensors" "$MODEL_DIR"
    hf_download Lightricks/LTX-2.3 "$SPATIAL_UPSCALER_FILE" "$MODEL_DIR"
    download_gemma
    ;;
  *)
    echo "unknown LTX_DOWNLOAD_VARIANT: $VARIANT" >&2
    exit 2
    ;;
esac

python3 scripts/write_downloaded_model_registry.py \
  --variant "$VARIANT" \
  --model-dir "$MODEL_DIR" \
  --output configs/model_registry.toml

echo "Downloaded model variant: $VARIANT" >&2
echo "Updated configs/model_registry.toml" >&2
