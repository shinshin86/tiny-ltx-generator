#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -x ./target/release/ltx-runner ]]; then
  cargo build --release -p ltx-runner
fi

bash scripts/test_colab.sh
./target/release/ltx-runner check --json >/content/ltx_check.json

VALIDATION_ROOT="${LTX_VALIDATION_ROOT:-/content/ltx_validation}"
MODEL_DIR="$VALIDATION_ROOT/models"
OUT_DIR="$VALIDATION_ROOT/outputs"
REGISTRY="$VALIDATION_ROOT/model_registry.toml"
MARKER="$VALIDATION_ROOT/test_cuda_oom_once.marker"

rm -rf "$VALIDATION_ROOT"
mkdir -p "$MODEL_DIR/gemma" "$OUT_DIR"
touch "$MODEL_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
touch "$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
printf '{}\n' >"$MODEL_DIR/gemma/config.json"

cat >"$REGISTRY" <<EOF
[models.ltx2_3_distilled]
display_name = "LTX 2.3 distilled validation fake"
checkpoint_path = "$MODEL_DIR/ltx-2.3-22b-distilled-1.1.safetensors"
config_path = ""
gemma_root = "$MODEL_DIR/gemma"
text_encoder_path = ""
vae_path = ""
spatial_upsampler_path = "$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
temporal_upsampler_path = ""
supports_audio = false
supports_t2v = true
supports_i2v = true
supports_fp8_cast = false
supports_fp8_scaled_mm = false
quantization = "none"
preferred_profiles = ["colab_tiny", "colab_eco", "colab_balanced"]
notes = "Validation-only registry entry for fake worker pipeline."
EOF

LTX_WORKER_TEST_FAKE_PIPELINE=1 \
LTX_WORKER_TEST_FAIL_FIRST_GENERATE=cuda_oom \
LTX_WORKER_TEST_FAIL_MARKER="$MARKER" \
./target/release/ltx-runner generate \
  --profile colab_quality \
  --model ltx2_3_distilled \
  --mode text-to-video \
  --prompt "validation fake worker prompt" \
  --width 1280 \
  --height 720 \
  --frames 97 \
  --fps 16 \
  --steps 30 \
  --out-dir "$OUT_DIR" \
  --model-registry "$REGISTRY" \
  --jsonl-events \
  --json >"$VALIDATION_ROOT/generate_result.json"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("/content/ltx_validation")
result = json.loads((root / "generate_result.json").read_text())
assert result["status"] == "success", result
assert result.get("retried_after_cuda_oom") is True, result
job_dir = Path(result["job_dir"])
assert (job_dir / "output.mp4").exists(), job_dir
events = (job_dir / "events.jsonl").read_text()
assert "retry_after_cuda_oom" in events
resolved = json.loads((job_dir / "resolved_request.json").read_text())
assert resolved["profile"] == "colab_tiny", resolved
assert resolved["width"] == 512, resolved
assert resolved["height"] == 512, resolved
assert resolved["frames"] == 33, resolved
assert resolved["fps"] == 8, resolved
assert resolved["steps"] == 8, resolved
assert any(d["field"] == "profile" for d in resolved["downgrades"])
metadata = json.loads((job_dir / "metadata.json").read_text())
assert metadata["status"] == "success", metadata
print(json.dumps({"validated_job_dir": str(job_dir), "output": result["output"]}, indent=2))
PY

echo "Colab smoke validation passed. Artifacts: $VALIDATION_ROOT"
