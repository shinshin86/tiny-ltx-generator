# Colab

Colab is the Phase 1 target. `/content` is temporary and disappears when the runtime is reset, so copy outputs to Drive when you need persistence.

## Setup

```bash
export TINY_LTX_REPO_URL="https://github.com/shinshin86/tiny-ltx-generator.git"
git clone "$TINY_LTX_REPO_URL" /content/tiny-ltx-generator
cd /content/tiny-ltx-generator
bash scripts/bootstrap_colab.sh --download-models
```

Optional Drive mount:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Recommended paths:

- project: `/content/tiny-ltx-generator`
- outputs: `/content/outputs`
- temp: `/content/ltx_tmp`
- runtime models: `/content/models`
- Drive outputs: `/content/drive/MyDrive/tiny-ltx-generator/outputs`
- Drive models: `/content/drive/MyDrive/tiny-ltx-generator/models`

## Model Cache Strategy

The default path is to download models into `/content/models` on each setup:

```bash
LTX_DOWNLOAD_VARIANT=ltx2_3_distilled_fp8 bash scripts/bootstrap_colab.sh --download-models
```

This keeps Drive optional. Use `LTX_DOWNLOAD_VARIANT=ltx2_3_distilled` when FP8 is unavailable or not desired. If repeated runtimes need persistence, you can still keep large downloads in Drive and copy or symlink selected files into `/content/models`. Edit `configs/model_registry.toml` so every required file path is explicit.

To try the experimental Sulphur 2 LTX 2.3-derived model explicitly:

```bash
LTX_DOWNLOAD_VARIANT=sulphur_2_dev_fp8mixed bash scripts/bootstrap_colab.sh --download-models
./target/release/ltx-runner generate \
  --profile colab_quality \
  --model sulphur_2_dev_fp8mixed \
  --mode text-to-video \
  --prompt "a small cat walking across a sunlit wooden floor, realistic video" \
  --width 768 --height 512 --frames 49 --fps 24 \
  --steps 40 \
  --out-dir /content/outputs \
  --jsonl-events
```

Sulphur variants are not selected by `--model auto`. Validate the first output visually before using them in batch jobs. The upstream model card references a distill LoRA workflow, but this CLI does not apply LoRA adapters yet.

## CLI From Notebook Cells

```bash
./target/release/ltx-runner check --json
```

```bash
./target/release/ltx-runner generate \
  --profile colab_balanced \
  --model auto \
  --mode text-to-video \
  --prompt "a cinematic shot of a small white dog walking through Tokyo at night" \
  --width 768 --height 512 --frames 49 --fps 24 \
  --out-dir /content/outputs \
  --jsonl-events
```

Use a landscape LTX-shaped smoke test such as `768x512` for quality checks. `512x512` is only kept for low-VRAM fallback and CUDA OOM retry paths.

External notebook control tools can run these commands like any other shell cell. This project does not implement any external control protocol.

## Batch Prompts

```bash
./target/release/ltx-runner batch \
  --jobs configs/batch.example.jsonl \
  --profile auto \
  --model auto \
  --out-dir /content/outputs \
  --jsonl-events
```

## Copy Outputs To Drive

```bash
mkdir -p /content/drive/MyDrive/tiny-ltx-generator/outputs
cp -r /content/outputs/jobs /content/drive/MyDrive/tiny-ltx-generator/outputs/
```

## Validation

For Colab smoke validation, including fake worker CUDA OOM retry verification, see `docs/colab_validation.md`.
