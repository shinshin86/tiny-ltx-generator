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
LTX_DOWNLOAD_VARIANT=ltx2_3_distilled bash scripts/bootstrap_colab.sh --download-models
```

This keeps Drive optional. If repeated runtimes need persistence, you can still keep large downloads in Drive and copy or symlink selected files into `/content/models`. Edit `configs/model_registry.toml` so every required file path is explicit.

## CLI From Notebook Cells

```bash
./target/release/ltx-runner check --json
```

```bash
./target/release/ltx-runner generate \
  --profile auto \
  --model auto \
  --mode text-to-video \
  --prompt "a cinematic shot of a small white dog walking through Tokyo at night" \
  --width 512 --height 512 --frames 33 --fps 8 \
  --out-dir /content/outputs \
  --jsonl-events
```

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
