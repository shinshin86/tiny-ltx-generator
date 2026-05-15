# tiny-ltx-generator

`tiny-ltx-generator` is a Colab-first local video generation CLI for LTX models, centered on LTX 2.3.

It is not ComfyUI, not a web UI, not a public API server, and does not implement MCP. The runtime interface is only the `ltx-runner` CLI, designed to run from Google Colab notebook cells or shell commands.

## Architecture

```text
notebook cell
  -> ltx-runner Rust CLI
  -> Python JSONL worker
  -> official Lightricks ltx-pipelines
  -> /content/outputs/jobs/{job_id}/
```

Rust handles orchestration, validation, profile resolution, metadata, batch execution, hardware checks, and the future ONNX path. Python is intentionally small and only owns LTX inference.

## Colab Setup

```bash
export TINY_LTX_REPO_URL="https://github.com/shinshin86/tiny-ltx-generator.git"
git clone "$TINY_LTX_REPO_URL" /content/tiny-ltx-generator
cd /content/tiny-ltx-generator
bash scripts/bootstrap_colab.sh --download-models
```

The bootstrap script installs Rust, `uv`, and `ffmpeg` only when missing, builds `ltx-runner`, prepares `/content/outputs`, `/content/ltx_tmp`, and `/content/models`, then runs `ltx-runner check`.

Large model files are downloaded only when `--download-models` is passed. The default download target is `LTX_DOWNLOAD_VARIANT=ltx2_3_distilled_fp8`, which downloads the official LTX 2.3 distilled FP8 checkpoint, spatial upscaler, and Gemma text encoder into `/content/models`, then writes `configs/model_registry.toml`.

Other supported setup variants:

```bash
LTX_DOWNLOAD_VARIANT=ltx2_3_distilled bash scripts/bootstrap_colab.sh --download-models
LTX_DOWNLOAD_VARIANT=ltx2_3_fp8 bash scripts/bootstrap_colab.sh --download-models
LTX_DOWNLOAD_VARIANT=ltx2_3_full bash scripts/bootstrap_colab.sh --download-models
LTX_DOWNLOAD_VARIANT=sulphur_2_dev_fp8mixed bash scripts/bootstrap_colab.sh --download-models
```

Set `HF_TOKEN` when Hugging Face access requires authentication.

Community LTX 2.3-derived variants can be used explicitly after download, for example `--model sulphur_2_dev_fp8mixed`. They are not preferred over official Lightricks models by `--model auto`.

## Configure Model Paths

If you used `--download-models`, `configs/model_registry.toml` is generated automatically. To edit manually:

```bash
cp -n configs/model_registry.example.toml configs/model_registry.toml
nano configs/model_registry.toml
```

Set the exact local paths for checkpoint/config files. Missing model paths fail with exit code `3`; the tool does not pretend generation succeeded.

## Check Runtime

```bash
./target/release/ltx-runner check --json
```

This reports OS, disk, Drive mount state, Python, `uv`, Rust, `ffmpeg`, `nvidia-smi`, parsed VRAM, torch CUDA status, and the recommended Colab profile.

## Tiny Generation

```bash
./target/release/ltx-runner generate \
  --profile auto \
  --model auto \
  --mode text-to-video \
  --prompt "a cinematic shot of a small white dog walking through Tokyo at night, realistic, soft lighting" \
  --width 512 \
  --height 512 \
  --frames 33 \
  --fps 8 \
  --out-dir /content/outputs \
  --jsonl-events
```

Outputs are written under:

```text
/content/outputs/jobs/{job_id}/
  output.mp4
  metadata.json
  resolved_request.json
  events.jsonl
  worker_stats.json
  prompt.txt
```

For smoke checks without real models:

```bash
./target/release/ltx-runner generate \
  --mock \
  --mode text-to-video \
  --prompt "mock smoke test" \
  --out-dir /content/outputs
```

Mock mode is clearly marked in metadata and does not claim real LTX generation.

## Batch Generation

```bash
./target/release/ltx-runner batch \
  --jobs configs/batch.example.jsonl \
  --profile auto \
  --model auto \
  --out-dir /content/outputs \
  --jsonl-events \
  --continue-on-error
```

Each JSONL line is one job. Batch writes `batch_summary.json` plus per-job metadata.

## Reduce Memory Usage

Reduce in this order:

1. frames
2. resolution
3. steps
4. fps
5. upscaler
6. audio
7. model variant

Use `colab_tiny` or `colab_eco`, distilled models, FP8 when supported, and keep audio/upscaling disabled for low-VRAM runs.

When CUDA OOM is reported by the worker and auto downgrade is enabled, `generate` and `batch` retry once with a `colab_tiny`-class request and a configured low-VRAM model. The retry writes the changed profile, model, resolution, frames, fps, and steps into `resolved_request.json` and `metadata.json` downgrade records.

## Tests

Run tests in Colab:

```bash
bash scripts/test_colab.sh
```

Real LTX generation tests should be skipped unless test assets and model paths are explicitly configured.
