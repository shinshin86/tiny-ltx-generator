# Colab Validation

This document validates behavior that unit tests cannot fully cover: CLI-to-worker lifecycle, JSONL events, metadata files, batch persistence, and CUDA OOM retry orchestration.

The project itself does not implement any external notebook control protocol. An external Colab control tool can run these cells and inspect outputs like a normal Colab user.

## Required Runtime

Use a Colab GPU runtime. The fake worker validation does not run the real LTX model, but `ltx-runner generate` still performs GPU/profile checks before starting the worker.

## Cell 1: Clone Or Update

```bash
cd /content
export TINY_LTX_REPO_URL="https://github.com/shinshin86/tiny-ltx-generator.git"
if [ -d tiny-ltx-generator/.git ]; then
  git -C tiny-ltx-generator pull
else
  git clone "$TINY_LTX_REPO_URL" tiny-ltx-generator
fi
cd /content/tiny-ltx-generator
```

## Cell 2: Bootstrap Without Real Model Download

Use this for smoke validation. It avoids downloading large model weights.

```bash
bash scripts/bootstrap_colab.sh
```

Expected:

- `cargo build --release -p ltx-runner` succeeds
- `ltx-runner check` prints GPU, torch, ffmpeg, and profile information

## Cell 3: Run Unit And Worker Tests

```bash
bash scripts/test_colab.sh
```

Expected:

- Rust tests pass
- Python worker tests pass
- No real LTX model files are required

## Cell 4: Run Fake Worker Integration Smoke

```bash
bash scripts/validate_colab_smoke.sh
```

Expected:

- `ltx-runner check --json` writes `/content/ltx_check.json`
- the fake worker intentionally returns one structured CUDA OOM
- `ltx-runner generate` retries once
- `resolved_request.json` records downgrade fields
- `events.jsonl` includes `retry_after_cuda_oom`
- `metadata.json` has `status = success`
- placeholder `output.mp4` exists under `/content/ltx_validation/outputs/jobs/{job_id}/`

This validates the CLI orchestration path without requiring a real model load.

## Cell 5: Real Model Download And Tiny Generation

Run only when model download time and storage are acceptable.

```bash
LTX_DOWNLOAD_VARIANT=ltx2_3_distilled bash scripts/bootstrap_colab.sh --download-models
./target/release/ltx-runner generate \
  --profile colab_tiny \
  --model auto \
  --mode text-to-video \
  --prompt "a small white dog walking through Tokyo at night, realistic, soft lighting" \
  --width 512 \
  --height 512 \
  --frames 33 \
  --fps 8 \
  --out-dir /content/outputs \
  --jsonl-events
```

Expected:

- `configs/model_registry.toml` contains downloaded `/content/models` paths
- `output.mp4`, `metadata.json`, `resolved_request.json`, `events.jsonl`, `worker_stats.json`, and `prompt.txt` are written
- `metadata.json` records the selected model, profile, seed, and any downgrades

## Cell 6: Real Batch Smoke

```bash
./target/release/ltx-runner batch \
  --jobs configs/batch.example.jsonl \
  --profile colab_tiny \
  --model auto \
  --out-dir /content/outputs \
  --jsonl-events \
  --continue-on-error \
  --max-jobs 1
```

Expected:

- the Python worker remains alive for the batch command
- per-job metadata is written under `/content/outputs/jobs/{job_id}/`
- `/content/outputs/batch_summary.json` exists

## What Colab MCP Should Capture

For each validation run, capture:

- notebook cell output
- `/content/ltx_check.json`
- `/content/ltx_validation/generate_result.json`
- latest job directory path
- `metadata.json`
- `resolved_request.json`
- `events.jsonl`
- `worker_stats.json`

The key pass/fail checks are deterministic in `scripts/validate_colab_smoke.sh`; real model generation remains dependent on Colab GPU and model download availability.
