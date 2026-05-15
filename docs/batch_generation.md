# Batch Generation

Batch input is JSONL: one JSON object per line.

```json
{"id":"dog-tokyo-001","mode":"text-to-video","prompt":"a cinematic shot of a small white dog walking through Tokyo at night","negative_prompt":"low quality, blurry","seed":12345,"width":512,"height":512,"frames":33,"fps":8,"profile":"auto","model":"auto"}
```

Run:

```bash
./target/release/ltx-runner batch \
  --jobs configs/batch.example.jsonl \
  --profile auto \
  --model auto \
  --out-dir /content/outputs \
  --jsonl-events \
  --continue-on-error
```

Use fixed seeds for reproducibility. Resume after errors with `--start-index`. Limit a run with `--max-jobs`.

Prompt variation example:

```json
{"id":"city-001","prompt":"wide shot of Tokyo at night, rain, realistic reflections","seed":1001}
{"id":"city-002","prompt":"close street-level shot of Tokyo at night, rain, realistic reflections","seed":1002}
```

Batch writes `batch_summary.json` and per-job metadata under `/content/outputs/jobs/{job_id}/`.
