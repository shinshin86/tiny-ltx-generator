# Troubleshooting

## Missing Model Files

Exit code `3` means one or more configured model paths do not exist. Edit `configs/model_registry.toml`.

## No GPU

Exit code `4` means CUDA GPU generation is unavailable. `ltx-runner check --json` still works.

## CUDA OOM

Exit code `5` means the worker caught a CUDA out-of-memory error. Use `colab_tiny`, fewer frames, smaller resolution, and distilled FP8 if available.

## Valid MP4 But Noise Or Static

A valid `output.mp4` only proves encoding succeeded. If the video looks like noise or static, first verify an official Lightricks model before testing community variants:

```bash
./target/release/ltx-runner generate \
  --profile colab_balanced \
  --model ltx2_3_distilled_fp8 \
  --mode text-to-video \
  --prompt "a small cat walking across a sunlit wooden floor, realistic video" \
  --width 768 --height 512 --frames 49 --fps 24 \
  --out-dir /content/outputs \
  --jsonl-events
```

Common causes are a wrong model/pipeline pairing, an experimental community checkpoint, applying extra FP8 casting to a checkpoint that is already FP8-mixed, too few steps for a full/dev model, using a checkpoint that expects a LoRA workflow without applying that LoRA, or using the 512x512 low-memory fallback as a quality check. For full/dev models, start around `--steps 40`; distilled models can use fewer steps. Sulphur variants must be selected explicitly and visually validated before batch use.

## Unsupported Pipeline

Exit code `8` means the installed `ltx-pipelines` package does not expose the requested pipeline or option. Check package version and `config_path`.
