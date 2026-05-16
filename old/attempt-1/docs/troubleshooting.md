# Troubleshooting

## Missing Model Files

Exit code `3` means one or more configured model paths do not exist. Edit `configs/model_registry.toml`.

## No GPU

Exit code `4` means CUDA GPU generation is unavailable. `ltx-runner check --json` still works.

## CUDA OOM

Exit code `5` means the worker caught a CUDA out-of-memory error. Use `colab_tiny`, fewer frames, smaller resolution, and the dev FP8 plus distilled LoRA default when available.

If A100 still OOMs with `ltx2_3_dev_fp8_distilled_lora`, first confirm the registry contains `extra = { backend = "comfy_ltx" }` and `text_encoder_path` points to a split quantized Gemma safetensors file such as `gemma_3_12B_it_fp4_mixed.safetensors`. If the worker falls back to `ltx-pipelines`, that is not the intended lightweight path.

## Valid MP4 But Noise Or Static

A valid `output.mp4` only proves encoding succeeded. If the video looks like noise or static, first verify an official Lightricks model before testing community variants:

```bash
./target/release/ltx-runner generate \
  --profile colab_balanced \
  --model ltx2_3_dev_fp8_distilled_lora \
  --mode text-to-video \
  --prompt "a small cat walking across a sunlit wooden floor, realistic video" \
  --width 768 --height 512 --frames 49 --fps 12 \
  --out-dir /content/outputs \
  --jsonl-events
```

`colab_balanced` caps FPS at 12. For 24 FPS on A100 or similar GPUs, use `colab_quality`.

Common causes are a wrong model/pipeline pairing, an experimental community checkpoint, applying extra FP8 casting to a checkpoint that is already FP8-mixed, too few steps for a full/dev model, using a checkpoint that expects a LoRA workflow without applying that LoRA, or using the 512x512 low-memory fallback as a quality check. For full/dev models, start around `--steps 40`; distilled models can use fewer steps. Sulphur variants must be selected explicitly and visually validated before batch use.

For ComfyUI-like LTX 2.3 runs, verify that the model set includes the FP8 checkpoint, distilled LoRA, spatial upscaler, and a quantized split Gemma text encoder. The default Comfy-compatible backend consumes `text_encoder_path`; missing or empty values fail early instead of pretending generation succeeded.

The headless Comfy route must keep the LTX 2.3 conditioning and latent shape intact. It should use the core `LTXAVTextEncoderLoader`, `LTXVConditioning`, video/audio latent concat, the distilled LoRA at the configured strength, and the Comfy-style manual sigma schedule. A plain `CLIPLoader(type="ltxv")` plus image-only latent path can load the text encoder but still fail or produce invalid output because LTX 2.3 expects AV conditioning.

## Unsupported Pipeline

Exit code `8` means the installed `ltx-pipelines` package does not expose the requested pipeline or option. Check package version and `config_path`.
