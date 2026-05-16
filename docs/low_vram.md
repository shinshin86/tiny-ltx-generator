# Low VRAM

LTX 2.3 is heavy because video generation holds temporal latents, transformer activations, VAE decode buffers, and sometimes upscaler/audio components. Colab free-tier GPUs can vary, so profile selection uses free VRAM when available.

Profiles:

- `colab_tiny`: lowest memory fallback, 512x512, 33 frames, 8 fps, dev FP8 plus distilled LoRA preferred
- `colab_eco`: moderate low memory, 512x512, 49 frames, 12 fps
- `colab_balanced`: larger Colab GPU, 768x512, 65 frames
- `colab_quality`: high-memory GPU, 1280x720, longer clips

Audio is disabled by default because it increases memory and runtime. Upscaling is disabled by default in `colab_tiny` and `colab_eco` for the same reason.

For visual quality checks, prefer a landscape LTX-shaped size such as 768x512 or 1280x720. The square 512x512 path exists for memory fallback and smoke tests, not for judging model quality.

FP8 and distilled models reduce memory by shrinking weights and reducing inference steps. The preferred direction is the same lightweight composition that works well in ComfyUI: FP8 video checkpoint, distilled LoRA when needed, and quantized Gemma text encoder. The current `ltx-pipelines` backend can load the FP8 checkpoint and LoRA, but it still uses the package's `gemma_root` loader path rather than ComfyUI's split `gemma_3_12B_it_fp8_scaled.safetensors` or `gemma_3_12B_it_fp4_mixed.safetensors` loaders. Treat that path as transitional until the Comfy-compatible lightweight backend is implemented.

The standalone `ltx2_3_distilled_fp8` checkpoint has produced noisy output when an additional `fp8-cast` pass is applied, so FP8 checkpoints are loaded with `quantization = "none"`. If a quantization mode is not exposed by the installed pipeline, the worker returns a structured error instead of silently continuing.

When CUDA OOM happens, reduce:

1. frames
2. resolution
3. steps
4. fps
5. upscaler
6. audio
7. model variant

Use `--no-auto-downgrade` when you want hard failure instead of profile-based reduction.

If the Python worker returns a structured CUDA OOM error, `generate` and `batch` retry once when auto downgrade is enabled. The retry switches to `colab_tiny`, caps the request to 512x512, 33 frames, 8 fps, and 8 steps, then selects the first configured low-VRAM model from the registry. Every changed field is recorded in the job downgrade list.

For `colab_tiny` and `colab_eco`, the Python worker requests `ltx-pipelines` disk offload when the selected pipeline exposes `OffloadMode`. Some `ltx-pipelines` builds do not expose that API; in that case the worker records an `offload_unavailable` log event and continues without pretending offload is active. Override this with `LTX_OFFLOAD_MODE=none`, `cpu`, or `disk` when needed.
