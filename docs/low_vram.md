# Low VRAM

LTX 2.3 is heavy because video generation holds temporal latents, transformer activations, VAE decode buffers, and sometimes upscaler/audio components. Colab free-tier GPUs can vary, so profile selection uses free VRAM when available.

Profiles:

- `colab_tiny`: lowest memory, 512x512, 33 frames, 8 fps, distilled FP8 preferred
- `colab_eco`: moderate low memory, 512x512, 49 frames, 12 fps
- `colab_balanced`: larger Colab GPU, 768x512, 65 frames
- `colab_quality`: high-memory GPU, 1280x720, longer clips

Audio is disabled by default because it increases memory and runtime. Upscaling is disabled by default in `colab_tiny` and `colab_eco` for the same reason.

FP8 and distilled models reduce memory by shrinking weights and reducing inference steps. If a quantization mode is not exposed by the installed pipeline, the worker returns a structured error instead of silently continuing.

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

For `colab_tiny` and `colab_eco`, the Python worker requests `ltx-pipelines` disk offload when the selected pipeline exposes `OffloadMode`. Override this with `LTX_OFFLOAD_MODE=none`, `cpu`, or `disk` when needed.
