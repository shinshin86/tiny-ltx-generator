# tiny-ltx-generator

This repository has been reset after the first implementation attempt.

The goal is narrower now: run the official ComfyUI LTX-2.3 lightweight workflow from a Rust CLI on Google Colab, while Rust owns validation, job planning, metadata, output contracts, and memory policy.

Phase 1 does not try to reimplement the LTX sampler graph. The ComfyUI workflow template is the source of truth. The CLI should patch only explicit user parameters such as prompt, width, height, duration, fps, seed, and model artifact names.

See:

- [Restart Plan](docs/restart_plan.md)
- [TDD Contract](docs/tdd_contract.md)
- [Colab Validation](docs/colab_validation.md)

Current verified commands:

```bash
cargo test --workspace

cargo run -p ltx-runner -- validate-template \
  --workflow fixtures/ltx23_comfy_template.json \
  --manifest fixtures/ltx23_template_manifest.example.json \
  --json

cargo run -p ltx-runner -- patch-template \
  --workflow fixtures/ltx23_comfy_template.json \
  --manifest fixtures/ltx23_template_manifest.example.json \
  --prompt "a small orange cat walking through a sunlit room" \
  --negative-prompt "low quality, noisy, distorted" \
  --width 768 \
  --height 512 \
  --duration-seconds 4 \
  --fps 12 \
  --seed 12345 \
  --checkpoint ltx-2.3-22b-dev-fp8.safetensors \
  --text-encoder gemma_3_12B_it_fp4_mixed.safetensors \
  --distilled-lora ltx-2.3-22b-distilled-lora-384.safetensors \
  --lora-strength 0.5 \
  --spatial-upscaler ltx-2.3-spatial-upscaler-x2-1.1.safetensors
```

The next implementation step is the Colab-only headless adapter that runs the patched template through ComfyUI and writes the required job artifacts.

`scripts/comfy_headless_adapter.py --dry-run` validates the adapter request shape. Without `--dry-run`, the adapter passes `api_prompt.json` to ComfyUI's headless executor and writes `worker_stats.json` plus `visual_check.json`.
