# Colab Validation

The Colab validation flow should be reproducible from notebook cells and from Colab MCP.

1. Start an A100 runtime when available.
2. Clone or update `https://github.com/shinshin86/tiny-ltx-generator`.
3. Download the ComfyUI LTX-2.3 template and required model artifacts.
4. Run Rust tests.
5. Run the template contract check.
6. Generate one short cat-walking video.
7. Create a contact sheet from the output MP4.
8. Verify that CLI success matches actual output validity.

The first real generation test must run through the template-driven path, not a handcrafted prompt dictionary.

## One-command smoke

For the current TDD path, use:

```bash
cd /content/tiny-ltx-generator
scripts/run_ltx23_cat_smoke_colab.sh
```

This command is intentionally Colab-only. It installs Rust if needed, installs ComfyUI and ComfyUI-LTXVideo, downloads the model files referenced by the canonical LTX-2.3 template, runs `cargo test --workspace`, builds `ltx-runner`, prepares the job artifacts, and runs `scripts/comfy_headless_adapter.py` without `--dry-run`.

The smoke script must fail if `output.mp4` is missing or empty. A later visual gate should replace the current file-size check with frame-level inspection so corrupted video cannot be reported as success.

## Pre-generation checks

Before running the ComfyUI adapter on Colab, run:

```bash
./target/release/ltx-runner validate-template \
  --workflow /content/tiny-ltx-generator/fixtures/ltx23_comfy_template.json \
  --manifest /content/tiny-ltx-generator/fixtures/ltx23_template_manifest.example.json \
  --json
```

The checked-in canonical workflow fixture is the actual ComfyUI LTX-2.3 template. The manifest must be updated alongside it, and tests must fail if the node ids or widget indexes drift.

## Required ComfyUI model locations

The setup script downloads into the ComfyUI directories used by the template:

```text
/content/ComfyUI/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors
/content/ComfyUI/models/loras/ltx-2.3-22b-distilled-lora-384.safetensors
/content/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
/content/ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors
```

If a gated artifact ever requires authentication, set `HF_TOKEN` in the Colab environment. The token must not be written to the repository or to job metadata.
