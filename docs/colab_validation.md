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

## Pre-generation checks

Before running the ComfyUI adapter on Colab, run:

```bash
./target/release/ltx-runner validate-template \
  --workflow /content/tiny-ltx-generator/fixtures/ltx23_comfy_template.json \
  --manifest /content/tiny-ltx-generator/fixtures/ltx23_template_manifest.example.json \
  --json
```

The checked-in canonical workflow fixture is the actual ComfyUI LTX-2.3 template. The manifest must be updated alongside it, and tests must fail if the node ids or widget indexes drift.
