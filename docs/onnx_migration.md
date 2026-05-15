# ONNX Migration

Full LTX ONNX export is not Phase 1. The model is large, multimodal, and pipeline behavior must be validated component by component before enabling ONNX in production paths.

Migration priority:

1. VAE decoder
2. VAE encoder
3. text encoder
4. spatial upscaler
5. temporal upscaler
6. transformer

Validation for every component:

- max absolute error
- mean absolute error
- output shape
- latency
- memory usage

The first validation tool stage compares saved PyTorch baseline output with candidate output:

```bash
python tools/validate_onnx_component.py \
  --component vae-decoder \
  --baseline-output /content/baseline.json \
  --candidate-output /content/candidate.json \
  --json
```

Use `.json` nested numeric arrays for lightweight tests or `.npy` files when numpy is available. The tool reports shape, max absolute error, mean absolute error, and pass/fail thresholds.

Provider fallback:

```text
tensorrt -> cuda -> cpu
cuda -> cpu
cpu
```

TensorRT should not be assumed available in Colab. If ONNX Runtime GPU is unavailable, the engine must fall back to PyTorch.

Build the Rust ONNX scaffold:

```bash
cargo build -p ltx-onnx --features onnx
```
