# Architecture

`tiny-ltx-generator` keeps Rust independent from Python internals except the JSONL protocol.

Rust responsibilities:

- CLI parsing and deterministic exit codes
- config and model registry loading
- hardware inspection and profile resolution
- request validation and visible downgrade records
- job directory and metadata writing
- batch orchestration
- Python worker lifecycle
- future ONNX Runtime integration

Python responsibilities:

- lazy import of torch, headless ComfyUI-compatible LTX code, and `ltx-pipelines` fallback code
- model/backend loading
- text-to-video and image-to-video generation
- memory stats
- structured CUDA OOM and unsupported-pipeline errors

No browser UI, public API server, tunnel, Docker, or MCP interface is included. The default LTX 2.3 low-memory path may use ComfyUI internals headlessly inside the worker because that is the practical route for matching the lightweight Colab behavior.
