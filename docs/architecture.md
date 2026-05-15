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

- lazy import of torch and `ltx-pipelines`
- model loading
- text-to-video and image-to-video generation
- memory stats
- structured CUDA OOM and unsupported-pipeline errors

No server, web framework, ComfyUI import, node graph, workflow JSON, tunnel, Docker, or MCP interface is included.
