# Project Plan

## Goal

Build a Colab-first Rust CLI that can run lightweight LTX video generation workflows with deterministic job artifacts and clear validation.

The initial execution path uses a checked-in ComfyUI-compatible template or API prompt as the source of truth. Rust owns validation, job preparation, metadata, output contracts, and memory policy.

## Shortest Path

1. Treat the checked-in LTX template or API prompt as the execution contract.
2. Write Rust tests that reject missing core nodes, missing model artifacts, and unsupported template shapes.
3. Patch only explicit user parameters such as prompt, seed, dimensions, frame count, fps, and output prefix.
4. Run the prepared prompt through a thin Colab-only Python/Comfy adapter.
5. Mark generation successful only after output, metadata, logs, and visual sanity checks exist.

## Non-goals

- No web UI.
- No public HTTP API.
- No Docker requirement.
- No local real-generation tests.
- No project-specific remote-control protocol.
- No handwritten replacement for the LTX sampler graph in Phase 1.
