# TDD Contract

Development starts from failing tests.

## Contract tests first

- The LTX-2.3 template must contain the required node types.
- Missing core nodes must fail before generation.
- Handcrafted Python dict graph construction must fail as a design regression.
- Model artifact checks must happen before Colab generation.
- A successful job must include `output.mp4`, `metadata.json`, `resolved_request.json`, `events.jsonl`, and `worker_stats.json`.

## Colab-only integration

Real model execution belongs on Colab. Local development is limited to Rust formatting and Rust tests unless explicitly approved.

## Success definition

The CLI exits 0 only when:

- ComfyUI execution completed.
- The MP4 exists and is readable by ffmpeg.
- Metadata records the exact template, model files, prompt, seed, dimensions, fps, duration, and downgrades.
- A basic visual sanity check says the output is not an all-noise/all-flat placeholder.

## Current tests

- Template contract tests reject incomplete LTX-2.3 workflows.
- The canonical ComfyUI LTX-2.3 template is checked in and validated.
- Template patcher tests verify that only manifest-declared controls are changed.
- CLI contract tests verify `validate-template` and `patch-template`.
- Job artifact tests reject success when `worker_stats.json` or `visual_check.json` is missing or failed.
