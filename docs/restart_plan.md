# Restart Plan

## Why the first attempt is archived

The previous implementation tried to bridge several approaches at once: `ltx-pipelines`, hand-built ComfyUI prompt dictionaries, model registry growth, and CLI orchestration. That made failures hard to isolate.

The clearest finding was that the user's target is not generic LTX inference. The target is the lightweight ComfyUI LTX-2.3 workflow behavior, operated from a Rust-first Colab CLI.

## New shortest path

1. Treat the ComfyUI LTX-2.3 template as the source of truth.
2. Write Rust tests that reject missing core nodes and handcrafted graph construction.
3. Build a template patcher that changes only safe parameters.
4. Run the patched template through a thin Python/Comfy adapter on Colab.
5. Mark generation successful only after output, metadata, logs, and visual sanity checks exist.

## Non-goals

- No MCP implementation.
- No web UI.
- No HTTP server.
- No Docker requirement.
- No local real-generation tests.
- No hand-written replacement of the ComfyUI LTX-2.3 sampler graph in Phase 1.
