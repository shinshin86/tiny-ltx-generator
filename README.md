# tiny-ltx-generator

This repository has been reset after the first implementation attempt.

The goal is narrower now: run the official ComfyUI LTX-2.3 lightweight workflow from a Rust CLI on Google Colab, while Rust owns validation, job planning, metadata, output contracts, and memory policy.

Phase 1 does not try to reimplement the LTX sampler graph. The ComfyUI workflow template is the source of truth. The CLI should patch only explicit user parameters such as prompt, width, height, duration, fps, seed, and model artifact names.

See:

- [Restart Plan](docs/restart_plan.md)
- [TDD Contract](docs/tdd_contract.md)
- [Colab Validation](docs/colab_validation.md)
