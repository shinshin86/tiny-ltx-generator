# Troubleshooting

## Missing Model Files

Exit code `3` means one or more configured model paths do not exist. Edit `configs/model_registry.toml`.

## No GPU

Exit code `4` means CUDA GPU generation is unavailable. `ltx-runner check --json` still works.

## CUDA OOM

Exit code `5` means the worker caught a CUDA out-of-memory error. Use `colab_tiny`, fewer frames, smaller resolution, and distilled FP8 if available.

## Unsupported Pipeline

Exit code `8` means the installed `ltx-pipelines` package does not expose the requested pipeline or option. Check package version and `config_path`.
