#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser(description="Export one LTX component to ONNX. Phase 3 scaffold.")
    parser.add_argument("--component", required=True, choices=["vae-decoder", "vae-encoder", "text-encoder", "spatial-upscaler", "temporal-upscaler", "transformer"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise SystemExit(f"ONNX export for {args.component} is scaffolded for Phase 3; no export was performed.")


if __name__ == "__main__":
    main()
