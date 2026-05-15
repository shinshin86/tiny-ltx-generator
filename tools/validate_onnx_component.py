#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX component against PyTorch baseline. Phase 3 scaffold.")
    parser.add_argument("--component", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise SystemExit("ONNX validation is scaffolded for Phase 3; no validation was performed.")


if __name__ == "__main__":
    main()
