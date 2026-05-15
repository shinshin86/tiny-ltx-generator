#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


COMPONENTS = {
    "vae-decoder",
    "vae-encoder",
    "text-encoder",
    "spatial-upscaler",
    "temporal-upscaler",
    "transformer",
}


def tensor_shape(value):
    shape = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape


def flatten_numbers(value):
    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    raise TypeError(f"non-numeric tensor value: {value!r}")


def load_tensor(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".npy":
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("reading .npy requires numpy") from exc
        return np.load(path).tolist()
    raise ValueError(f"unsupported tensor file extension: {path.suffix}")


def compare_tensors(baseline, candidate):
    baseline_shape = tensor_shape(baseline)
    candidate_shape = tensor_shape(candidate)
    if baseline_shape != candidate_shape:
        return {
            "ok": False,
            "baseline_shape": baseline_shape,
            "candidate_shape": candidate_shape,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "error": "shape_mismatch",
        }
    baseline_flat = flatten_numbers(baseline)
    candidate_flat = flatten_numbers(candidate)
    if len(baseline_flat) != len(candidate_flat):
        return {
            "ok": False,
            "baseline_shape": baseline_shape,
            "candidate_shape": candidate_shape,
            "max_absolute_error": None,
            "mean_absolute_error": None,
            "error": "flat_length_mismatch",
        }
    errors = [abs(a - b) for a, b in zip(baseline_flat, candidate_flat)]
    max_error = max(errors) if errors else 0.0
    mean_error = sum(errors) / len(errors) if errors else 0.0
    return {
        "ok": True,
        "baseline_shape": baseline_shape,
        "candidate_shape": candidate_shape,
        "max_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "error": None,
    }


def threshold_pass(report, max_abs_error, mean_abs_error):
    if not report["ok"]:
        return False
    if max_abs_error is not None and report["max_absolute_error"] > max_abs_error:
        return False
    if mean_abs_error is not None and report["mean_absolute_error"] > mean_abs_error:
        return False
    return True


def finite_threshold(value):
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError("threshold must be a finite non-negative number")
    return value


def main():
    parser = argparse.ArgumentParser(description="Validate an exported ONNX component against a PyTorch baseline output.")
    parser.add_argument("--component", required=True, choices=sorted(COMPONENTS))
    parser.add_argument("--onnx", help="Path to exported ONNX component. Recorded in the report.")
    parser.add_argument("--config", help="Path to component validation config. Recorded in the report.")
    parser.add_argument("--baseline-output", required=True, help="PyTorch baseline tensor output as .json or .npy")
    parser.add_argument("--candidate-output", required=True, help="ONNX candidate tensor output as .json or .npy")
    parser.add_argument("--max-abs-error", type=float, default=1e-3)
    parser.add_argument("--mean-abs-error", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    max_abs_error = finite_threshold(args.max_abs_error)
    mean_abs_error = finite_threshold(args.mean_abs_error)
    report = compare_tensors(load_tensor(args.baseline_output), load_tensor(args.candidate_output))
    report.update({
        "component": args.component,
        "onnx": args.onnx,
        "config": args.config,
        "thresholds": {
            "max_absolute_error": max_abs_error,
            "mean_absolute_error": mean_abs_error,
        },
    })
    report["passed"] = threshold_pass(report, max_abs_error, mean_abs_error)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"component: {args.component}")
        print(f"baseline_shape: {report['baseline_shape']}")
        print(f"candidate_shape: {report['candidate_shape']}")
        print(f"max_absolute_error: {report['max_absolute_error']}")
        print(f"mean_absolute_error: {report['mean_absolute_error']}")
        print(f"passed: {report['passed']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
