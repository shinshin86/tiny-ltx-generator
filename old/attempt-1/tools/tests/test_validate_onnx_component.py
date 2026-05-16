import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_onnx_component.py"
SPEC = importlib.util.spec_from_file_location("validate_onnx_component", MODULE_PATH)
validate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate)


def test_tensor_shape_nested_lists():
    assert validate.tensor_shape([[[1, 2], [3, 4]]]) == [1, 2, 2]


def test_compare_tensors_reports_error_metrics():
    report = validate.compare_tensors([[1.0, 2.0]], [[1.1, 1.9]])
    assert report["ok"] is True
    assert round(report["max_absolute_error"], 6) == 0.1
    assert round(report["mean_absolute_error"], 6) == 0.1


def test_compare_tensors_detects_shape_mismatch():
    report = validate.compare_tensors([[1.0, 2.0]], [[1.0], [2.0]])
    assert report["ok"] is False
    assert report["error"] == "shape_mismatch"


def test_threshold_pass_requires_both_thresholds():
    report = validate.compare_tensors([[1.0, 2.0]], [[1.01, 2.0]])
    assert validate.threshold_pass(report, max_abs_error=0.02, mean_abs_error=0.01)
    assert not validate.threshold_pass(report, max_abs_error=0.001, mean_abs_error=0.01)
