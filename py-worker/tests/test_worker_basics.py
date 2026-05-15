from ltx_worker.memory import torch_health, memory_stats
from ltx_worker.generate import Generator
from ltx_worker.generate import _image_conditioning_input
from ltx_worker.model_manager import ModelManager
from ltx_worker.output import persist_result
from ltx_worker.validators import validate_request
from ltx_worker.errors import WorkerError
import sys
import types


def test_torch_health_shape():
    health = torch_health()
    assert "import_ok" in health
    assert "cuda_available" in health


def test_memory_stats_no_raise():
    stats = memory_stats()
    assert "torch_import_ok" in stats


def test_validate_request_accepts_text_to_video():
    validate_request({
        "mode": "text-to-video",
        "prompt": "hello",
        "width": 512,
        "height": 512,
        "frames": 33,
    })


def test_validate_request_rejects_bad_frames():
    try:
        validate_request({
            "mode": "text-to-video",
            "prompt": "hello",
            "width": 512,
            "height": 512,
            "frames": 40,
        })
    except WorkerError as exc:
        assert exc.code == "validation_error"
    else:
        raise AssertionError("bad frame count should fail")


def test_model_manager_rejects_unsupported_registry_quantization():
    manager = ModelManager(lambda *_args: None)
    try:
        manager._quantization_kwargs({
            "quantization": "fp8-cast",
            "supports_fp8_cast": False,
            "supports_fp8_scaled_mm": False,
        })
    except WorkerError as exc:
        assert exc.code == "unsupported_option"
    else:
        raise AssertionError("unsupported quantization should fail")


def test_model_manager_reuses_loaded_pipeline_without_reimport():
    events = []
    manager = ModelManager(lambda *_args: events.append(_args))
    manager.pipeline = object()
    manager.loaded_key = ("model-a", "colab_tiny")
    manager.load("req", {"id": "model-a"}, "colab_tiny")
    assert events == []


def test_model_manager_defaults_low_vram_profiles_to_disk_offload(monkeypatch):
    class FakeOffloadMode:
        DISK = "disk"
        CPU = "cpu"
        NONE = "none"

        def __new__(cls, value):
            if value not in {"disk", "cpu", "none"}:
                raise ValueError(value)
            return value

    fake_types = types.SimpleNamespace(OffloadMode=FakeOffloadMode)
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils.types", fake_types)
    manager = ModelManager(lambda *_args: None)
    assert manager._offload_mode("colab_tiny") == "disk"
    assert manager._offload_mode("colab_eco") == "disk"
    assert manager._offload_mode("colab_balanced") is None


def test_model_manager_prefers_from_config_when_config_path_exists():
    calls = []

    class FakePipeline:
        @classmethod
        def from_config(cls, config_path, quantization=None, offload_mode=None):
            calls.append({
                "config_path": config_path,
                "quantization": quantization,
                "offload_mode": offload_mode,
            })
            return cls()

    manager = ModelManager(lambda *_args: None)
    pipeline = manager._load_from_config(
        FakePipeline,
        {"config_path": "/content/models/config.yaml"},
        {"quantization": "q"},
        "disk",
    )
    assert isinstance(pipeline, FakePipeline)
    assert calls == [{
        "config_path": "/content/models/config.yaml",
        "quantization": "q",
        "offload_mode": "disk",
    }]


def test_model_manager_rejects_full_model_on_distilled_loading_path():
    class DistilledPipeline:
        pass

    manager = ModelManager(lambda *_args: None)
    try:
        manager.pipeline = None
        manager.loaded_key = None
        manager._find_pipeline = lambda *_args: DistilledPipeline
        manager.load("req", {
            "id": "ltx2_3_full",
            "checkpoint_path": "/content/models/ltx-2.3-22b-dev.safetensors",
            "gemma_root": "/content/models/gemma",
            "spatial_upsampler_path": "/content/models/upscaler.safetensors",
        }, "colab_balanced")
    except WorkerError as exc:
        assert exc.code == "unsupported_pipeline"
        assert "distilled model" in exc.message
    else:
        raise AssertionError("full model should not be loaded through DistilledPipeline")


def test_generator_maps_request_to_pipeline_kwargs(monkeypatch):
    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = []

    class FakePipeline:
        def __call__(self, prompt, width, height, num_frames, frame_rate, seed, images=None, **kwargs):
            kwargs.update({
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "seed": seed,
                "images": images,
            })
            calls.append(kwargs)
            return object()

    fake_torch = types.SimpleNamespace(inference_mode=lambda: InferenceMode())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("ltx_worker.generate.persist_result", lambda _result, path, **_kwargs: path)

    events = []
    generator = Generator(lambda *_args: events.append(_args))
    generator.models.pipeline = FakePipeline()
    generator.models.load = lambda *_args: None

    result = generator.generate("req", {
        "request": {
            "mode": "text-to-video",
            "prompt": "hello",
            "width": 512,
            "height": 512,
            "frames": 33,
            "fps": 8,
            "seed": 123,
            "steps": 8,
            "guidance_scale": 1.0,
            "output_path": "/tmp/out.mp4",
        },
        "model": {"id": "model-a"},
    })

    assert result["output_path"] == "/tmp/out.mp4"
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["num_frames"] == 33
    assert calls[0]["frame_rate"] == 8.0
    assert calls[0]["num_inference_steps"] == 8


def test_generator_uses_ltx_image_conditioning_input_for_images_parameter(monkeypatch):
    class FakeImageConditioningInput(tuple):
        def __new__(cls, path, frame_idx, strength, crf=23):
            value = tuple.__new__(cls, (path, frame_idx, strength, crf))
            value.path = path
            value.frame_idx = frame_idx
            value.strength = strength
            value.crf = crf
            return value

    fake_args = types.SimpleNamespace(ImageConditioningInput=FakeImageConditioningInput)
    monkeypatch.setitem(sys.modules, "ltx_pipelines", types.ModuleType("ltx_pipelines"))
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils", types.ModuleType("ltx_pipelines.utils"))
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils.args", fake_args)

    conditioning = _image_conditioning_input("/tmp/input.png")

    assert isinstance(conditioning, FakeImageConditioningInput)
    assert conditioning.path == "/tmp/input.png"
    assert conditioning.frame_idx == 0
    assert conditioning.strength == 1.0


def test_persist_result_encodes_ltx_tuple_with_chunk_count(monkeypatch, tmp_path):
    calls = []

    def fake_encode_video(**kwargs):
        calls.append(kwargs)
        Path = __import__("pathlib").Path
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    def fake_get_video_chunks_number(num_frames, tiling_config):
        assert num_frames == 33
        assert tiling_config == "tiling"
        return 7

    monkeypatch.setitem(sys.modules, "ltx_pipelines", types.ModuleType("ltx_pipelines"))
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils", types.ModuleType("ltx_pipelines.utils"))
    monkeypatch.setitem(
        sys.modules,
        "ltx_pipelines.utils.media_io",
        types.SimpleNamespace(encode_video=fake_encode_video),
    )
    monkeypatch.setitem(sys.modules, "ltx_core", types.ModuleType("ltx_core"))
    monkeypatch.setitem(sys.modules, "ltx_core.model", types.ModuleType("ltx_core.model"))
    monkeypatch.setitem(
        sys.modules,
        "ltx_core.model.video_vae",
        types.SimpleNamespace(get_video_chunks_number=fake_get_video_chunks_number),
    )

    output = tmp_path / "out.mp4"
    path = persist_result(("video", "audio"), output, fps=8, num_frames=33, tiling_config="tiling")

    assert path == str(output)
    assert output.read_bytes() == b"mp4"
    assert calls[0]["video"] == "video"
    assert calls[0]["audio"] == "audio"
    assert calls[0]["fps"] == 8
    assert calls[0]["video_chunks_number"] == 7


def test_persist_result_encodes_ltx_tuple_with_safe_default_chunk_count(monkeypatch, tmp_path):
    calls = []

    def fake_encode_video(**kwargs):
        calls.append(kwargs)
        Path = __import__("pathlib").Path
        Path(kwargs["output_path"]).write_bytes(b"mp4")

    monkeypatch.setitem(sys.modules, "ltx_pipelines", types.ModuleType("ltx_pipelines"))
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils", types.ModuleType("ltx_pipelines.utils"))
    monkeypatch.setitem(
        sys.modules,
        "ltx_pipelines.utils.media_io",
        types.SimpleNamespace(encode_video=fake_encode_video),
    )

    output = tmp_path / "out.mp4"
    persist_result(("video", None), output, fps=12, num_frames=None, tiling_config=None)

    assert output.read_bytes() == b"mp4"
    assert calls[0]["video_chunks_number"] == 1


def test_generator_test_cuda_oom_hook_fails_once(monkeypatch, tmp_path):
    generator = Generator(lambda *_args: None)
    marker = tmp_path / "oom.marker"
    monkeypatch.setenv("LTX_WORKER_TEST_FAIL_FIRST_GENERATE", "cuda_oom")
    monkeypatch.setenv("LTX_WORKER_TEST_FAIL_MARKER", str(marker))

    try:
        generator._maybe_raise_test_cuda_oom_once()
    except RuntimeError as exc:
        assert "out of memory" in str(exc).lower()
    else:
        raise AssertionError("first call should simulate CUDA OOM")

    generator._maybe_raise_test_cuda_oom_once()
    assert marker.exists()


def test_generator_fake_pipeline_writes_requested_output(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, check, stdout, stderr):
        calls.append(args)
        Path = __import__("pathlib").Path
        Path(args[-1]).write_bytes(b"fake mp4")

    monkeypatch.setenv("LTX_WORKER_TEST_FAKE_PIPELINE", "1")
    monkeypatch.setattr("ltx_worker.generate.subprocess.run", fake_run)

    generator = Generator(lambda *_args: None)
    output = tmp_path / "out.mp4"
    result = generator.generate("req", {
        "request": {
            "mode": "text-to-video",
            "prompt": "hello",
            "width": 512,
            "height": 512,
            "frames": 33,
            "fps": 8,
            "seed": 123,
            "steps": 8,
            "guidance_scale": 1.0,
            "output_path": str(output),
        },
        "model": {"id": "model-a"},
    })

    assert result["test_fake_pipeline"] is True
    assert result["output_path"] == str(output)
    assert output.read_bytes() == b"fake mp4"
    assert calls
