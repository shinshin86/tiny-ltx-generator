from .memory import memory_stats
from .model_manager import ModelManager
from .output import persist_result
from .validators import validate_request
import inspect
import os
import subprocess
from pathlib import Path


class Generator:
    def __init__(self, emit):
        self.emit = emit
        self.models = ModelManager(emit)

    def load_model(self, request_id, model_entry, profile):
        self.models.load(request_id, model_entry, profile)

    def unload_model(self):
        self.models.unload()

    def generate(self, request_id, payload):
        request = payload.get("request") or {}
        model = payload.get("model") or {}
        validate_request(request)
        self._maybe_raise_test_cuda_oom_once()
        if os.environ.get("LTX_WORKER_TEST_FAKE_PIPELINE") == "1":
            self.emit(request_id, "progress", {"stage": "test_fake_generation", "memory": memory_stats()})
            output = self._write_test_video(request)
            return {"output_path": output, "memory": memory_stats(), "test_fake_pipeline": True}
        self.models.load(request_id, model, request.get("profile"))
        self.emit(request_id, "progress", {"stage": "before_generation", "memory": memory_stats()})
        kwargs = {
            "prompt": request["prompt"],
            "width": request["width"],
            "height": request["height"],
            "num_frames": request["frames"],
            "seed": request["seed"],
        }
        call_signature = inspect.signature(self.models.pipeline.__call__)
        if "fps" in call_signature.parameters:
            kwargs["fps"] = float(request["fps"])
        if "frame_rate" in call_signature.parameters:
            kwargs["frame_rate"] = float(request["fps"])
        if request.get("negative_prompt"):
            kwargs["negative_prompt"] = request["negative_prompt"]
        if request.get("guidance_scale") is not None:
            kwargs["guidance_scale"] = request["guidance_scale"]
        if request.get("steps") is not None:
            kwargs["num_inference_steps"] = request["steps"]
        if "images" in call_signature.parameters:
            kwargs["images"] = []
        if "tiling_config" in call_signature.parameters:
            try:
                from ltx_core.model.video_vae import TilingConfig
                kwargs["tiling_config"] = TilingConfig.default()
            except Exception:
                pass
        if request.get("mode") == "image-to-video":
            if "image" in call_signature.parameters:
                from PIL import Image
                image = Image.open(request["input_image"]).convert("RGB")
                kwargs["image"] = image
            elif "images" in call_signature.parameters:
                kwargs["images"] = [_image_conditioning_input(request["input_image"])]
            else:
                raise RuntimeError("selected pipeline does not expose image conditioning parameters")
        import torch
        with torch.inference_mode():
            result = self.models.pipeline(**kwargs)
        output = persist_result(
            result,
            request["output_path"],
            fps=request["fps"],
            num_frames=request["frames"],
            tiling_config=kwargs.get("tiling_config"),
        )
        self.emit(request_id, "progress", {"stage": "after_generation", "memory": memory_stats(), "output": output})
        return {"output_path": output, "memory": memory_stats()}

    def _maybe_raise_test_cuda_oom_once(self):
        if os.environ.get("LTX_WORKER_TEST_FAIL_FIRST_GENERATE") != "cuda_oom":
            return
        marker = Path(os.environ.get("LTX_WORKER_TEST_FAIL_MARKER", "/content/ltx_tmp/test_cuda_oom_once.marker"))
        marker.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            return
        marker.write_text("failed-once", encoding="utf-8")
        raise RuntimeError("CUDA out of memory: simulated by LTX_WORKER_TEST_FAIL_FIRST_GENERATE")

    def _write_test_video(self, request):
        output = Path(request["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={request['width']}x{request['height']}:r={request['fps']}:d=1",
                "-vf",
                "drawtext=text='tiny-ltx fake worker':fontcolor=white:fontsize=24:x=20:y=20",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(output)


def _image_conditioning_input(path):
    try:
        from ltx_pipelines.utils.args import ImageConditioningInput

        return ImageConditioningInput(path=str(path), frame_idx=0, strength=1.0)
    except Exception:
        return (str(path), 0, 1.0)
