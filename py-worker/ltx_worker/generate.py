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
        if getattr(self.models.pipeline, "uses_comfy_backend", False):
            kwargs.update({
                "mode": request.get("mode"),
                "input_image": request.get("input_image"),
                "output_path": request["output_path"],
                "fps": float(request["fps"]),
            })
        call_signature = inspect.signature(self.models.pipeline.__call__)
        if "fps" in call_signature.parameters:
            kwargs["fps"] = float(request["fps"])
        if "frame_rate" in call_signature.parameters:
            kwargs["frame_rate"] = float(request["fps"])
        if _accepts_parameter(call_signature, "negative_prompt"):
            negative_prompt = request.get("negative_prompt")
            if negative_prompt is not None or _is_required_parameter(call_signature, "negative_prompt"):
                kwargs["negative_prompt"] = negative_prompt or ""
        if request.get("guidance_scale") is not None and _accepts_parameter(call_signature, "guidance_scale"):
            kwargs["guidance_scale"] = request["guidance_scale"]
        if request.get("steps") is not None and _accepts_parameter(call_signature, "num_inference_steps"):
            kwargs["num_inference_steps"] = request["steps"]
        if "video_guider_params" in call_signature.parameters:
            kwargs["video_guider_params"] = _multimodal_guider_params(request, modality_scale=3.0)
        if "audio_guider_params" in call_signature.parameters:
            kwargs["audio_guider_params"] = _multimodal_guider_params(request, modality_scale=0.0, cfg_scale=1.0)
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
        with _torch_generation_context(torch):
            result = self.models.pipeline(**kwargs)
        output = persist_result(
            result,
            request["output_path"],
            fps=request["fps"],
            num_frames=request["frames"],
            tiling_config=kwargs.get("tiling_config"),
            include_audio=False,
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


def _accepts_parameter(signature, name):
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _is_required_parameter(signature, name):
    parameter = signature.parameters.get(name)
    return parameter is not None and parameter.default is inspect.Parameter.empty


def _multimodal_guider_params(request, modality_scale, cfg_scale=None):
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(
        cfg_scale=cfg_scale if cfg_scale is not None else float(request.get("guidance_scale") or 3.0),
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=modality_scale,
        skip_step=0,
        stg_blocks=[29],
    )


def _torch_generation_context(torch_module):
    # ltx-pipelines encoders may run autograd-aware ops while consuming returned video tensors.
    # no_grad keeps generation non-training while avoiding inference tensor incompatibilities.
    if hasattr(torch_module, "no_grad"):
        return torch_module.no_grad()
    return torch_module.inference_mode()
