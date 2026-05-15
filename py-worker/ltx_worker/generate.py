from .memory import memory_stats
from .model_manager import ModelManager
from .output import persist_result
from .validators import validate_request
import inspect


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
            from PIL import Image
            image = Image.open(request["input_image"]).convert("RGB")
            if "image" in call_signature.parameters:
                kwargs["image"] = image
            elif "images" in call_signature.parameters:
                kwargs["images"] = [(image, 0)]
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
