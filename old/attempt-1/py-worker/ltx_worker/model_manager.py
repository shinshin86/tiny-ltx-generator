import importlib
import inspect
import os
from .errors import WorkerError
from .memory import cleanup_cuda, memory_stats
from .profiles import preferred_pipeline_names
from .comfy_backend import ComfyLtxPipeline, is_comfy_backend


class ModelManager:
    def __init__(self, emit):
        self.emit = emit
        self.pipeline = None
        self.loaded_key = None
        self.load_warnings = []

    def load(self, request_id, model_entry, profile):
        model_key = model_entry.get("id") or model_entry.get("display_name") or "configured_model"
        if self.pipeline is not None and self.loaded_key == (model_key, profile):
            return
        self.unload()
        self.emit(request_id, "progress", {"stage": "before_load", "memory": memory_stats()})
        if is_comfy_backend(model_entry):
            self.pipeline = ComfyLtxPipeline(model_entry, profile, self.emit)
            self.loaded_key = (model_key, profile)
            self.emit(request_id, "progress", {"stage": "after_load", "backend": "comfy_ltx", "memory": memory_stats()})
            return
        pipeline_cls = self._find_pipeline(model_key, profile)
        kwargs = self._quantization_kwargs(model_entry)
        offload_mode = self._offload_mode(profile)
        for warning in self.load_warnings:
            self.emit(request_id, "log", warning)
        self.load_warnings = []
        if model_entry.get("config_path") and hasattr(pipeline_cls, "from_config"):
            self.pipeline = self._load_from_config(pipeline_cls, model_entry, kwargs, offload_mode)
        elif pipeline_cls.__name__ == "DistilledPipeline":
            if not _is_distilled_model(model_key, model_entry):
                raise WorkerError(
                    "unsupported_pipeline",
                    "DistilledPipeline requires a distilled model; provide a config_path for non-distilled variants",
                )
            self.pipeline = self._load_distilled_pipeline(pipeline_cls, model_entry, kwargs, offload_mode)
        elif pipeline_cls.__name__ == "TI2VidTwoStagesPipeline":
            self.pipeline = self._load_two_stage_pipeline(pipeline_cls, model_entry, kwargs)
        elif pipeline_cls.__name__ == "TI2VidOneStagePipeline":
            self.pipeline = self._load_one_stage_pipeline(pipeline_cls, model_entry, kwargs)
        elif hasattr(pipeline_cls, "from_config"):
            raise WorkerError("unsupported_pipeline", "this pipeline requires config_path in the model registry")
        else:
            raise WorkerError("unsupported_pipeline", f"{pipeline_cls.__name__} loading path is not supported yet")
        self._to_cuda_if_possible()
        self.loaded_key = (model_key, profile)
        self.emit(request_id, "progress", {"stage": "after_load", "memory": memory_stats()})

    def unload(self):
        self.pipeline = None
        self.loaded_key = None
        cleanup_cuda()

    def _find_pipeline(self, model_key, profile):
        errors = []
        for module_name, class_name in preferred_pipeline_names(model_key, profile):
            try:
                module = importlib.import_module(module_name)
                pipeline_cls = getattr(module, class_name)
                return pipeline_cls
            except Exception as exc:
                errors.append(f"{module_name}.{class_name}: {exc}")
        raise WorkerError("unsupported_pipeline", "no supported ltx-pipelines pipeline found", attempts=errors)

    def _quantization_kwargs(self, model_entry):
        mode = model_entry.get("quantization")
        if not mode or mode == "none":
            return {}
        checkpoint_path = model_entry.get("checkpoint_path") or ""
        if mode == "fp8-cast" and _looks_like_fp8_checkpoint(checkpoint_path):
            raise WorkerError(
                "unsupported_option",
                "fp8-cast must not be applied to FP8 checkpoint files; load FP8 checkpoints with quantization='none'",
            )
        if mode == "fp8-cast" and not model_entry.get("supports_fp8_cast"):
            raise WorkerError("unsupported_option", "model registry says fp8-cast is unsupported")
        if mode == "fp8-scaled-mm" and not model_entry.get("supports_fp8_scaled_mm"):
            raise WorkerError("unsupported_option", "model registry says fp8-scaled-mm is unsupported")
        try:
            from ltx_core.quantization.policy import QuantizationPolicy
        except Exception as exc:
            raise WorkerError("unsupported_option", f"quantization requested but QuantizationPolicy is unavailable: {exc}")
        if mode == "fp8-cast":
            return {"quantization": QuantizationPolicy.fp8_cast()}
        if mode == "fp8-scaled-mm":
            return {"quantization": QuantizationPolicy.fp8_scaled_mm()}
        raise WorkerError("unsupported_option", f"unsupported quantization mode: {mode}")

    def _load_from_config(self, pipeline_cls, model_entry, kwargs, offload_mode):
        config_kwargs = dict(kwargs)
        signature = inspect.signature(pipeline_cls.from_config)
        if offload_mode is not None and "offload_mode" in signature.parameters:
            config_kwargs["offload_mode"] = offload_mode
        return pipeline_cls.from_config(model_entry["config_path"], **config_kwargs)

    def _load_distilled_pipeline(self, pipeline_cls, model_entry, kwargs, offload_mode):
        checkpoint_path = model_entry.get("checkpoint_path")
        gemma_root = model_entry.get("gemma_root")
        spatial_upsampler_path = model_entry.get("spatial_upsampler_path")
        missing = [
            name
            for name, value in [
                ("checkpoint_path", checkpoint_path),
                ("gemma_root", gemma_root),
                ("spatial_upsampler_path", spatial_upsampler_path),
            ]
            if not value
        ]
        if missing:
            raise WorkerError("missing_model_files", f"DistilledPipeline requires: {', '.join(missing)}")
        signature = inspect.signature(pipeline_cls)
        init_kwargs = {
            "distilled_checkpoint_path": checkpoint_path,
            "gemma_root": gemma_root,
            "spatial_upsampler_path": spatial_upsampler_path,
            "loras": _lora_specs(model_entry),
        }
        if offload_mode is not None:
            init_kwargs["offload_mode"] = offload_mode
        init_kwargs.update(kwargs)
        filtered = {key: value for key, value in init_kwargs.items() if key in signature.parameters}
        return pipeline_cls(**filtered)

    def _load_one_stage_pipeline(self, pipeline_cls, model_entry, kwargs):
        checkpoint_path = model_entry.get("checkpoint_path")
        gemma_root = model_entry.get("gemma_root")
        missing = [
            name
            for name, value in [
                ("checkpoint_path", checkpoint_path),
                ("gemma_root", gemma_root),
            ]
            if not value
        ]
        if missing:
            raise WorkerError("missing_model_files", f"TI2VidOneStagePipeline requires: {', '.join(missing)}")
        signature = inspect.signature(pipeline_cls)
        init_kwargs = {
            "checkpoint_path": checkpoint_path,
            "gemma_root": gemma_root,
            "loras": _lora_specs(model_entry),
        }
        init_kwargs.update(kwargs)
        filtered = {key: value for key, value in init_kwargs.items() if key in signature.parameters}
        return pipeline_cls(**filtered)

    def _load_two_stage_pipeline(self, pipeline_cls, model_entry, kwargs):
        checkpoint_path = model_entry.get("checkpoint_path")
        gemma_root = model_entry.get("gemma_root")
        spatial_upsampler_path = model_entry.get("spatial_upsampler_path")
        missing = [
            name
            for name, value in [
                ("checkpoint_path", checkpoint_path),
                ("gemma_root", gemma_root),
                ("spatial_upsampler_path", spatial_upsampler_path),
            ]
            if not value
        ]
        if missing:
            raise WorkerError("missing_model_files", f"TI2VidTwoStagesPipeline requires: {', '.join(missing)}")
        signature = inspect.signature(pipeline_cls)
        init_kwargs = {
            "checkpoint_path": checkpoint_path,
            "gemma_root": gemma_root,
            "spatial_upsampler_path": spatial_upsampler_path,
            "distilled_lora": _lora_specs(model_entry),
            "loras": (),
        }
        init_kwargs.update(kwargs)
        filtered = {key: value for key, value in init_kwargs.items() if key in signature.parameters}
        return pipeline_cls(**filtered)

    def _offload_mode(self, profile):
        requested = os.environ.get("LTX_OFFLOAD_MODE")
        if not requested and profile in {"colab_tiny", "colab_eco"}:
            requested = "disk"
        if not requested or requested == "none":
            return None
        try:
            from ltx_pipelines.utils.types import OffloadMode
        except Exception as exc:
            self.load_warnings.append({
                "stage": "offload_unavailable",
                "requested": requested,
                "reason": f"OffloadMode is unavailable: {exc}",
            })
            return None
        try:
            return OffloadMode(requested)
        except ValueError:
            raise WorkerError("unsupported_option", f"unsupported offload mode: {requested}")

    def _to_cuda_if_possible(self):
        try:
            import torch
            if torch.cuda.is_available() and hasattr(self.pipeline, "to"):
                self.pipeline.to("cuda")
        except Exception:
            pass


def _is_distilled_model(model_key, model_entry):
    values = [
        model_key or "",
        model_entry.get("display_name") or "",
        model_entry.get("checkpoint_path") or "",
    ]
    return any("distilled" in value.lower() or "distil" in value.lower() for value in values)


def _looks_like_fp8_checkpoint(path):
    name = os.path.basename(path).lower()
    return "fp8" in name


def _lora_specs(model_entry):
    lora_path = model_entry.get("lora_path")
    if not lora_path:
        return ()
    strength = float(model_entry.get("lora_strength") or 1.0)
    try:
        from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
    except Exception:
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP

    return (LoraPathStrengthAndSDOps(lora_path, strength, LTXV_LORA_COMFY_RENAMING_MAP),)
