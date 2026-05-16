import json
import os
import shutil
import subprocess
from pathlib import Path

from .errors import WorkerError


class ComfyLtxPipeline:
    uses_comfy_backend = True

    def __init__(self, model_entry, profile, emit=None):
        self.model_entry = model_entry
        self.profile = profile
        self.emit = emit or (lambda *_args: None)
        self.comfy_root = Path(
            model_entry.get("extra", {}).get("comfy_root")
            or os.environ.get("LTX_COMFYUI_ROOT")
            or "/content/ComfyUI"
        )
        self.runner = Path(
            model_entry.get("extra", {}).get("comfy_runner")
            or os.environ.get("LTX_COMFY_RUNNER")
            or Path(__file__).resolve().parents[2] / "scripts" / "comfy_headless_ltx.py"
        )

    def __call__(self, **kwargs):
        if kwargs.get("mode") == "image-to-video":
            raise WorkerError(
                "unsupported_pipeline",
                "Comfy LTX image-to-video backend is not implemented yet; use text-to-video for this backend",
            )
        request = {
            "mode": kwargs.get("mode", "text-to-video"),
            "prompt": kwargs["prompt"],
            "negative_prompt": kwargs.get("negative_prompt") or "",
            "input_image": kwargs.get("input_image"),
            "width": kwargs["width"],
            "height": kwargs["height"],
            "frames": kwargs["num_frames"],
            "fps": kwargs.get("fps") or kwargs.get("frame_rate") or 12,
            "steps": kwargs.get("num_inference_steps") or _default_steps(self.profile),
            "seed": kwargs["seed"],
            "guidance_scale": kwargs.get("guidance_scale") or _default_cfg(self.profile),
            "output_path": kwargs["output_path"],
            "profile": self.profile,
        }
        payload = {
            "comfy_root": str(self.comfy_root),
            "model": self._comfy_model_paths(),
            "request": request,
        }
        self._ensure_ready(payload)
        command = [os.environ.get("PYTHON", "python3"), str(self.runner)]
        proc = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.comfy_root),
        )
        if proc.returncode != 0:
            raise WorkerError(
                "comfy_backend_error",
                "ComfyUI headless backend failed",
                returncode=proc.returncode,
                stderr=proc.stderr[-4000:],
                stdout=proc.stdout[-4000:],
            )
        try:
            response = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            raise WorkerError(
                "comfy_backend_error",
                f"ComfyUI headless backend returned invalid JSON: {exc}",
                stdout=proc.stdout[-4000:],
                stderr=proc.stderr[-4000:],
            )
        if response.get("status") != "success":
            raise WorkerError(
                response.get("code") or "comfy_backend_error",
                response.get("message") or "ComfyUI headless backend failed",
                details=response,
            )
        output = response.get("output_path")
        if not output or not Path(output).exists():
            raise WorkerError("comfy_backend_error", "ComfyUI backend did not produce an output file")
        return output

    def _ensure_ready(self, payload):
        missing = []
        if not self.comfy_root.exists():
            missing.append(f"LTX_COMFYUI_ROOT={self.comfy_root}")
        if not self.runner.exists():
            missing.append(f"comfy runner={self.runner}")
        for key, path in payload["model"].items():
            if path and not Path(path).exists():
                missing.append(f"{key}={path}")
        if missing:
            raise WorkerError("missing_model_files", "Comfy backend requires existing paths", missing=missing)
        self._link_model_files(payload["model"])

    def _link_model_files(self, paths):
        mapping = {
            "checkpoint_path": "checkpoints",
            "lora_path": "loras",
            "text_encoder_path": "text_encoders",
            "spatial_upsampler_path": "latent_upscale_models",
        }
        for key, folder in mapping.items():
            source = paths.get(key)
            if not source:
                continue
            source_path = Path(source)
            dest_dir = self.comfy_root / "models" / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / source_path.name
            if dest.exists():
                continue
            try:
                dest.symlink_to(source_path)
            except OSError:
                shutil.copyfile(source_path, dest)

    def _comfy_model_paths(self):
        text_encoder_path = self.model_entry.get("text_encoder_path")
        if not text_encoder_path:
            raise WorkerError(
                "missing_model_files",
                "Comfy LTX backend requires text_encoder_path; use a split Gemma safetensors file",
            )
        return {
            "checkpoint_path": self.model_entry.get("checkpoint_path") or "",
            "lora_path": self.model_entry.get("lora_path") or "",
            "lora_strength": self.model_entry.get("lora_strength") or 0.5,
            "text_encoder_path": text_encoder_path,
            "spatial_upsampler_path": self.model_entry.get("spatial_upsampler_path") or "",
        }


def is_comfy_backend(model_entry):
    extra = model_entry.get("extra") or {}
    return extra.get("backend") in {"comfy_ltx", "comfy_ltx_headless"}


def _default_steps(profile):
    if profile in {"colab_tiny", "colab_eco"}:
        return 8
    return 12


def _default_cfg(profile):
    if profile in {"colab_tiny", "colab_eco"}:
        return 1.0
    return 2.0
