#!/usr/bin/env python3
import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path


class _Server:
    client_id = None
    last_node_id = None

    def send_sync(self, *_args, **_kwargs):
        return None

    def queue_updated(self):
        return None


def main():
    payload = json.load(sys.stdin)
    comfy_root = Path(payload["comfy_root"])
    sys.path.insert(0, str(comfy_root))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import folder_paths
    import nodes
    import execution

    output_path = Path(payload["request"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folder_paths.set_output_directory(str(output_path.parent))

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True))
    prompt = _build_prompt(payload)
    prompt_id = str(uuid.uuid4())
    executor = execution.PromptExecutor(_Server(), cache_args={"ram": 0})
    executor.execute(prompt, prompt_id, extra_data={"client_id": prompt_id}, execute_outputs=["15"])
    if not executor.success:
        print(json.dumps({
            "status": "error",
            "code": "comfy_execution_error",
            "message": "ComfyUI prompt execution failed",
            "details": getattr(executor, "history_result", {}),
            "events": getattr(executor, "status_messages", [])[-10:],
        }))
        return 1
    produced = _find_latest_video(output_path.parent)
    if produced is None:
        print(json.dumps({
            "status": "error",
            "code": "comfy_output_missing",
            "message": "ComfyUI did not write a video file",
        }))
        return 1
    if produced.resolve() != output_path.resolve():
        shutil.copyfile(produced, output_path)
    print(json.dumps({"status": "success", "output_path": str(output_path)}))
    return 0


def _build_prompt(payload):
    request = payload["request"]
    model = payload["model"]
    checkpoint = Path(model["checkpoint_path"]).name
    text_encoder = Path(model["text_encoder_path"]).name
    lora = Path(model["lora_path"]).name if model.get("lora_path") else None
    fps = float(request["fps"])
    steps = int(request.get("steps") or 8)
    cfg = float(request.get("guidance_scale") or 1.0)
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LTXVGemmaCLIPModelLoader", "inputs": {"gemma_path": text_encoder, "ltxv_path": checkpoint, "max_length": 1024}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request["prompt"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request.get("negative_prompt") or ""}},
        "5": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": fps}},
        "6": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": int(request["width"]), "height": int(request["height"]), "length": int(request["frames"]), "batch_size": 1}},
        "7": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": 2.05, "base_shift": 0.95, "latent": ["6", 0]}},
        "8": {"class_type": "LTXVScheduler", "inputs": {"steps": steps, "max_shift": 2.05, "base_shift": 0.95, "stretch": True, "terminal": 0.1, "latent": ["6", 0]}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "10": {"class_type": "CFGGuider", "inputs": {"model": ["7", 0], "positive": ["5", 0], "negative": ["5", 1], "cfg": cfg}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(request["seed"])}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["10", 0], "sampler": ["9", 0], "sigmas": ["8", 0], "latent_image": ["6", 0]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "fps": fps}},
        "15": {"class_type": "SaveVideo", "inputs": {"video": ["14", 0], "filename_prefix": Path(request["output_path"]).stem, "format": "auto", "codec": "auto"}},
    }
    if lora:
        prompt["16"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1.0}}
        prompt["7"]["inputs"]["model"] = ["16", 0]
    return prompt


def _find_latest_video(output_dir):
    candidates = []
    for pattern in ("*.mp4", "*.webm", "*.mov", "**/*.mp4", "**/*.webm", "**/*.mov"):
        candidates.extend(output_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())
