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

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))
    prompt = _build_prompt(payload)
    save_node_id = "29" if "29" in prompt else "20"
    prompt_id = str(uuid.uuid4())
    executor = execution.PromptExecutor(_Server(), cache_args={"ram": 0})
    executor.execute(prompt, prompt_id, extra_data={"client_id": prompt_id}, execute_outputs=[save_node_id])
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
    if not _same_file(produced, output_path):
        shutil.copyfile(produced, output_path)
    print(json.dumps({"status": "success", "output_path": str(output_path)}))
    return 0


def _build_prompt(payload):
    request = payload["request"]
    model = payload["model"]
    checkpoint = Path(model["checkpoint_path"]).name
    text_encoder = Path(model["text_encoder_path"]).name
    lora = Path(model["lora_path"]).name if model.get("lora_path") else None
    spatial_upsampler = (
        Path(model["spatial_upsampler_path"]).name if model.get("spatial_upsampler_path") else None
    )
    fps = float(request["fps"])
    steps = int(request.get("steps") or 8)
    cfg = float(request.get("guidance_scale") or 1.0)
    lora_strength = float(model.get("lora_strength") or 0.5)
    width = int(request["width"])
    height = int(request["height"])
    frames = int(request["frames"])
    first_width = width // 2 if spatial_upsampler else width
    first_height = height // 2 if spatial_upsampler else height
    first_sigmas = _first_stage_sigmas(steps)
    second_stage_seed = int(model.get("second_stage_seed") or 42)
    prompt = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": text_encoder, "ckpt_name": checkpoint, "device": "default"},
        },
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request["prompt"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": request.get("negative_prompt") or ""}},
        "5": {"class_type": "LTXVConditioning", "inputs": {"positive": ["3", 0], "negative": ["4", 0], "frame_rate": fps}},
        "6": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": first_width, "height": first_height, "length": frames, "batch_size": 1},
        },
        "7": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": checkpoint}},
        "8": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {"audio_vae": ["7", 0], "frames_number": frames, "frame_rate": int(fps), "batch_size": 1},
        },
        "9": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["6", 0], "audio_latent": ["8", 0]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(request["seed"])}},
        "12": {"class_type": "ManualSigmas", "inputs": {"sigmas": first_sigmas}},
        "13": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1], "cfg": cfg}},
        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["11", 0], "guider": ["13", 0], "sampler": ["10", 0], "sigmas": ["12", 0], "latent_image": ["9", 0]},
        },
        "15": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["14", 0]}},
        "16": {"class_type": "LTXVCropGuides", "inputs": {"positive": ["5", 0], "negative": ["5", 1], "latent": ["15", 0]}},
    }
    if lora:
        prompt["17"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": lora_strength},
        }
        prompt["13"]["inputs"]["model"] = ["17", 0]
    model_ref = prompt["13"]["inputs"]["model"]
    if spatial_upsampler:
        prompt.update(
            {
                "18": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": spatial_upsampler}},
                "19": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["15", 0], "upscale_model": ["18", 0], "vae": ["1", 2]}},
                "20": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["19", 0], "audio_latent": ["15", 1]}},
                "21": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_cfg_pp"}},
                "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": second_stage_seed}},
                "23": {"class_type": "ManualSigmas", "inputs": {"sigmas": "0.85, 0.7250, 0.4219, 0.0"}},
                "24": {"class_type": "CFGGuider", "inputs": {"model": model_ref, "positive": ["16", 0], "negative": ["16", 1], "cfg": cfg}},
                "25": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {
                        "noise": ["22", 0],
                        "guider": ["24", 0],
                        "sampler": ["21", 0],
                        "sigmas": ["23", 0],
                        "latent_image": ["20", 0],
                    },
                },
                "26": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["25", 0]}},
                "27": {
                    "class_type": "VAEDecodeTiled",
                    "inputs": {"samples": ["26", 0], "vae": ["1", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4},
                },
                "30": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["26", 1], "audio_vae": ["7", 0]}},
                "28": {"class_type": "CreateVideo", "inputs": {"images": ["27", 0], "fps": fps, "audio": ["30", 0]}},
                "29": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["28", 0],
                        "filename_prefix": Path(request["output_path"]).stem,
                        "format": "auto",
                        "codec": "auto",
                    },
                },
            }
        )
    else:
        prompt.update(
            {
                "18": {
                    "class_type": "VAEDecodeTiled",
                    "inputs": {"samples": ["15", 0], "vae": ["1", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4},
                },
                "21": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["15", 1], "audio_vae": ["7", 0]}},
                "19": {"class_type": "CreateVideo", "inputs": {"images": ["18", 0], "fps": fps, "audio": ["21", 0]}},
                "20": {
                    "class_type": "SaveVideo",
                    "inputs": {
                        "video": ["19", 0],
                        "filename_prefix": Path(request["output_path"]).stem,
                        "format": "auto",
                        "codec": "auto",
                    },
                },
            }
        )
    return prompt


def _first_stage_sigmas(steps):
    if steps == 8:
        return "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
    values = [1.0]
    for idx in range(1, steps):
        t = idx / steps
        values.append(max(0.0, 1.0 - (t * t)))
    values.append(0.0)
    return ", ".join(f"{value:.6g}" for value in values)


def _find_latest_video(output_dir):
    candidates = []
    for pattern in ("*.mp4", "*.webm", "*.mov", "**/*.mp4", "**/*.webm", "**/*.mov"):
        candidates.extend(output_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _same_file(left, right):
    try:
        return Path(left).resolve() == Path(right).resolve() or os.path.samefile(left, right)
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
