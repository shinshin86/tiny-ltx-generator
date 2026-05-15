import shutil
from pathlib import Path

from .errors import WorkerError


def persist_result(result, output_path, fps=12, num_frames=None, tiling_config=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(result, (str, Path)) and Path(result).exists():
        shutil.copyfile(result, output)
        return str(output)
    if hasattr(result, "save"):
        result.save(str(output))
        return str(output)
    if isinstance(result, tuple) and len(result) == 2:
        video, audio = result
        try:
            from ltx_core.model.video_vae import get_video_chunks_number
            from ltx_pipelines.utils.media_io import encode_video
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config) if num_frames and tiling_config else None
            kwargs = {"video": video, "fps": fps, "audio": audio, "output_path": str(output)}
            if video_chunks_number is not None:
                kwargs["video_chunks_number"] = video_chunks_number
            encode_video(**kwargs)
            return str(output)
        except Exception as exc:
            raise WorkerError("unsupported_pipeline", f"could not encode ltx-pipelines tuple result: {exc}")
    for attr in ("video_path", "output_path", "path"):
        value = getattr(result, attr, None)
        if value and Path(value).exists():
            shutil.copyfile(value, output)
            return str(output)
    video = None
    for attr in ("video", "videos", "frames"):
        candidate = getattr(result, attr, None)
        if candidate is not None:
            video = candidate
            break
    if video is not None:
        _write_video(video, output, fps)
        return str(output)
    raise WorkerError("unsupported_pipeline", "pipeline returned an unsupported result type; cannot persist video")


def _write_video(video, output, fps):
    import imageio.v3 as iio
    try:
        import torch
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu()
            if video.ndim == 5:
                video = video[0]
            video = video.numpy()
    except Exception:
        pass
    iio.imwrite(output, video, fps=fps)
