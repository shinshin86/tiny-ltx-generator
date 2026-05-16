from pathlib import Path
from .errors import WorkerError


def validate_request(request):
    if not request.get("prompt", "").strip():
        raise WorkerError("validation_error", "prompt must not be empty")
    if request["width"] % 32 or request["height"] % 32:
        raise WorkerError("validation_error", "width and height must be divisible by 32")
    if request["frames"] not in {33, 49, 65, 81, 97, 121, 161}:
        raise WorkerError("validation_error", "unsupported frame count")
    if request["mode"] == "image-to-video":
        image = request.get("input_image")
        if not image or not Path(image).exists():
            raise WorkerError("validation_error", "image-to-video input image is missing")
