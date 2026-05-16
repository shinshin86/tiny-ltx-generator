import subprocess


def _cuda_module(torch):
    return getattr(torch, "cuda", None)


def _cuda_available(torch):
    cuda = _cuda_module(torch)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    return bool(is_available())


def torch_health():
    try:
        import torch
    except Exception as exc:
        return {
            "import_ok": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "gpu_name": None,
            "allocated_vram_mb": None,
            "reserved_vram_mb": None,
            "error": str(exc),
        }
    cuda = _cuda_module(torch)
    cuda_available = _cuda_available(torch)
    device_count_fn = getattr(cuda, "device_count", None)
    device_count = (
        int(device_count_fn()) if cuda_available and callable(device_count_fn) else 0
    )
    torch_version = getattr(torch, "version", None)
    return {
        "import_ok": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch_version, "cuda", None),
        "device_count": device_count,
        "gpu_name": cuda.get_device_name(0) if device_count else None,
        "allocated_vram_mb": int(cuda.memory_allocated(0) / 1024 / 1024)
        if device_count
        else None,
        "reserved_vram_mb": int(cuda.memory_reserved(0) / 1024 / 1024)
        if device_count
        else None,
        "error": None,
    }


def memory_stats():
    try:
        import torch
    except Exception as exc:
        return {"torch_import_ok": False, "error": str(exc)}
    cuda = _cuda_module(torch)
    cuda_available = _cuda_available(torch)
    stats = {"torch_import_ok": True, "cuda_available": cuda_available}
    if cuda_available:
        stats.update({
            "allocated_vram_mb": int(cuda.memory_allocated(0) / 1024 / 1024),
            "reserved_vram_mb": int(cuda.memory_reserved(0) / 1024 / 1024),
            "max_allocated_vram_mb": int(
                cuda.max_memory_allocated(0) / 1024 / 1024
            ),
            "max_reserved_vram_mb": int(cuda.max_memory_reserved(0) / 1024 / 1024),
        })
    stats.update(_nvidia_smi_memory())
    return stats


def _nvidia_smi_memory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return {}
    if result.returncode != 0:
        return {}
    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return {}
    try:
        used, free, total = [int(part) for part in parts]
    except ValueError:
        return {}
    return {
        "nvidia_smi_used_vram_mb": used,
        "nvidia_smi_free_vram_mb": free,
        "nvidia_smi_total_vram_mb": total,
    }


def cleanup_cuda():
    import gc
    gc.collect()
    try:
        import torch
        cuda = _cuda_module(torch)
        empty_cache = getattr(cuda, "empty_cache", None)
        if _cuda_available(torch) and callable(empty_cache):
            empty_cache()
    except Exception:
        pass
