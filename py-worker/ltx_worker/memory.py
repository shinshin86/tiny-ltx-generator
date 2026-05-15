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
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "import_ok": True,
        "version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": device_count,
        "gpu_name": torch.cuda.get_device_name(0) if device_count else None,
        "allocated_vram_mb": int(torch.cuda.memory_allocated(0) / 1024 / 1024) if device_count else None,
        "reserved_vram_mb": int(torch.cuda.memory_reserved(0) / 1024 / 1024) if device_count else None,
        "error": None,
    }


def memory_stats():
    try:
        import torch
    except Exception as exc:
        return {"torch_import_ok": False, "error": str(exc)}
    stats = {"torch_import_ok": True, "cuda_available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        stats.update({
            "allocated_vram_mb": int(torch.cuda.memory_allocated(0) / 1024 / 1024),
            "reserved_vram_mb": int(torch.cuda.memory_reserved(0) / 1024 / 1024),
            "max_allocated_vram_mb": int(torch.cuda.max_memory_allocated(0) / 1024 / 1024),
            "max_reserved_vram_mb": int(torch.cuda.max_memory_reserved(0) / 1024 / 1024),
        })
    return stats


def cleanup_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
