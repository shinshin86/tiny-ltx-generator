#!/usr/bin/env python3
try:
    import torch
    print({"cuda": torch.cuda.is_available(), "device_count": torch.cuda.device_count()})
    if torch.cuda.is_available():
        print({"name": torch.cuda.get_device_name(0), "allocated": torch.cuda.memory_allocated(0), "reserved": torch.cuda.memory_reserved(0)})
except Exception as exc:
    print({"error": str(exc)})
