#!/usr/bin/env python3
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from ltx_worker.errors import WorkerError, error_payload
from ltx_worker.generate import Generator
from ltx_worker.memory import torch_health, memory_stats


class JsonlWorker:
    def __init__(self):
        self.generator = Generator(self.emit)

    def emit(self, request_id, response_type, payload):
        sys.stdout.write(json.dumps({"id": request_id, "type": response_type, "payload": payload}, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def handle(self, request):
        request_id = request.get("id", "")
        cmd = request.get("cmd")
        payload = request.get("payload") or {}
        try:
            if cmd == "health":
                self.emit(request_id, "result", {"torch": torch_health(), "memory": memory_stats()})
            elif cmd == "load_model":
                self.generator.load_model(request_id, payload.get("model") or {}, payload.get("profile"))
                self.emit(request_id, "result", {"loaded": True, "memory": memory_stats()})
            elif cmd == "generate":
                result = self.generator.generate(request_id, payload)
                self.emit(request_id, "result", result)
            elif cmd == "unload_model":
                self.generator.unload_model()
                self.emit(request_id, "result", {"unloaded": True, "memory": memory_stats()})
            elif cmd == "memory_stats":
                self.emit(request_id, "stats", memory_stats())
            elif cmd == "cancel_current":
                self.emit(request_id, "result", {"cancelled": False, "message": "cancel is cooperative and no job was active"})
            elif cmd == "shutdown":
                self.generator.unload_model()
                self.emit(request_id, "result", {"shutdown": True})
                return False
            else:
                raise WorkerError("validation_error", f"unknown command: {cmd}")
        except WorkerError as exc:
            self.emit(request_id, "error", error_payload(exc))
        except RuntimeError as exc:
            text = str(exc)
            if "out of memory" in text.lower() or "cuda oom" in text.lower():
                self.emit(request_id, "error", {
                    "code": "cuda_oom",
                    "message": text,
                    "suggested_profile": "colab_tiny",
                    "suggested_width": 512,
                    "suggested_height": 512,
                    "suggested_frames": 33,
                    "suggested_model": "ltx2_3_dev_fp8_distilled_lora",
                })
            else:
                self.emit(request_id, "error", {"code": "worker_error", "message": text})
        except Exception as exc:
            self.emit(request_id, "error", {"code": "worker_error", "message": str(exc)})
        return True


def main():
    worker = JsonlWorker()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"id": "", "type": "error", "payload": {"code": "protocol_error", "message": str(exc)}}) + "\n")
            sys.stdout.flush()
            continue
        if not worker.handle(request):
            break


if __name__ == "__main__":
    main()
