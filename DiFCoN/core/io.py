from __future__ import annotations

import json
import platform
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def update_run_status(output_dir: str | Path, state: str, **details: Any) -> Path:
    payload = {"state": str(state), "updated_at_utc": utc_now(), **details}
    return write_json(Path(output_dir) / "run_status.json", payload)


def collect_environment(device: str) -> dict[str, Any]:
    import torch

    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "created_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gpu_names": gpu_names,
        "requested_device": str(device),
    }


__all__ = [
    "collect_environment",
    "prepare_run_directory",
    "update_run_status",
    "utc_now",
    "write_json",
]
