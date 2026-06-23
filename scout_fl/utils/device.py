"""Device selection for GPU-accelerated runs.

``resolve_device("auto")`` picks the best available backend:
  NVIDIA CUDA  ->  "cuda"   (Linux / Windows GPU servers)
  Apple MPS    ->  "mps"    (Apple-silicon macOS)
  otherwise    ->  "cpu"

An explicit spec ("cuda", "cuda:1", "mps", "cpu") is passed through unchanged.
The FL modules only ever call ``tensor.to(device)`` / ``model.to(device)`` and
move results back with ``.cpu().numpy()``, so a resolved string is all they need.
"""
from __future__ import annotations

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(spec: str | None = "auto") -> str:
    """Map a config device spec to a concrete torch device string."""
    spec = (spec or "auto").strip().lower()
    if spec != "auto":
        return spec
    if cuda_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable one-liner for logging which accelerator a run uses."""
    if device.startswith("cuda") and cuda_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        return f"CUDA GPU: {torch.cuda.get_device_name(idx)}"
    if device == "mps":
        return "Apple-silicon GPU (Metal Performance Shaders)"
    return "CPU"
