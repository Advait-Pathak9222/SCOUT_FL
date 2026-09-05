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


def tune_backend(device: str, *, deterministic: bool = True,
                 memory_fraction: float | None = None) -> dict:
    """Configure the backend for throughput, and report what was set.

    The campaign is many short kernels on a small model, so the wins are in launch
    overhead and in matmul precision rather than in anything algorithmic.

    ``deterministic`` keeps the run bit reproducible, which costs the cuDNN autotuner
    and some fast kernels. Set it False for exploratory sweeps and True for the runs
    whose numbers reach the paper.

    ``memory_fraction`` caps this process's share of the card. Several campaign shards
    usually share one GPU, and a cap turns a would-be out of memory crash in one shard
    into an allocation failure the fast path can fall back from.
    """
    info: dict = {"device": device, "deterministic": deterministic}
    if device.startswith("cuda") and cuda_available():
        # TF32 costs a few mantissa bits on convolution and matmul and buys a large
        # speedup on Ampere and later. The quantities that matter here are test accuracy
        # and an aggregation error of order 1e-3, both far above that noise floor.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        # Fixed shapes every round, so the autotuner pays for itself after one round.
        torch.backends.cudnn.benchmark = not deterministic
        info["tf32"] = True
        info["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        if memory_fraction is not None:
            idx = 0
            if ":" in device:
                try:
                    idx = int(device.split(":", 1)[1])
                except ValueError:
                    idx = 0
            try:
                torch.cuda.set_per_process_memory_fraction(float(memory_fraction), idx)
                info["memory_fraction"] = float(memory_fraction)
            except Exception:
                pass
        info["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass
    return info


def free_memory_bytes(device: str) -> float | None:
    """Free memory on the device, or None when the notion does not apply."""
    if device.startswith("cuda") and cuda_available():
        free, _total = torch.cuda.mem_get_info()
        return float(free)
    return None


def release(device: str) -> None:
    """Return cached blocks to the allocator between units, to limit fragmentation."""
    if device.startswith("cuda") and cuda_available():
        torch.cuda.empty_cache()
    elif device == "mps" and mps_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
