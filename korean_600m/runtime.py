"""Enforce the cloud-only training policy before any model download."""

from pathlib import Path


def require_studio() -> None:
    if not Path("/teamspace/studios").is_dir():
        raise RuntimeError(
            "Training is restricted to Lightning AI Studio. Open this repository "
            "inside a GPU Studio under /teamspace/studios. Local training is disabled."
        )


def require_studio_cuda():
    require_studio()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Attach a CUDA GPU to the Lightning AI Studio before training.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This preset requires one visible GPU (CUDA_VISIBLE_DEVICES=0).")
    return torch
