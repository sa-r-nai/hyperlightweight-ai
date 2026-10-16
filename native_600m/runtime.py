"""Cloud-only guards for model and tokenizer training."""

from pathlib import Path


def require_studio() -> None:
    if not Path("/teamspace/studios").is_dir():
        raise RuntimeError("Training requires Lightning AI Studio. Local training is disabled.")


def require_studio_cuda():
    require_studio()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Attach a CUDA GPU to the Studio before model training.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Use one visible GPU; set CUDA_VISIBLE_DEVICES=0 for this preset.")
    return torch
