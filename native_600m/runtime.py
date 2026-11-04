"""CUDA requirements for single-GPU model training on any platform."""


def require_cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Model training requires an available CUDA GPU.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Use one visible GPU; set CUDA_VISIBLE_DEVICES=0 for this preset.")
    return torch
