from collections.abc import Mapping

import torch

from .model import InferenceModel


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or install a CUDA PyTorch build")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("The selected CUDA device index is unavailable")
    return device


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a state dictionary")
    if "state_dict" in checkpoint:
        if checkpoint.get("smoke_test", False):
            raise ValueError("Smoke-test weights cannot be used for inference")
        if checkpoint.get("is_initialization", False) or checkpoint.get("untrained", False):
            raise ValueError("Initialization weights cannot be used for inference")
        for key in ("role", "kind", "weights"):
            marker = checkpoint.get(key)
            if isinstance(marker, str) and any(
                label in marker.lower() for label in ("initialization", "untrained", "smoke")
            ):
                raise ValueError("Checkpoint is marked as initialization or smoke-test weights")
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Checkpoint state_dict must be a nonempty mapping")
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state_dict.items()):
        raise ValueError("Checkpoint state_dict must map parameter names to tensors")
    model = InferenceModel()
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
