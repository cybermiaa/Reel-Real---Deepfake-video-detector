"""Model construction and checkpoint I/O for Hugging Face Deep-Fake-Detector."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor

import config

# --------------------------------------------------------------------------
# AMP compatibility (keeping your original robust scaling logic)
# --------------------------------------------------------------------------
try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast

    def make_grad_scaler(device_type: str, enabled: bool):
        return _GradScaler(device_type, enabled=enabled)

    def amp_autocast(device_type: str, enabled: bool):
        return _autocast(device_type, enabled=enabled)

except ImportError:  # torch < 2.4
    from torch.cuda.amp import GradScaler as _CudaScaler, autocast as _cuda_autocast
    from contextlib import nullcontext

    def make_grad_scaler(device_type: str, enabled: bool):
        return _CudaScaler(enabled=enabled and device_type == "cuda")

    def amp_autocast(device_type: str, enabled: bool):
        if device_type != "cuda" or not enabled:
            return nullcontext()
        return _cuda_autocast()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(arch: str = None, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Loads the pre-trained Hugging Face Vision Transformer deepfake detector."""
    model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    
    # Load the pre-trained model directly (ignoring local training weights since we are inference-only)
    model = ViTForImageClassification.from_pretrained(model_id)
    device = get_device()
    return model.to(device)


def get_processor():
    """Returns the Hugging Face image processor required to format face crops for the ViT model."""
    model_id = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    return ViTImageProcessor.from_pretrained(model_id)


def score_face_crop(model: nn.Module, processor, face_crop_pil, device: str = None) -> float:
    """Takes an MTCNN face crop (PIL Image), processes it, and returns the deepfake probability."""
    device = device or get_device()
    model.eval()
    
    # Prepare inputs using the Hugging Face processor
    inputs = processor(images=face_crop_pil, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Convert logits to probabilities (Softmax)
        probs = outputs.logits.softmax(dim=-1)
        
        # Assuming index 1 corresponds to the 'fake' class probability
        fake_prob = probs[0][1].item()
        
    return fake_prob