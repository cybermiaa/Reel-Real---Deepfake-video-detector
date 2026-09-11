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
    
    # Load the pre-trained model directly
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


# --------------------------------------------------------------------------
# Calibration & Checkpoint Wrappers (Required for fit_clip_calibration.py)
# --------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: Path | str, device: str) -> Tuple[nn.Module, dict]:
    """Compatible wrapper for scripts expecting a local checkpoint loader."""
    path = Path(checkpoint_path)
    ckpt_dict = {}
    if path.exists() and path.is_file():
        try:
            ckpt_dict = torch.load(path, map_location=device)
        except Exception:
            pass
    
    # Build and load the Hugging Face model
    model = build_model(pretrained=True)
    model.to(device)
    model.eval()
    
    # Default metadata if none saved
    if "temperature" not in ckpt_dict:
        ckpt_dict["temperature"] = 1.0
        
    return model, ckpt_dict


def update_checkpoint(checkpoint_path: Path | str, clip_calibrator: dict, high_thresh: float, decision_thresh: float) -> None:
    """Saves updated thresholds and calibrator parameters."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {}
    if path.exists():
        try:
            data = torch.load(path, map_location="cpu")
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
            
    data["clip_calibrator"] = clip_calibrator
    data["high_thresh"] = high_thresh
    data["decision_thresh"] = decision_thresh
    
    torch.save(data, path)