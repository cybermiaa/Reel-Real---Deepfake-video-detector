"""Grad-CAM, and turning a heat map into a defensible sentence.

Grad-CAM shows where the model attended. It is not evidence of manipulation,
and the interface says so. What it can honestly support is a sentence like
"attention concentrated on the mouth and jaw", provided that claim is derived
from a measured quantity rather than asserted.

Two things this module gets right that a naive implementation does not:

  * Region naming uses MTCNN's five facial landmarks, mapped into crop
    coordinates by preprocess.crop_face(). A fixed coordinate mapping --
    "the bottom third of the image is the mouth" -- breaks as soon as the face
    is off-centre, rotated, or differently scaled, which is most real footage.

  * The reported quantity is the share of total CAM mass falling near each
    landmark anchor, not the single peak pixel. On a ResNet-18 the final
    feature map is 7x7, so one "pixel" of the CAM covers roughly 32x32 input
    pixels; a single argmax over that is a coarse thing to build a claim on.
    When no region clearly dominates, region_report() returns None and the
    evidence panel omits the sentence rather than inventing one.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Landmark order returned by MTCNN.
LM_LEFT_EYE, LM_RIGHT_EYE, LM_NOSE, LM_MOUTH_L, LM_MOUTH_R = range(5)

REGION_PHRASES = {
    "eyes": "the eye region, where blending seams and inconsistent gaze often appear",
    "nose and cheeks": "the nose and cheeks, where face-swap blending boundaries often fall",
    "mouth and jaw": "the mouth and jaw area, where lip-sync manipulation usually leaves traces",
}


class GradCAM:
    """Minimal Grad-CAM. No external dependency."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = 1,
                 out_size: Optional[int] = None) -> np.ndarray:
        """x: (1,3,H,W). Returns a HxW map normalised to [0,1].

        Runs with gradients enabled -- this cannot live inside a no_grad block,
        which is why scoring and explanation are separate passes.
        """
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            logits[:, class_idx].sum().backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((out_size or x.shape[-1],) * 2, dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        size = out_size or x.shape[-1]
        cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        if was_training:
            self.model.train()

        lo, hi = float(cam.min()), float(cam.max())
        return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def region_report(cam: np.ndarray, landmarks: np.ndarray,
                  dominance: float = 0.40) -> Optional[Dict]:
    """Which facial region the attention map concentrates on.

    Returns None when no region reaches `dominance` share of CAM mass, so the
    interface can stay silent rather than over-claim.
    """
    if cam is None or landmarks is None or len(landmarks) < 5:
        return None
    total = float(cam.sum())
    if total <= 0:
        return None

    lm = np.asarray(landmarks, dtype=np.float32)
    eyes = (lm[LM_LEFT_EYE] + lm[LM_RIGHT_EYE]) / 2.0
    mouth = (lm[LM_MOUTH_L] + lm[LM_MOUTH_R]) / 2.0
    nose = lm[LM_NOSE]
    # Jaw sits below the mouth by roughly half the nose-to-mouth distance.
    jaw = mouth + (mouth - nose) * 0.5

    iod = float(np.linalg.norm(lm[LM_RIGHT_EYE] - lm[LM_LEFT_EYE]))
    if not np.isfinite(iod) or iod <= 1:
        return None
    radius = 0.9 * iod

    h, w = cam.shape
    yy, xx = np.mgrid[0:h, 0:w]

    def mass_near(points) -> float:
        near = np.zeros((h, w), dtype=bool)
        for px, py in points:
            near |= ((xx - px) ** 2 + (yy - py) ** 2) <= radius ** 2
        return float(cam[near].sum()) / total

    shares = {
        "eyes": mass_near([eyes]),
        "nose and cheeks": mass_near([nose]),
        "mouth and jaw": mass_near([mouth, jaw]),
    }
    best = max(shares, key=shares.get)
    if shares[best] < dominance:
        return {"region": None, "shares": shares, "peak_xy": None,
                "reason": f"no region reached {dominance:.0%} of attention mass"}

    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {
        "region": best,
        "phrase": REGION_PHRASES[best],
        "share": shares[best],
        "shares": shares,
        "peak_xy": (int(peak[1]), int(peak[0])),
        "interocular_px": iod,
    }


def overlay(crop_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Heat map over the crop, for the evidence panel."""
    import matplotlib
    matplotlib.use("Agg")
    # matplotlib.cm.get_cmap was removed in 3.9; this form works from 3.5 on.
    from matplotlib import colormaps

    base = np.asarray(crop_rgb, dtype=np.float32) / 255.0
    heat = colormaps["inferno"](np.clip(cam, 0, 1))[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1).astype(np.float32)
