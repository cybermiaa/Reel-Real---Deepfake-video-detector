"""Simulated real-world degradation, at both image and video level.

Two levels, kept separate on purpose. "Low-bitrate re-encoding" is a video
codec operation and cannot be applied to a JPEG crop -- reporting an
image-level JPEG requantisation as though it were a bitrate sweep would be
overclaiming. So:

  IMAGE level (apply_image) runs on the held-out crop test set. Honest label:
      requantisation and rescaling, not codec re-encoding.

  VIDEO level (apply_video) runs ffmpeg over whole clips and is the real
      thing: H.264 re-encode at a target bitrate, 360p downscale, and a
      social-media-style recompression chain. Requires held-out *videos*,
      which is what the clip-level split provides.

Report which level produced each number. They are not interchangeable.
"""
from __future__ import annotations

import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageFilter

# --------------------------------------------------------------------------
# Image level -- for the crop test set
# --------------------------------------------------------------------------
IMAGE_CONDITIONS: Dict[str, dict] = {
    "clean":            {},
    "jpeg_q50":         {"jpeg": 50},
    "jpeg_q30":         {"jpeg": 30},
    "downscale_0.5":    {"scale": 0.5},
    "downscale_0.25":   {"scale": 0.25},
    "blur_1.0":         {"blur": 1.0},
    # Rough stand-in for a re-upload: shrink, recompress, restore size.
    "social_recompress": {"scale": 0.5, "jpeg": 40},
    "heavy":            {"scale": 0.35, "jpeg": 30, "blur": 0.6},
}


def apply_image(img: Image.Image, scale: Optional[float] = None,
                jpeg: Optional[int] = None, blur: Optional[float] = None
                ) -> Image.Image:
    """Degrade a crop, then restore original size so the model input is unchanged."""
    out = img.convert("RGB")
    w, h = out.size
    if scale and scale != 1.0:
        nw, nh = max(16, int(w * scale)), max(16, int(h * scale))
        out = out.resize((nw, nh), Image.BILINEAR)
    if blur:
        out = out.filter(ImageFilter.GaussianBlur(radius=float(blur)))
    if jpeg:
        buf = BytesIO()
        out.save(buf, format="JPEG", quality=int(jpeg))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    if out.size != (w, h):
        out = out.resize((w, h), Image.BILINEAR)
    return out


class DegradeTransform:
    """Wraps a degradation ahead of the eval transform. Picklable."""

    def __init__(self, condition: str, eval_transform):
        if condition not in IMAGE_CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; "
                             f"choose from {list(IMAGE_CONDITIONS)}")
        self.params = IMAGE_CONDITIONS[condition]
        self.condition = condition
        self.eval_transform = eval_transform

    def __call__(self, img):
        return self.eval_transform(apply_image(img, **self.params) if self.params else img)


# --------------------------------------------------------------------------
# Video level -- for the clip test set
# --------------------------------------------------------------------------
VIDEO_CONDITIONS: Dict[str, list] = {
    "clean": [],
    "bitrate_300k": ["-c:v", "libx264", "-b:v", "300k", "-maxrate", "300k",
                     "-bufsize", "600k", "-preset", "veryfast"],
    "bitrate_100k": ["-c:v", "libx264", "-b:v", "100k", "-maxrate", "100k",
                     "-bufsize", "200k", "-preset", "veryfast"],
    "scale_360p": ["-vf", "scale=-2:360", "-c:v", "libx264", "-crf", "28",
                   "-preset", "veryfast"],
    "social": ["-vf", "scale=-2:480", "-c:v", "libx264", "-crf", "32",
               "-preset", "veryfast", "-r", "25"],
}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def apply_video(src, dst, condition: str, overwrite: bool = True) -> Path:
    """Re-encode a clip under a named condition. Returns the output path."""
    if condition not in VIDEO_CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; "
                         f"choose from {list(VIDEO_CONDITIONS)}")
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if condition == "clean":
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        return dst
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH. It is preinstalled on Colab; "
                           "locally, install it or restrict the sweep to image level.")
    if dst.exists() and not overwrite:
        return dst

    cmd = (["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
           + VIDEO_CONDITIONS[condition] + ["-an", str(dst)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src.name} [{condition}]: "
                           f"{proc.stderr.strip()[:400]}")
    return dst
