"""Face detection, cropping and tensor transforms -- shared by training and inference.

This module exists to guarantee one thing: a crop produced during dataset
extraction and a crop produced during live inference go through byte-identical
code. The previous version of this pipeline detected faces one way for
training and another way for inference, which produced a model that scored
well on held-out crops and garbage on real uploads, with no error raised.

Two specific traps this module avoids:

1. facenet-pytorch's MTCNN.__call__ returns a tensor that has already been
   standardised with (x - 127.5) / 128 when post_process=True (the default).
   Feeding that to an ImageNet-normalised model is silently wrong. We never
   call MTCNN.__call__ -- only MTCNN.detect(), which returns raw boxes.

2. MTCNN(margin=N) is a fixed pixel margin defined against the output image
   size; it does not scale with the detected box. We ignore it and expand the
   box ourselves by config.CROP_SCALE.

All transform classes are module-level and picklable (no lambdas), so
DataLoader workers work under both fork and spawn.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from torchvision import transforms

import config


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
class RandomDownscale:
    """Downscale then restore size, simulating a re-uploaded low-res video.

    Real manipulated media arrives re-encoded and re-uploaded, not pristine.
    A detector trained only on clean crops learns artifacts that recompression
    destroys.
    """

    def __init__(self, scale_range=(0.35, 1.0), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        s = random.uniform(*self.scale_range)
        w, h = img.size
        nw, nh = max(16, int(w * s)), max(16, int(h * s))
        return img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)


class RandomJPEG:
    """Re-encode as JPEG at random quality.

    This is the augmentation the robustness sweep depends on. Without it the
    model overfits to the compression signature of the training extraction and
    the social-media recompression numbers collapse.
    """

    def __init__(self, quality_range=(30, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_train_transform(image_size: int = None):
    s = image_size or config.IMAGE_SIZE
    # Resize first so the degradation augmentations operate at a fixed scale;
    # otherwise their strength depends on the source crop resolution.
    return transforms.Compose([
        transforms.Resize((s, s)),
        transforms.RandomHorizontalFlip(),
        RandomDownscale(),
        RandomJPEG(),
        transforms.ToTensor(),
        transforms.Normalize(list(config.NORM_MEAN), list(config.NORM_STD)),
    ])


def build_eval_transform(image_size: int = None):
    s = image_size or config.IMAGE_SIZE
    return transforms.Compose([
        transforms.Resize((s, s)),
        transforms.ToTensor(),
        transforms.Normalize(list(config.NORM_MEAN), list(config.NORM_STD)),
    ])


# Module-level singletons so training and inference import the same objects.
train_tf = build_train_transform()
eval_tf = build_eval_transform()


# --------------------------------------------------------------------------
# Face detection
# --------------------------------------------------------------------------
@dataclass
class Face:
    box: np.ndarray          # [x1, y1, x2, y2] in source-frame pixels
    prob: float
    landmarks: np.ndarray    # (5, 2): left eye, right eye, nose, mouth L, mouth R

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


@dataclass
class Crop:
    image: Image.Image
    origin: tuple            # (x, y) of the crop's top-left in source frame
    side: int                # crop side length in source-frame pixels
    landmarks: np.ndarray    # (5, 2) mapped into crop pixel coordinates
    face: Face


_detector = None


def get_detector(device: str = "cpu"):
    """Lazily build a single shared MTCNN.

    keep_all=True because we do our own face selection; MTCNN's built-in
    selection picks the highest-probability box, not the largest, which is
    wrong for multi-person footage.
    """
    global _detector
    if _detector is None:
        from facenet_pytorch import MTCNN
        _detector = MTCNN(keep_all=True, post_process=False, device=device)
    return _detector


def detect_faces(frame_rgb: np.ndarray, detector, min_prob: float = None) -> List[Face]:
    min_prob = config.MIN_FACE_PROB if min_prob is None else min_prob
    boxes, probs, landmarks = detector.detect(Image.fromarray(frame_rgb), landmarks=True)
    if boxes is None:
        return []
    faces = []
    for b, p, lm in zip(boxes, probs, landmarks):
        if p is None or float(p) < min_prob:
            continue
        faces.append(Face(box=np.asarray(b, dtype=np.float32),
                          prob=float(p),
                          landmarks=np.asarray(lm, dtype=np.float32)))
    return faces


def select_faces(faces: Sequence[Face], policy: str = None) -> List[Face]:
    policy = policy or config.FACE_SELECT_POLICY
    if not faces:
        return []
    if policy == "all":
        return list(faces)
    return [max(faces, key=lambda f: f.area)]


def crop_face(frame_rgb: np.ndarray, face: Face,
              crop_scale: float = None, out_size: int = None) -> Crop:
    """Square crop centred on the detected box, expanded by crop_scale.

    Square because resizing a non-square crop to 224x224 distorts the aspect
    ratio, and the distortion varies with head pose -- which is exactly the
    kind of nuisance signal a detector will happily latch onto.

    Out-of-frame regions are edge-padded rather than clipped, so the face stays
    centred instead of drifting when it sits near a frame border.
    """
    crop_scale = config.CROP_SCALE if crop_scale is None else crop_scale
    out_size = config.IMAGE_SIZE if out_size is None else out_size

    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = face.box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * crop_scale
    side_i = max(2, int(round(side)))
    X1 = int(round(cx - side_i / 2.0))
    Y1 = int(round(cy - side_i / 2.0))
    X2, Y2 = X1 + side_i, Y1 + side_i

    pad_l, pad_t = max(0, -X1), max(0, -Y1)
    pad_r, pad_b = max(0, X2 - w), max(0, Y2 - h)
    sub = frame_rgb[max(0, Y1):min(h, Y2), max(0, X1):min(w, X2)]
    if sub.size == 0:
        sub = np.zeros((side_i, side_i, 3), dtype=frame_rgb.dtype)
    elif pad_l or pad_t or pad_r or pad_b:
        sub = np.pad(sub, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")

    img = Image.fromarray(sub).resize((out_size, out_size), Image.BILINEAR)
    scale = out_size / float(side_i)
    lm = (face.landmarks - np.array([X1, Y1], dtype=np.float32)) * scale
    return Crop(image=img, origin=(X1, Y1), side=side_i, landmarks=lm, face=face)


def crops_from_frame(frame_rgb: np.ndarray, detector,
                     policy: str = None, crop_scale: float = None,
                     out_size: int = None) -> List[Crop]:
    """Full frame -> list of face crops. The one entry point everything uses."""
    faces = select_faces(detect_faces(frame_rgb, detector), policy)
    return [crop_face(frame_rgb, f, crop_scale, out_size) for f in faces]
