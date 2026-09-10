"""Single source of truth for paths, preprocessing and hyperparameters.

Every path literal in this project lives here. If you need to point the
pipeline somewhere else, edit this file or set the matching environment
variable -- do not hardcode paths in the other modules.

The PREPROCESSING BLOCK below is the important part. Those four values
(CROP_SCALE, IMAGE_SIZE, NORM_MEAN, NORM_STD) define the exact pixel
distribution the model is trained on. They are written into every checkpoint
and re-checked at inference time, because a silent mismatch between training
crops and inference crops is the single easiest way to build a detector that
scores well on paper and fails on real uploads.
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser()


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Keep DATA_ROOT on Colab's local session disk (/content), NOT on mounted
# Drive. Reading tens of thousands of small crop files through the Drive FUSE
# layer can triple epoch time.
DATA_ROOT = _env_path("DFD_DATA_ROOT", "/content/data")

# Checkpoints go to Drive so a session disconnect costs one epoch, not the run.
CKPT_DIR = _env_path("DFD_CKPT_DIR", "/content/drive/MyDrive/dfd_ckpts")

# Evaluation artifacts: metrics JSON, reliability diagrams, confusion matrices.
REPORT_DIR = _env_path("DFD_REPORT_DIR", "/content/reports")

# --- dataset roots ---------------------------------------------------------
# FaceForensics++ is the training set; DFD ships from the same downloader and
# lands under the same tree. Celeb-DF v2 is the cross-dataset test.
FFPP_ROOT = DATA_ROOT / "ffpp"
CELEBDF_ROOT = DATA_ROOT / "celebdf"

# Which FF++ compression to train on. c23 is the standard protocol and is
# already lightly compressed, which matches the threat model: real media
# arrives re-encoded, not pristine. The same videos also exist at c40, giving
# a genuine codec robustness test set at no extra cost.
FFPP_COMPRESSION = "c23"

# --- derived defaults. Any script flag overrides these. --------------------
SPLITS_JSON = DATA_ROOT / "splits.json"          # written by plan_splits.py
CROPS_DIR = DATA_ROOT / "crops"                  # written by extract_crops.py
MANIFEST_CSV = DATA_ROOT / "ffpp_c23_manifest.csv"
CELEBDF_MANIFEST_CSV = DATA_ROOT / "celebdf_manifest.csv"

BEST_CKPT = CKPT_DIR / "model_best.pt"


# --------------------------------------------------------------------------
# PREPROCESSING BLOCK -- train and inference must agree on all of this
# --------------------------------------------------------------------------
# CROP_SCALE is the multiplier applied to the *detected face box* to produce
# the square crop. 1.3 means the crop is 30% wider than the detected box,
# centred on it. This is what "~30% margin" means here, stated unambiguously.
#
# NOTE: this is NOT the same as facenet-pytorch's MTCNN(margin=...) argument,
# which is a fixed pixel count defined against the output image size and does
# not scale with the detected box. We never use that argument; we only call
# MTCNN.detect() and do our own cropping in preprocess.py.
CROP_SCALE = 1.3

IMAGE_SIZE = 224

# ImageNet statistics, because the backbone is ImageNet-pretrained.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# Minimum MTCNN detection probability for a face to count.
MIN_FACE_PROB = 0.90

# Which face to score when a frame contains several. "largest" is deliberate:
# MTCNN's default picks the highest-confidence detection, which in
# press-conference footage is often a bystander rather than the speaker.
FACE_SELECT_POLICY = "largest"  # "largest" | "all"


def preproc_signature() -> dict:
    """Written into checkpoints, re-checked at load time."""
    return {
        "crop_scale": CROP_SCALE,
        "image_size": IMAGE_SIZE,
        "norm_mean": list(NORM_MEAN),
        "norm_std": list(NORM_STD),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
ARCH = "resnet18"          # "resnet18" | "efficientnet_b0"
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
SEED = 42

# Fallback split sizes, used only if a manifest has no `split` column. The
# normal path is plan_splits.py, which assigns groups once and writes them to
# SPLITS_JSON so every downstream script agrees.
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Frames sampled per video during crop extraction. Crops from one video are
# near-duplicates, so more videos beats more frames per video: the effective
# sample size is the number of source groups, not the number of crops.
FRAMES_PER_VIDEO = 12

# FF++ has one real video per manipulation method, so with all five methods
# the raw ratio is 1 real : 5 fake. Sampling proportionally more frames from
# each real video rebalances without discarding any fakes. None = auto
# (frames_per_video x number of methods).
REAL_FRAMES_PER_VIDEO = None

# DataLoader workers. Forced to 0 on Windows: the spawn start method has to
# pickle the transform pipeline, and multiprocessing there is more trouble
# than it is worth for local smoke tests.
NUM_WORKERS = 0 if os.name == "nt" else 2


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
N_SAMPLE_FRAMES = 30

# Videos longer than this get time-based sampling instead of 30 frames spread
# across the whole duration. Thirty frames evenly spaced over a 40-minute
# press conference is not a meaningful sample.
LONG_VIDEO_SECONDS = 60.0
LONG_VIDEO_TARGET_FPS = 0.5
LONG_VIDEO_MAX_FRAMES = 120

# Below this many frames we decode sequentially rather than seeking. Frame
# seeking on H.264 lands on the nearest keyframe and is unreliable; for short
# clips sequential decode is both more accurate and usually faster.
SEQUENTIAL_DECODE_MAX_FRAMES = 1500

# INSUFFICIENT EVIDENCE when a face is found in fewer than this fraction of
# the frames we intended to sample.
MIN_FACE_COVERAGE = 1 / 3

# Fallback thresholds. fit_clip_calibration.py overwrites these in the
# checkpoint with values tuned on held-out clips; these are only used if you
# run inference before calibrating, and the interface says so when it does.
DEFAULT_HIGH_THRESH = 0.80
DEFAULT_DECISION_THRESH = 0.50

# Confidence wording is a function of how far the CALIBRATED CLIP probability
# sits from the decision threshold, normalised so 0 = exactly at the threshold
# and 1 = at either extreme. See aggregate.decision_margin().
#
# Expressed as a margin rather than an absolute probability because the
# decision threshold is tuned on held-out clips and is not necessarily 0.5.
CONF_STRONG = 0.70
CONF_MODERATE = 0.35


def ensure_dirs() -> None:
    for d in (DATA_ROOT, CKPT_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
