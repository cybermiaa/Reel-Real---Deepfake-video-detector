"""Robust frame sampling from video files.

Shared by extract_crops.py (dataset building) and infer_pipeline.py (live
inference), so training crops and inference crops come off the decoder the
same way.

Three failure modes handled here that a naive VideoCapture loop gets wrong:

1. CAP_PROP_FRAME_COUNT returns 0 or -1 for some containers and for streams
   with a broken index. np.linspace(0, -1, 30) collapses to thirty copies of
   frame zero, which then all "detect a face" and produce a confident verdict
   from a single still. We detect that and fall back to a sequential scan.

2. Seeking with CAP_PROP_POS_FRAMES on H.264 lands on the nearest keyframe,
   not the requested frame, and repeated seeks can silently return the same
   frame. For short clips we decode sequentially instead -- more accurate and
   usually faster than 30 seek operations.

3. A 40-minute press conference sampled at 30 evenly spaced frames is not a
   sample of anything. Long videos switch to time-based sampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

# Hard ceiling on frames decoded during a blind sequential scan, so a corrupt
# duration on a very long file cannot hang the interface.
_BLIND_SCAN_CAP = 6000


@dataclass
class SampledFrame:
    index: int
    t_sec: float
    image: np.ndarray  # RGB, HxWx3


@dataclass
class VideoMeta:
    path: str
    total_frames: int = 0
    fps: float = 0.0
    duration_sec: float = 0.0
    width: int = 0
    height: int = 0
    fps_assumed: bool = False
    duration_assumed: bool = False
    decode_mode: str = ""
    n_requested: int = 0
    n_decoded: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["errors"] = list(self.errors)
        return d


def probe_video(path) -> VideoMeta:
    meta = VideoMeta(path=str(path))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        meta.errors.append("could not open video")
        cap.release()
        return meta
    meta.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    meta.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if not np.isfinite(meta.fps) or meta.fps <= 0:
        meta.fps = 25.0
        meta.fps_assumed = True
        meta.errors.append("fps unavailable, assumed 25")
    if meta.total_frames <= 0:
        meta.duration_assumed = True
        meta.errors.append("frame count unavailable")
    else:
        meta.duration_sec = meta.total_frames / meta.fps
    return meta


def plan_frame_indices(meta: VideoMeta, n_frames: int = None) -> List[int]:
    """Which frame indices to sample, given what we know about the file."""
    n_frames = n_frames or config.N_SAMPLE_FRAMES
    if meta.total_frames <= 0:
        return []

    duration = meta.duration_sec
    if duration > config.LONG_VIDEO_SECONDS:
        # Time-based: one frame every 1/LONG_VIDEO_TARGET_FPS seconds, capped.
        step_sec = 1.0 / max(config.LONG_VIDEO_TARGET_FPS, 1e-6)
        n = int(min(config.LONG_VIDEO_MAX_FRAMES, max(n_frames, duration / step_sec)))
    else:
        n = n_frames

    n = max(1, min(n, meta.total_frames))
    return sorted(set(np.linspace(0, meta.total_frames - 1, n).astype(int).tolist()))


def _to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _read_sequential(cap, wanted: List[int], fps: float) -> List[SampledFrame]:
    """Decode straight through, keeping only the frames we asked for."""
    want = set(wanted)
    last = max(wanted) if wanted else -1
    out, idx = [], 0
    while idx <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            out.append(SampledFrame(index=idx, t_sec=idx / fps, image=_to_rgb(frame)))
        idx += 1
    return out


def _read_seek(cap, wanted: List[int], fps: float) -> List[SampledFrame]:
    """Seek to each index. Used only for long videos, where scanning is too slow."""
    out = []
    for i in wanted:
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)):
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # Trust the decoder's reported position over the requested one: on
        # keyframe-only seeks these differ, and the timestamp should reflect
        # the frame we actually got.
        actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or (i + 1)) - 1
        if actual < 0:
            actual = int(i)
        out.append(SampledFrame(index=actual, t_sec=actual / fps, image=_to_rgb(frame)))
    return out


def _blind_scan(cap, n_frames: int, fps: float) -> List[SampledFrame]:
    """No usable frame count: decode everything (capped), then subsample."""
    frames = []
    idx = 0
    while idx < _BLIND_SCAN_CAP:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((idx, frame))
        idx += 1
    if not frames:
        return []
    picks = sorted(set(np.linspace(0, len(frames) - 1, min(n_frames, len(frames)))
                       .astype(int).tolist()))
    return [SampledFrame(index=frames[p][0], t_sec=frames[p][0] / fps,
                         image=_to_rgb(frames[p][1])) for p in picks]


def sample_video(path, n_frames: int = None) -> Tuple[List[SampledFrame], VideoMeta]:
    """Sample frames from a video. Returns ([] , meta) rather than raising."""
    n_frames = n_frames or config.N_SAMPLE_FRAMES
    meta = probe_video(path)
    if "could not open video" in meta.errors:
        return [], meta

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        meta.errors.append("could not reopen video")
        return [], meta

    try:
        wanted = plan_frame_indices(meta, n_frames)
        meta.n_requested = len(wanted) if wanted else n_frames

        if not wanted:
            meta.decode_mode = "blind_scan"
            frames = _blind_scan(cap, n_frames, meta.fps)
            if frames:
                meta.total_frames = max(f.index for f in frames) + 1
                meta.duration_sec = meta.total_frames / meta.fps
        elif meta.total_frames <= config.SEQUENTIAL_DECODE_MAX_FRAMES:
            meta.decode_mode = "sequential"
            frames = _read_sequential(cap, wanted, meta.fps)
        else:
            meta.decode_mode = "seek"
            frames = _read_seek(cap, wanted, meta.fps)
            # Guard against a decoder that ignores seeks and hands back one
            # frame repeatedly -- that would look like a confident verdict
            # computed from a single still.
            if len({f.index for f in frames}) <= max(1, len(frames) // 4):
                meta.errors.append("seek returned duplicate frames, rescanned")
                cap.release()
                cap = cv2.VideoCapture(str(path))
                meta.decode_mode = "sequential_fallback"
                frames = _read_sequential(cap, wanted, meta.fps)
    finally:
        cap.release()

    meta.n_decoded = len(frames)
    if meta.n_decoded == 0:
        meta.errors.append("no frames decoded")
    return frames, meta
