"""Turn a sequence of per-frame probabilities into one clip-level decision.

Why this is its own module: the frame scores and the clip verdict are
different quantities, and the calibration story only holds together if the
step between them is explicit.

Averaging dilutes a partial manipulation -- a lip-sync fake leaves authentic
frames between manipulated ones -- so a count of confidently-flagged frames
carries signal the mean loses. But a raw count is not a probability, and
showing one to a user as "confidence" would be exactly the overconfidence the
project sets out to avoid. So the aggregate features here feed a small
clip-level calibrator (fit on held-out whole videos in
fit_clip_calibration.py), and that calibrator's output is what the interface
displays.

A note on attribution, since this design is often justified by pointing at the
DFDC winners: Seferbekov's first-place solution used a *mean* with a
"confident strategy" post-process -- when a large fraction of frames agree
confidently, the mean is replaced by a more extreme value. That is not the
same as a raw count over a threshold. The count is defensible here, but
justify it with your own validation numbers rather than that citation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

import config

# Order is load-bearing: the clip calibrator stores coefficients positionally.
FEATURE_NAMES = ["frac_flagged", "mean", "p90", "max", "std", "longest_run_frac"]


def longest_true_run(mask: Sequence[bool]) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def clip_features(probs: Sequence[float], high_thresh: float = None) -> Dict[str, float]:
    """Descriptive statistics over per-frame P(fake)."""
    high_thresh = config.DEFAULT_HIGH_THRESH if high_thresh is None else high_thresh
    p = np.asarray(list(probs), dtype=np.float64)
    if p.size == 0:
        return {k: 0.0 for k in FEATURE_NAMES} | {"n_frames": 0, "n_flagged": 0,
                                                   "longest_run": 0}
    flagged = p > high_thresh
    run = longest_true_run(flagged.tolist())
    return {
        "frac_flagged": float(flagged.mean()),
        "mean": float(p.mean()),
        "p90": float(np.percentile(p, 90)),
        "max": float(p.max()),
        "std": float(p.std()),
        "longest_run_frac": float(run / p.size),
        "n_frames": int(p.size),
        "n_flagged": int(flagged.sum()),
        "longest_run": int(run),
    }


def feature_vector(probs: Sequence[float], high_thresh: float = None) -> np.ndarray:
    f = clip_features(probs, high_thresh)
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float64)


def apply_clip_calibrator(probs: Sequence[float], calib: Optional[dict]) -> tuple:
    """(clip_probability, is_calibrated).

    FIX (was max-of-frames): the fallback used when no calibrator has been
    fitted used to return the single highest-scoring frame as the "clip
    probability". A maximum is monotonically non-decreasing in the number of
    frames sampled -- every extra frame is one more chance to draw a single
    noisy outlier, so longer clips (or clips sampled at more frames) drifted
    toward SYNTHETIC regardless of content. It also silently contradicted
    evidence.calibration_caveat(), which already told the user the shown
    number is "a raw flagged-frame fraction" -- it never actually was one.

    The fallback now IS that fraction: n_flagged / n_frames at the pipeline's
    own high_thresh. A fraction is bounded in [0, 1] independent of how many
    frames were sampled, so a 30-frame clip and a 120-frame clip with the
    same proportion of flagged frames score the same.
    """
    if not calib:
        f = clip_features(probs, config.DEFAULT_HIGH_THRESH)
        # Flagged-frame fraction: bounded regardless of frame count, and
        # matches what evidence.py already tells the user is being shown.
        fallback_prob = float(f["frac_flagged"]) if f["n_frames"] > 0 else 0.0
        return fallback_prob, False

    ht = float(calib.get("high_thresh", config.DEFAULT_HIGH_THRESH))
    names = calib.get("feature_names", FEATURE_NAMES)
    f = clip_features(probs, ht)
    x = np.array([f[k] for k in names], dtype=np.float64)
    w = np.asarray(calib["coef"], dtype=np.float64)
    b = float(calib["intercept"])
    z = float(np.dot(w, x) + b)
    return float(1.0 / (1.0 + np.exp(-z))), True


def decision_margin(clip_prob: float, decision_thresh: float) -> float:
    """0 at the threshold, 1 at either extreme. Drives the confidence word."""
    dt = min(max(decision_thresh, 1e-6), 1 - 1e-6)
    if clip_prob >= dt:
        return (clip_prob - dt) / (1.0 - dt)
    return (dt - clip_prob) / dt


def confidence_word(clip_prob: float, decision_thresh: float,
                    is_calibrated: bool = True) -> str:
    if not is_calibrated:
        return "Uncalibrated"
    m = decision_margin(clip_prob, decision_thresh)
    if m >= config.CONF_STRONG:
        return "Strong"
    if m >= config.CONF_MODERATE:
        return "Moderate"
    return "Weak"


def decide(probs: Sequence[float], coverage: float, calib: Optional[dict] = None,
            decision_thresh: float = None, min_coverage: float = None) -> dict:
    """Full clip decision: verdict, calibrated probability, confidence word."""
    decision_thresh = (config.DEFAULT_DECISION_THRESH if decision_thresh is None
                       else decision_thresh)
    min_coverage = config.MIN_FACE_COVERAGE if min_coverage is None else min_coverage

    high_thresh = float((calib or {}).get("high_thresh", config.DEFAULT_HIGH_THRESH))
    feats = clip_features(probs, high_thresh)
    clip_prob, is_cal = apply_clip_calibrator(probs, calib)

    if coverage < min_coverage or feats["n_frames"] == 0:
        verdict = "INSUFFICIENT EVIDENCE"
    elif clip_prob >= decision_thresh or (not is_cal and feats["n_flagged"] >= 2):
        verdict = "SYNTHETIC"
    else:
        # Not "REAL": a face-only detector cannot certify a video as authentic.
        verdict = "NO MANIPULATION DETECTED"

    return {
        "verdict": verdict,
        "clip_prob": clip_prob,
        "is_calibrated": is_cal,
        "confidence_word": confidence_word(clip_prob, decision_thresh, is_cal),
        "decision_margin": decision_margin(clip_prob, decision_thresh),
        "decision_thresh": decision_thresh,
        "high_thresh": high_thresh,
        "coverage": float(coverage),
        **feats,
    }