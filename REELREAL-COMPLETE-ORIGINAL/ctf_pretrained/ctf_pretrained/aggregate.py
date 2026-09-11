"""Turn a sequence of per-frame probabilities into one clip-level decision."""
from __future__ import annotations

import sys
from pathlib import Path

# Explicitly add the project root (2 levels up from aggregate.py if nested, or parent) to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from typing import Dict, List, Optional, Sequence
import numpy as np
from ctf_pretrained import config

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
    if not calib:
        f = clip_features(probs, config.DEFAULT_HIGH_THRESH)
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
            decision_thresh: float = None, min_coverage: float = None, high_thresh: float = None) -> dict:
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