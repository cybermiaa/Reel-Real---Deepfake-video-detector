"""Calibration: temperature scaling, clip-level calibration, reliability.

Raw softmax output is systematically overconfident and cannot honestly be
called a confidence score. Two separate calibrations are needed here, and
conflating them is a mistake worth spelling out:

  FRAME level -- one temperature T fitted on held-out crops, applied as
      logits / T. Makes per-frame P(fake) mean what it says.

  CLIP level  -- the thing the interface actually shows. A frame-level
      temperature does NOT transfer to a clip aggregate: "22 of 30 frames
      flagged" is a count, not a probability. So a small logistic regression
      is fitted over aggregate features on held-out WHOLE VIDEOS, and its
      output is the number the user sees. Reliability diagrams must be
      reported at the level you display, which is clip level.

Prior shift, which temperature scaling does not fix: class-weighted loss and
undersampling both change the prior the model encodes, and no training set's
fake rate resembles the deployment base rate for government communications,
where almost everything is authentic. adjust_prior() below applies the standard
logit correction. Whether you apply it or simply report the caveat, say which
in the writeup.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Collecting logits
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_logits(model, dataloader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for batch in dataloader:
        x, y = batch[0], batch[1]
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y.cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def softmax_fake_prob(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """P(fake) with temperature applied. Class 1 is fake."""
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(axis=1))


# --------------------------------------------------------------------------
# Frame-level temperature
# --------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    max_iter: int = 200) -> float:
    """Fit one temperature by NLL on held-out data.

    Optimises log(T) rather than T. An unconstrained T can be driven to zero
    or negative by the optimiser, which flips the sign of every logit and
    silently inverts the classifier.
    """
    z = torch.tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.long)

    log_t = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_t).item())

    if not np.isfinite(T) or T <= 0:
        return 1.0
    return float(np.clip(T, 0.05, 20.0))


def adjust_prior(logits: np.ndarray, train_prior: float, deploy_prior: float
                 ) -> np.ndarray:
    """Shift binary logits from a training base rate to a deployment base rate.

    train_prior / deploy_prior are P(fake). For government communications the
    deployment prior is very low; using a training-set prior directly makes
    every borderline authentic clip look suspicious.
    """
    train_prior = float(np.clip(train_prior, 1e-6, 1 - 1e-6))
    deploy_prior = float(np.clip(deploy_prior, 1e-6, 1 - 1e-6))
    shift = (np.log(deploy_prior / (1 - deploy_prior)) -
             np.log(train_prior / (1 - train_prior)))
    out = np.array(logits, dtype=np.float64, copy=True)
    out[:, 1] += shift
    return out


# --------------------------------------------------------------------------
# Clip-level calibrator
# --------------------------------------------------------------------------
def fit_clip_calibrator(features: np.ndarray, labels: np.ndarray,
                        feature_names: Sequence[str], high_thresh: float,
                        C: float = 1.0) -> Dict:
    """Logistic regression over aggregate features -> calibrated clip P(fake).

    Stored as plain coefficients so inference needs numpy only, not sklearn.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        raise ValueError("Clip calibration needs both classes in the calibration "
                         "split. Check the group split over videos.")

    lr = LogisticRegression(C=C, max_iter=2000, class_weight=None)
    lr.fit(X, y)
    return {
        "feature_names": list(feature_names),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "high_thresh": float(high_thresh),
        "n_train_clips": int(len(y)),
        "train_fake_rate": float(y.mean()),
    }


# --------------------------------------------------------------------------
# Reliability / calibration error
# --------------------------------------------------------------------------
def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
                      ) -> Dict[str, list]:
    """Predicted probability vs observed frequency, with per-bin counts.

    Counts are returned, not discarded: a bin holding three clips is noise,
    and a reliability diagram that hides that is misleading.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, freq, count, lo_edges = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        lo_edges.append(float(lo))
        count.append(n)
        conf.append(float(p[mask].mean()) if n else float((lo + hi) / 2))
        freq.append(float(y[mask].mean()) if n else float("nan"))
    return {"bin_lower": lo_edges, "bin_confidence": conf,
            "bin_frequency": freq, "bin_count": count, "n_bins": n_bins}


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if p.size == 0:
        return float("nan")
    curve = reliability_curve(p, y, n_bins)
    total = 0.0
    for c, f, n in zip(curve["bin_confidence"], curve["bin_frequency"], curve["bin_count"]):
        if n and np.isfinite(f):
            total += (n / p.size) * abs(c - f)
    return float(total)


def brier_score(probs, labels) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def plot_reliability(probs, labels, out_path, title: str = "Reliability",
                     n_bins: int = 10) -> str:
    """Save a reliability diagram annotated with per-bin sample counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    curve = reliability_curve(probs, labels, n_bins)
    ece = expected_calibration_error(probs, labels, n_bins)

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})

    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1, label="perfect")
    xs = [c for c, n in zip(curve["bin_confidence"], curve["bin_count"]) if n]
    ys = [f for f, n in zip(curve["bin_frequency"], curve["bin_count"]) if n]
    ax.plot(xs, ys, "o-", color="#c1440e", lw=1.6, label="observed")
    for x, y, n in zip(curve["bin_confidence"], curve["bin_frequency"], curve["bin_count"]):
        if n:
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="0.35")
    ax.set_ylabel("observed frequency of fake")
    ax.set_title(f"{title}\nECE={ece:.3f}  n={len(probs)}  (labels = per-bin count)",
                 fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    axh.bar(curve["bin_lower"], curve["bin_count"], width=1.0 / n_bins,
            align="edge", color="#3b6ea5", alpha=0.75)
    axh.set_xlabel("predicted probability")
    axh.set_ylabel("count")
    axh.grid(alpha=0.25)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
