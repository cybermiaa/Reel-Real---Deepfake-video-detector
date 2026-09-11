"""Clip-level calibration and threshold tuning on held-out whole videos."""
from __future__ import annotations

import sys
from pathlib import Path

# Bulletproof path setup so root-level 'config' and modules are always found
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

import aggregate
import config
import data_sources as ds
import degrade
from calibrate import (brier_score, expected_calibration_error, fit_clip_calibrator,
                       plot_reliability)
from evaluate import metrics_report, plot_confusion, print_report
from ctf_pretrained.infer_pipeline import VideoAnalyzer
from model import update_checkpoint


def _stem(p) -> str:
    return Path(str(p)).stem


def collect_jobs(args):
    if args.videos_csv:
        df = pd.read_csv(args.videos_csv)
        need = {"video_path", "label"}
        if not need.issubset(df.columns):
            raise SystemExit(f"--videos-csv needs columns {need}")
        return [(Path(r["video_path"]), int(r["label"]),
                 str(r["group_id"]) if "group_id" in df.columns and pd.notna(r.get("group_id"))
                 else _stem(r["video_path"]))
                for _, r in df.iterrows()]

    records = ds.scan(args.dataset, args.root, args.compression, args.methods,
                      testing_list_only=args.official_test_only)

    if args.splits:
        p = Path(args.splits)
        if not p.exists():
            raise SystemExit(f"--splits not found: {p}. Run plan_splits.py first.")
        blob = json.loads(p.read_text(encoding="utf-8"))
        wanted = set(blob["groups"].get(args.split_name, []))
        if not wanted:
            raise SystemExit(f"Split '{args.split_name}' is empty in {p}.")
        records = [r for r in records if r.group_id in wanted]

    return [(Path(r.path), r.label, r.group_id) for r in records]


def run_clips(analyzer, jobs, n_frames=None, desc="scoring clips"):
    records = []
    for vpath, label, group_id in tqdm(jobs, desc=desc):
        if not vpath.exists():
            continue
        try:
            r = analyzer.analyze(vpath, n_frames=n_frames, want_gradcam=False)
        except Exception as exc:
            print(f"  [skip] {vpath.name}: {exc}")
            continue
        records.append({
            "video": str(vpath), "video_id": vpath.stem, "label": int(label),
            "group_id": str(group_id), "probs": list(r["probs"]),
            "coverage": float(r["coverage"]), "n_sampled": int(r["n_sampled"]),
            "n_faces": int(r["n_faces"]),
        })
    return records


def grid_search_high_thresh(calib_recs, holdout_recs, grid):
    yc = np.array([r["label"] for r in calib_recs])
    yh = np.array([r["label"] for r in holdout_recs])
    best = None
    for ht in grid:
        Xc = np.stack([aggregate.feature_vector(r["probs"], ht) for r in calib_recs])
        Xh = np.stack([aggregate.feature_vector(r["probs"], ht) for r in holdout_recs])
        try:
            cal = fit_clip_calibrator(Xc, yc, aggregate.FEATURE_NAMES, ht)
        except ValueError:
            continue
        w = np.asarray(cal["coef"]); b = cal["intercept"]
        ph = 1.0 / (1.0 + np.exp(-(Xh @ w + b)))
        eps = 1e-9
        ll = float(-np.mean(yh * np.log(ph + eps) + (1 - yh) * np.log(1 - ph + eps)))
        if best is None or ll < best[1]:
            best = (ht, ll, None)
    if best is None:
        raise SystemExit("Grid search failed.")
    return best[0]


def tune_decision_threshold(y_true, clip_probs, fp_cost: float):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(clip_probs, dtype=float)
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        cost = fp_cost * fp + fn
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=config.BEST_CKPT)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--videos-csv", type=Path)
    ap.add_argument("--dataset", default="ffpp", choices=ds.DATASETS)
    ap.add_argument("--compression", default=config.FFPP_COMPRESSION, choices=ds.COMPRESSIONS)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--official-test-only", action="store_true")
    ap.add_argument("--splits", type=Path, default=None)
    ap.add_argument("--split-name", default="clip")
    ap.add_argument("--n-frames", type=int, default=config.N_SAMPLE_FRAMES)
    ap.add_argument("--holdout-size", type=float, default=0.4)
    ap.add_argument("--fp-cost", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.out_dir / "clip_probs_cache.json")

    analyzer = VideoAnalyzer.load(args.checkpoint)
    analyzer.clip_calibrator = None
    print(f"loaded {args.checkpoint}")

    if cache_path.exists() and not args.refresh_cache:
        records = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        jobs = collect_jobs(args)
        records = run_clips(analyzer, jobs, args.n_frames)
        cache_path.write_text(json.dumps(records), encoding="utf-8")

    usable = [r for r in records if r["probs"] and r["coverage"] >= config.MIN_FACE_COVERAGE]
    if len(usable) < 10:
        raise SystemExit("Too few usable clips to calibrate.")

    groups = np.array([r["group_id"] for r in usable])
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_size, random_state=args.seed)
    ci, hi = next(gss.split(np.zeros(len(usable)), groups=groups))
    calib = [usable[i] for i in ci]
    hold = [usable[i] for i in hi]

    ht = grid_search_high_thresh(calib, hold, np.arange(0.50, 0.96, 0.05))
    Xc = np.stack([aggregate.feature_vector(r["probs"], ht) for r in calib])
    yc = np.array([r["label"] for r in calib])
    calibrator = fit_clip_calibrator(Xc, yc, aggregate.FEATURE_NAMES, ht)

    def clip_prob(recs):
        return np.array([aggregate.apply_clip_calibrator(r["probs"], calibrator)[0] for r in recs])

    p_calib, p_hold = clip_prob(calib), clip_prob(hold)
    y_hold = np.array([r["label"] for r in hold])

    dt, cost = tune_decision_threshold(yc, p_calib, args.fp_cost)
    m = metrics_report(y_hold, p_hold, dt)

    report = {
        "high_thresh": float(ht), "decision_thresh": float(dt),
        "fp_cost": args.fp_cost, "calibrator": calibrator,
        "n_calib_clips": len(calib), "n_holdout_clips": len(hold),
        "holdout_metrics": m,
    }
    (args.out_dir / "clip_calibration.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    if args.dry_run:
        return 0

    update_checkpoint(args.checkpoint, clip_calibrator=calibrator, high_thresh=float(ht), decision_thresh=float(dt))
    print(f"\nwrote calibrator + thresholds into {args.checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())