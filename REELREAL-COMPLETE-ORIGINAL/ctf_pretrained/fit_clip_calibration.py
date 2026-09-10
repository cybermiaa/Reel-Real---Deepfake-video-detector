"""Clip-level calibration and threshold tuning on held-out whole videos.

This is the step that makes the confidence number honest. Frame-level
temperature scaling calibrates individual crops; it does not transfer to a
clip aggregate, because "22 of 30 frames flagged" is a count and not a
probability. So this script:

  1. runs the full video pipeline over labelled clips, caching per-frame
     probabilities so re-tuning is instant
  2. splits those clips BY GROUP (a fake and its source video stay together)
  3. grid-searches the per-frame flag threshold
  4. fits a logistic regression over aggregate features -> calibrated clip P(fake)
  5. tunes the decision threshold against an explicit error cost
  6. evaluates on a held-out clip slice and draws a CLIP-LEVEL reliability
     diagram -- the level the interface actually displays
  7. writes the calibrator and both thresholds into the checkpoint

Input clips must NOT overlap training. plan_splits.py reserves a `clip` split
of whole groups for exactly this, and extract_crops.py refuses to crop them, so
pointing this script at that split is leak-free by construction.

  python fit_clip_calibration.py \
      --checkpoint /content/drive/MyDrive/dfd_ckpts/model_best.pt \
      --dataset ffpp --root /content/data/ffpp \
      --splits /content/data/splits.json

The false-positive cost defaults to 3: wrongly calling an authentic official
video synthetic is treated as three times worse than missing a fake. Set
--fp-cost to whatever your writeup argues for, and state it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from infer_pipeline import VideoAnalyzer
from model import update_checkpoint


def _stem(p) -> str:
    return Path(str(p)).stem


def collect_jobs(args):
    """-> list of (video_path, label, group_id)"""
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
            raise SystemExit(
                f"Split '{args.split_name}' is empty in {p}. Re-run plan_splits.py "
                "with a larger --clip fraction.")
        before = len(records)
        records = [r for r in records if r.group_id in wanted]
        print(f"[clips] {len(records)} of {before} videos are in the "
              f"'{args.split_name}' split (held out from training)")
    else:
        print("[clips] WARNING: no --splits given. Cannot verify these clips were "
              "held out of training. A calibrator fitted on training videos "
              "reads far better than it is.")

    return [(Path(r.path), r.label, r.group_id) for r in records]


def run_clips(analyzer, jobs, n_frames=None, desc="scoring clips"):
    records = []
    for vpath, label, group_id in tqdm(jobs, desc=desc):
        if not vpath.exists():
            continue
        try:
            r = analyzer.analyze(vpath, n_frames=n_frames, want_gradcam=False)
        except Exception as exc:
            print(f"  [skip] {vpath.name}: {type(exc).__name__}: {exc}")
            continue
        records.append({
            "video": str(vpath), "video_id": vpath.stem, "label": int(label),
            "group_id": str(group_id), "probs": list(r["probs"]),
            "coverage": float(r["coverage"]), "n_sampled": int(r["n_sampled"]),
            "n_faces": int(r["n_faces"]),
        })
    return records


def grid_search_high_thresh(calib_recs, holdout_recs, grid):
    """Pick the per-frame flag threshold by holdout log-loss."""
    yc = np.array([r["label"] for r in calib_recs])
    yh = np.array([r["label"] for r in holdout_recs])
    best = None
    print("\n  high_thresh   holdout_logloss   holdout_auc")
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
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(yh, ph)) if len(np.unique(yh)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        print(f"  {ht:>10.2f}   {ll:>15.4f}   {auc:>11.4f}")
        if best is None or ll < best[1]:
            best = (ht, ll, auc)
    if best is None:
        raise SystemExit("Grid search failed: no threshold produced two classes.")
    return best[0]


def tune_decision_threshold(y_true, clip_probs, fp_cost: float):
    """Minimise fp_cost * FP + FN over candidate thresholds."""
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=config.BEST_CKPT)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path,
                     help="dataset root, used with --dataset and --splits")
    src.add_argument("--videos-csv", type=Path,
                     help="alternative: csv with video_path,label[,group_id]")
    ap.add_argument("--dataset", default="ffpp", choices=ds.DATASETS)
    ap.add_argument("--compression", default=config.FFPP_COMPRESSION,
                    choices=ds.COMPRESSIONS)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--official-test-only", action="store_true")
    ap.add_argument("--splits", type=Path, default=None,
                    help="splits.json; clips are taken from --split-name")
    ap.add_argument("--split-name", default="clip",
                    help="which split holds the reserved calibration videos")
    ap.add_argument("--n-frames", type=int, default=config.N_SAMPLE_FRAMES)
    ap.add_argument("--holdout-size", type=float, default=0.4)
    ap.add_argument("--fp-cost", type=float, default=3.0,
                    help="cost of a false SYNTHETIC relative to a missed fake")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--cache", type=Path, default=None,
                    help="JSON cache of per-clip frame probabilities")
    ap.add_argument("--refresh-cache", action="store_true")
    ap.add_argument("--video-robustness", action="store_true",
                    help="re-encode holdout clips with ffmpeg and re-evaluate")
    ap.add_argument("--robustness-max-clips", type=int, default=60)
    ap.add_argument("--out-dir", type=Path, default=config.REPORT_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="compute and report, but do not modify the checkpoint")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache or (args.out_dir / "clip_probs_cache.json")

    analyzer = VideoAnalyzer.load(args.checkpoint)
    # Ignore any existing calibrator: we are refitting it.
    analyzer.clip_calibrator = None
    print(f"loaded {args.checkpoint}  temperature={analyzer.temperature:.4f}")

    # --- per-clip frame probabilities ------------------------------------
    if cache_path.exists() and not args.refresh_cache:
        records = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"loaded {len(records)} cached clip scores from {cache_path}")
    else:
        jobs = collect_jobs(args)
        if not jobs:
            raise SystemExit("No labelled clips found.")
        print(f"scoring {len(jobs)} clips ({sum(1 for j in jobs if j[1]==1)} fake)")
        records = run_clips(analyzer, jobs, args.n_frames)
        cache_path.write_text(json.dumps(records), encoding="utf-8")
        print(f"cached clip scores -> {cache_path}")

    usable = [r for r in records if r["probs"] and
              r["coverage"] >= config.MIN_FACE_COVERAGE]
    dropped = len(records) - len(usable)
    print(f"\nclips: {len(records)} scored, {len(usable)} usable, "
          f"{dropped} below the {config.MIN_FACE_COVERAGE:.0%} face-coverage floor")
    if len(usable) < 40:
        print("WARNING: fewer than 40 usable clips. The calibrator and the "
              "reliability diagram will be very noisy. Treat the numbers as "
              "indicative and say so.")
    if len(usable) < 10:
        raise SystemExit("Too few usable clips to calibrate.")

    y_all = np.array([r["label"] for r in usable])
    if len(np.unique(y_all)) < 2:
        raise SystemExit("Calibration clips are all one class.")

    # --- group split over clips -----------------------------------------
    groups = np.array([r["group_id"] for r in usable])
    gss = GroupShuffleSplit(n_splits=1, test_size=args.holdout_size,
                            random_state=args.seed)
    ci, hi = next(gss.split(np.zeros(len(usable)), groups=groups))
    calib = [usable[i] for i in ci]
    hold = [usable[i] for i in hi]
    assert not (set(r["group_id"] for r in calib) & set(r["group_id"] for r in hold))
    print(f"calibration clips {len(calib)}  holdout clips {len(hold)}  "
          f"(group-disjoint)")

    # --- per-frame flag threshold ----------------------------------------
    ht = grid_search_high_thresh(calib, hold, np.arange(0.50, 0.96, 0.05))
    print(f"\nselected high_thresh = {ht:.2f}")

    # --- clip calibrator --------------------------------------------------
    Xc = np.stack([aggregate.feature_vector(r["probs"], ht) for r in calib])
    yc = np.array([r["label"] for r in calib])
    calibrator = fit_clip_calibrator(Xc, yc, aggregate.FEATURE_NAMES, ht)
    print("clip calibrator coefficients:")
    for name, w in zip(calibrator["feature_names"], calibrator["coef"]):
        print(f"    {name:<18} {w:+.4f}")
    print(f"    {'intercept':<18} {calibrator['intercept']:+.4f}")

    def clip_prob(recs):
        return np.array([aggregate.apply_clip_calibrator(r["probs"], calibrator)[0]
                         for r in recs])

    p_calib, p_hold = clip_prob(calib), clip_prob(hold)
    y_hold = np.array([r["label"] for r in hold])

    # --- decision threshold ----------------------------------------------
    dt, cost = tune_decision_threshold(yc, p_calib, args.fp_cost)
    print(f"\nselected decision_thresh = {dt:.2f}  "
          f"(fp_cost={args.fp_cost}, calibration-set cost={cost:.0f})")

    # --- holdout evaluation ----------------------------------------------
    m = metrics_report(y_hold, p_hold, dt)
    print_report("holdout clips (CLIP level)", m)
    print(f"  clip ECE {expected_calibration_error(p_hold, y_hold):.4f}   "
          f"brier {brier_score(p_hold, y_hold):.4f}")

    n_bins = 10 if len(hold) >= 80 else 5
    rel_png = plot_reliability(
        p_hold, y_hold, args.out_dir / "reliability_clips_holdout.png",
        title=f"Clip-level reliability (holdout, n={len(hold)})", n_bins=n_bins)
    plot_confusion(m, args.out_dir / "confusion_clips_holdout.png",
                   "Holdout clips")
    print(f"  reliability diagram -> {rel_png}")
    if len(hold) < 80:
        print(f"  ({n_bins} bins, not 10: with {len(hold)} clips ten bins would be "
              "mostly noise.)")

    report = {
        "high_thresh": float(ht), "decision_thresh": float(dt),
        "fp_cost": args.fp_cost, "calibrator": calibrator,
        "n_calib_clips": len(calib), "n_holdout_clips": len(hold),
        "n_dropped_low_coverage": dropped,
        "holdout_metrics": m,
        "holdout_clip_ece": expected_calibration_error(p_hold, y_hold),
        "holdout_clip_brier": brier_score(p_hold, y_hold),
    }

    # --- video-level robustness ------------------------------------------
    if args.video_robustness:
        if not degrade.ffmpeg_available():
            print("\nffmpeg not found; skipping video-level robustness sweep.")
        else:
            print("\n" + "=" * 62)
            print("VIDEO-LEVEL robustness sweep (real codec re-encoding)")
            print("=" * 62)
            tmp = args.out_dir / "degraded"
            tmp.mkdir(parents=True, exist_ok=True)
            subset = hold[:args.robustness_max_clips]
            sweep = {}
            for cond in degrade.VIDEO_CONDITIONS:
                jobs = []
                for r in subset:
                    src_p = Path(r["video"])
                    dst = tmp / f"{cond}__{src_p.stem}.mp4"
                    try:
                        degrade.apply_video(src_p, dst, cond)
                        jobs.append((dst, r["label"], r["group_id"]))
                    except Exception as exc:
                        print(f"  [skip] {src_p.name} [{cond}]: {exc}")
                if not jobs:
                    continue
                recs = run_clips(analyzer, jobs, args.n_frames, desc=f"  {cond}")
                use = [x for x in recs if x["probs"] and
                       x["coverage"] >= config.MIN_FACE_COVERAGE]
                if len(use) < 5:
                    print(f"  {cond:<16} too few usable clips ({len(use)})")
                    continue
                pc = np.array([aggregate.apply_clip_calibrator(x["probs"], calibrator)[0]
                               for x in use])
                yc2 = np.array([x["label"] for x in use])
                mc = metrics_report(yc2, pc, dt)
                sweep[cond] = mc
                print(f"  {cond:<16} n={len(use):<4} acc={mc['accuracy']:.3f} "
                      f"auc={mc.get('roc_auc', float('nan')):.3f} "
                      f"faceless_dropped={len(recs)-len(use)}")
            report["robustness_video_level"] = sweep

    (args.out_dir / "clip_calibration.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out_dir / 'clip_calibration.json'}")

    if args.dry_run:
        print("\n--dry-run: checkpoint NOT modified.")
        return 0

    update_checkpoint(args.checkpoint, clip_calibrator=calibrator,
                      high_thresh=float(ht), decision_thresh=float(dt))
    print(f"\nwrote calibrator + thresholds into {args.checkpoint}")
    print("The interface will now show a calibrated clip probability instead of "
          "a raw flagged-frame fraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
