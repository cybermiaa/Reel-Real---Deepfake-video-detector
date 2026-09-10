"""Frame-level evaluation: metrics, reliability, robustness sweep, cross-dataset.

  python evaluate.py --checkpoint /content/drive/MyDrive/dfd_ckpts/model_best.pt \
      --manifest /content/data/crops_manifest.csv \
      --robustness \
      --cross-manifest /content/data/ffpp_manifest.csv

Everything is computed on the group-split test slice, which train.py never
saw. Notes on what is and is not being measured:

  * Accuracy alone is meaningless on an imbalanced set -- on FF++ with all five
    manipulations the raw ratio is 1 real : 5 fake, so "always predict fake"
    scores 83%. The majority-class baseline is printed alongside every accuracy
    figure so the comparison is unavoidable.

  * Precision and recall are reported for BOTH classes. For this application
    the expensive error is calling an authentic official video synthetic, so
    the operating point at low false-positive rate is reported too.

  * The robustness sweep here is IMAGE level: requantisation and rescaling of
    crops. It is not codec re-encoding, and it is labelled accordingly. The
    video-level sweep needs whole clips -- see fit_clip_calibration.py and the
    --video-robustness flag there.

  * The cross-dataset manifest must be built with extract_crops.py so its
    crops come from the same detector and margin as training. Otherwise the
    "generalization drop" is partly your own preprocessing inconsistency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from torch.utils.data import DataLoader

import config
from calibrate import (brier_score, collect_logits, expected_calibration_error,
                       plot_reliability, softmax_fake_prob)
from dataset import FaceCropDataset, load_manifest, load_splits, split_summary
from degrade import IMAGE_CONDITIONS, DegradeTransform
from model import get_device, load_checkpoint
from preprocess import eval_tf


def metrics_report(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    out = {"n": int(y_true.size), "threshold": float(threshold),
           "accuracy": float((y_pred == y_true).mean())}

    # The number every accuracy figure has to beat.
    rate = float(y_true.mean())
    out["fake_rate"] = rate
    out["majority_baseline_accuracy"] = float(max(rate, 1 - rate))

    if len(np.unique(y_true)) > 1:
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0)
        out["real"] = {"precision": float(p[0]), "recall": float(r[0]),
                       "f1": float(f[0]), "support": int(s[0])}
        out["fake"] = {"precision": float(p[1]), "recall": float(r[1]),
                       "f1": float(f[1]), "support": int(s[1])}
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out["confusion_matrix"] = {"tn": int(tn), "fp": int(fp),
                                   "fn": int(fn), "tp": int(tp)}
        out["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else float("nan")

        # Operating point for the error that actually costs something here:
        # wrongly calling an authentic official video synthetic.
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        for target in (0.01, 0.05, 0.10):
            idx = np.where(fpr <= target)[0]
            if idx.size:
                k = idx[-1]
                out[f"recall_at_fpr_{target:g}"] = {
                    "recall": float(tpr[k]), "fpr": float(fpr[k]),
                    "threshold": float(thr[k])}
    out["brier"] = brier_score(y_prob, y_true)
    out["ece"] = expected_calibration_error(y_prob, y_true)
    return out


def print_report(name: str, m: dict) -> None:
    print(f"\n--- {name} " + "-" * max(4, 54 - len(name)))
    print(f"  n={m['n']}  fake_rate={m.get('fake_rate', float('nan')):.3f}")
    print(f"  accuracy      {m['accuracy']:.4f}   "
          f"(majority baseline {m.get('majority_baseline_accuracy', float('nan')):.4f})")
    if "roc_auc" in m:
        print(f"  roc_auc       {m['roc_auc']:.4f}    pr_auc {m['pr_auc']:.4f}")
        print(f"  fake   P={m['fake']['precision']:.4f} R={m['fake']['recall']:.4f} "
              f"F1={m['fake']['f1']:.4f}  (n={m['fake']['support']})")
        print(f"  real   P={m['real']['precision']:.4f} R={m['real']['recall']:.4f} "
              f"F1={m['real']['f1']:.4f}  (n={m['real']['support']})")
        c = m["confusion_matrix"]
        print(f"  confusion     tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}")
        if "recall_at_fpr_0.05" in m:
            rf = m["recall_at_fpr_0.05"]
            print(f"  recall @5% FPR {rf['recall']:.4f}  (thr={rf['threshold']:.3f})")
    print(f"  ece           {m['ece']:.4f}    brier {m['brier']:.4f}")


def plot_confusion(m: dict, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = m.get("confusion_matrix")
    if not c:
        return
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            frac = mat[i, j] / mat[i].sum() if mat[i].sum() else 0
            ax.text(j, i, f"{int(mat[i,j])}\n{frac:.1%}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=10)
    ax.set_xticks([0, 1], ["pred real", "pred fake"])
    ax.set_yticks([0, 1], ["true real", "true fake"])
    ax.set_title(title, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def make_loader(df, transform, batch_size, workers):
    return DataLoader(FaceCropDataset(df, transform), batch_size=batch_size,
                      shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def score_df(model, df, device, transform, batch_size, workers, temperature):
    dl = make_loader(df, transform, batch_size, workers)
    logits, labels = collect_logits(model, dl, device)
    return softmax_fake_prob(logits, temperature), labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=config.BEST_CKPT)
    ap.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    ap.add_argument("--cross-manifest", type=Path, default=None,
                    help="second dataset, e.g. FF++ crops from extract_crops.py")
    ap.add_argument("--robustness", action="store_true",
                    help="image-level degradation sweep on the test split")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--by-method", action="store_true",
                    help="break results down per manipulation method")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out-dir", type=Path, default=config.REPORT_DIR)
    args = ap.parse_args(argv)

    device = get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    T = float(ckpt.get("temperature") or 1.0)
    print(f"checkpoint {args.checkpoint}  arch={ckpt.get('arch')}  temperature={T:.4f}")
    if ckpt.get("temperature") is None:
        print("WARNING: no temperature in checkpoint; frame probabilities are "
              "uncalibrated (T=1).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(args.checkpoint), "temperature": T}

    # --- in-domain test --------------------------------------------------
    train_df, val_df, test_df = load_splits(args.manifest, seed=args.seed)
    print(split_summary(train_df, val_df, test_df))

    probs, labels = score_df(model, test_df, device, eval_tf,
                             args.batch_size, args.num_workers, T)
    m = metrics_report(labels, probs, args.threshold)
    print_report("in-domain test (frame level)", m)
    report["in_domain"] = m

    plot_confusion(m, args.out_dir / "confusion_test.png", "In-domain test (frames)")
    plot_reliability(probs, labels, args.out_dir / "reliability_frames_test.png",
                     title="Frame-level reliability (test)")

    # Uncalibrated comparison, to show the temperature did something.
    raw, _ = score_df(model, test_df, device, eval_tf,
                      args.batch_size, args.num_workers, 1.0)
    report["in_domain_uncalibrated_ece"] = expected_calibration_error(raw, labels)
    print(f"\n  ECE before temperature scaling: "
          f"{report['in_domain_uncalibrated_ece']:.4f}")
    print(f"  ECE after  temperature scaling: {m['ece']:.4f}")

    # --- per-method breakdown --------------------------------------------
    if args.by_method and "method" in test_df.columns:
        print("\n--- per-method (each method vs all real crops) " + "-" * 14)
        reals = test_df[test_df.label == 0]
        by_method = {}
        for m, sub in test_df[test_df.label == 1].groupby("method"):
            combined = pd.concat([reals, sub])
            p, y = score_df(model, combined, device, eval_tf,
                            args.batch_size, args.num_workers, T)
            mm = metrics_report(y, p, args.threshold)
            by_method[m] = mm
            print(f"  {m:<22} auc={mm.get('roc_auc', float('nan')):.4f}  "
                  f"recall={mm.get('fake', {}).get('recall', float('nan')):.4f}  "
                  f"(n_fake={len(sub)})")
        report["by_method"] = by_method
        if by_method:
            hardest = min(by_method, key=lambda k: by_method[k].get("roc_auc", 1.0))
            print(f"  hardest method: {hardest} "
                  f"(auc={by_method[hardest].get('roc_auc', float('nan')):.4f})")

    # --- robustness sweep (image level) ----------------------------------
    if args.robustness:
        conds = args.conditions or list(IMAGE_CONDITIONS)
        print("\n" + "=" * 62)
        print("IMAGE-LEVEL robustness sweep (requantisation/rescaling of crops).")
        print("This is NOT codec re-encoding -- see fit_clip_calibration.py")
        print("--video-robustness for the whole-clip sweep.")
        print("=" * 62)
        sweep = {}
        for cond in conds:
            p, y = score_df(model, test_df, device, DegradeTransform(cond, eval_tf),
                            args.batch_size, args.num_workers, T)
            mc = metrics_report(y, p, args.threshold)
            sweep[cond] = mc
            base = report["in_domain"].get("roc_auc", float("nan"))
            auc = mc.get("roc_auc", float("nan"))
            print(f"  {cond:<18} acc={mc['accuracy']:.4f}  auc={auc:.4f} "
                  f"(Δ{auc - base:+.4f})  "
                  f"fakeR={mc.get('fake', {}).get('recall', float('nan')):.4f}")
        report["robustness_image_level"] = sweep

    # --- cross-dataset ---------------------------------------------------
    if args.cross_manifest:
        cross_df = load_manifest(args.cross_manifest)
        cross_df = cross_df[cross_df.label >= 0]
        p, y = score_df(model, cross_df, device, eval_tf,
                        args.batch_size, args.num_workers, T)
        mc = metrics_report(y, p, args.threshold)
        print_report("cross-dataset (frame level)", mc)
        report["cross_dataset"] = mc
        plot_confusion(mc, args.out_dir / "confusion_cross.png", "Cross-dataset (frames)")
        plot_reliability(p, y, args.out_dir / "reliability_frames_cross.png",
                         title="Frame-level reliability (cross-dataset)")
        if "roc_auc" in m and "roc_auc" in mc:
            drop = m["roc_auc"] - mc["roc_auc"]
            report["generalization_drop_auc"] = float(drop)
            print(f"\n  generalization drop (ROC-AUC): {drop:+.4f}  "
                  f"[{m['roc_auc']:.4f} -> {mc['roc_auc']:.4f}]")
            print("  A substantial drop here is the expected, honest result. Report it.")

    out_json = args.out_dir / "eval_frames.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")
    print(f"plots in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
