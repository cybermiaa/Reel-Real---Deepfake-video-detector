"""Videos -> face crops + manifest, using this project's own crop pipeline.

Because crops are extracted here with preprocess.crops_from_frame -- the same
function inference calls -- training crops and inference crops are identical by
construction. That removes an entire class of silent failure: a model trained
on someone else's crop mirror sees a different face scale at inference, keeps
scoring well on held-out crops from that same mirror, and quietly degrades on
real uploads.

Crops are saved as PNG. Saving JPEG here would add a compression generation
before the JPEG augmentation in training and confound the robustness sweep.

Examples
--------
  # Training set: FF++ c23, split labels taken from the plan
  python extract_crops.py --dataset ffpp --root /content/data/ffpp \\
      --compression c23 --splits /content/data/splits.json \\
      --out-dir /content/data/crops/ffpp_c23 \\
      --out-manifest /content/data/ffpp_c23_manifest.csv

  # Cross-dataset test set: Celeb-DF official test list only
  python extract_crops.py --dataset celebdf --root /content/data/celebdf \\
      --official-test-only --out-dir /content/data/crops/celebdf \\
      --out-manifest /content/data/celebdf_manifest.csv

  # Same FF++ videos at c40, for the codec robustness test
  python extract_crops.py --dataset ffpp --root /content/data/ffpp \\
      --compression c40 --splits /content/data/splits.json --only-split test \\
      --out-dir /content/data/crops/ffpp_c40 \\
      --out-manifest /content/data/ffpp_c40_manifest.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
import data_sources as ds
import video_io
from model import get_device
from preprocess import crops_from_frame, get_detector


def load_split_map(path) -> tuple:
    """splits.json -> (group -> split name, held-out method or None)."""
    if not path:
        return {}, None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--splits file not found: {p}\nRun plan_splits.py first.")
    blob = json.loads(p.read_text(encoding="utf-8"))
    g2s = {g: s for s, gs in blob["groups"].items() for g in gs}
    return g2s, blob.get("holdout_method")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ffpp", choices=ds.DATASETS)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--compression", default=config.FFPP_COMPRESSION,
                    choices=ds.COMPRESSIONS)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--official-test-only", action="store_true",
                    help="Celeb-DF: restrict to List_of_testing_videos.txt")
    ap.add_argument("--splits", type=Path, default=None,
                    help="splits.json from plan_splits.py")
    ap.add_argument("--only-split", nargs="*", default=None,
                    help="extract only these splits, e.g. --only-split test")
    ap.add_argument("--skip-clip-split", action="store_true", default=True,
                    help="never crop videos reserved for clip calibration (default on)")
    ap.add_argument("--include-clip-split", dest="skip_clip_split",
                    action="store_false")
    ap.add_argument("--frames-per-video", type=int, default=config.FRAMES_PER_VIDEO)
    ap.add_argument("--real-frames-per-video", type=int,
                    default=config.REAL_FRAMES_PER_VIDEO,
                    help="default: frames-per-video x number of methods, to "
                         "rebalance FF++'s 1 real : N fake ratio")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--policy", choices=["largest", "all"], default="largest")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="reuse crops already on disk for a video")
    args = ap.parse_args(argv)

    records = ds.scan(args.dataset, args.root, args.compression, args.methods,
                      testing_list_only=args.official_test_only)
    print(ds.summarize(records))

    g2s, holdout_method = load_split_map(args.splits)

    # Rebalance: FF++ pairs each real video with one fake per method, so reals
    # are outnumbered N:1. Sampling N times more frames per real video evens
    # the crop counts without throwing any fakes away.
    n_methods = len({r.method for r in records if r.label == 1}) or 1
    real_fpv = args.real_frames_per_video or (args.frames_per_video * n_methods)

    kept, skipped = [], Counter()
    for r in records:
        split = g2s.get(r.group_id, "train" if g2s else "")
        if g2s and r.group_id not in g2s:
            skipped["group not in split plan"] += 1
            continue
        if args.skip_clip_split and split == "clip":
            skipped["reserved for clip calibration"] += 1
            continue
        if args.only_split and split not in args.only_split:
            skipped["not in --only-split"] += 1
            continue
        if holdout_method and r.method == holdout_method and split in ("train", "val"):
            skipped[f"held-out method {holdout_method}"] += 1
            continue
        kept.append((r, split))

    if args.max_videos:
        kept = kept[:args.max_videos]
    if not kept:
        raise SystemExit("Nothing to extract after filtering. Check --splits / "
                         "--only-split.")

    print(f"\nextracting from {len(kept)} videos "
          f"(fake: {args.frames_per_video} frames, real: {real_fpv} frames)")
    for reason, n in skipped.items():
        print(f"  skipped {n}: {reason}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    detector = get_detector(device)
    print(f"device={device}  crop_scale={config.CROP_SCALE}  "
          f"size={config.IMAGE_SIZE}")

    rows, failed = [], []
    for r, split in tqdm(kept, desc="videos"):
        vpath = Path(r.path)
        n_frames = real_fpv if r.label == 0 else args.frames_per_video
        vdir = args.out_dir / r.dataset / r.method / vpath.stem

        if args.resume and vdir.exists():
            existing = sorted(vdir.glob("*.png"))
            if existing:
                for p in existing:
                    rows.append({"crop_path": str(p), "label": r.label,
                                 "video_id": r.video_id, "group_id": r.group_id,
                                 "split": split, "dataset": r.dataset,
                                 "method": r.method, "compression": r.compression,
                                 "frame_index": -1, "t_sec": -1.0})
                continue

        frames, meta = video_io.sample_video(vpath, n_frames)
        if not frames:
            failed.append((vpath.name, "; ".join(meta.errors) or "no frames"))
            continue

        vdir.mkdir(parents=True, exist_ok=True)
        n_saved = 0
        for f in frames:
            for k, crop in enumerate(crops_from_frame(f.image, detector,
                                                      policy=args.policy)):
                out = vdir / f"{f.index:06d}_{k}.png"
                crop.image.save(out)
                rows.append({"crop_path": str(out), "label": r.label,
                             "video_id": r.video_id, "group_id": r.group_id,
                             "split": split, "dataset": r.dataset,
                             "method": r.method, "compression": r.compression,
                             "frame_index": int(f.index), "t_sec": round(f.t_sec, 3)})
                n_saved += 1
        if n_saved == 0:
            failed.append((vpath.name, f"no face found in {len(frames)} frames"))

    if not rows:
        raise SystemExit("No crops extracted. Check that videos decode and "
                         "contain detectable faces.")

    df = pd.DataFrame(rows)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_manifest, index=False)

    n_fake = int((df.label == 1).sum())
    print("\n" + "=" * 62)
    print(f"manifest written: {args.out_manifest}")
    print(f"  crops   {len(df)}")
    print(f"  videos  {df.video_id.nunique()}")
    print(f"  groups  {df.group_id.nunique()}   <- effective sample size")
    print(f"  fake    {n_fake} ({100.0*n_fake/len(df):.1f}%)")
    if "split" in df and df.split.notna().any() and df.split.iloc[0]:
        print("  by split:")
        for s, sub in df.groupby("split"):
            nf = int((sub.label == 1).sum())
            print(f"    {s:<6} crops={len(sub):<7} groups={sub.group_id.nunique():<5} "
                  f"fake={100.0*nf/len(sub):.1f}%")
    print("  by method:")
    for m, sub in df.groupby("method"):
        print(f"    {m:<22} {len(sub)}")
    print(f"  failed videos {len(failed)}")
    for name, why in failed[:10]:
        print(f"     {name}: {why}")
    print("=" * 62)

    balance = 100.0 * n_fake / len(df)
    if balance > 70 or balance < 30:
        print(f"NOTE: class balance is {balance:.0f}% fake. Training uses "
              "class-weighted loss, and evaluate.py prints accuracy next to the "
              "majority baseline, but consider adjusting --real-frames-per-video.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
