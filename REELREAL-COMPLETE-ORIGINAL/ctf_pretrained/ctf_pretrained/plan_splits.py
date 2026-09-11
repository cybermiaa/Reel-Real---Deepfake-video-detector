"""Assign every source group to exactly one split, once, and write it down.

Everything downstream reads this file, so the guarantee is made in one place
and can be audited. Four splits:

  train  fit the model
  val    model selection, temperature fitting, threshold search
  test   touched once, at the end
  clip   whole videos reserved for clip-level calibration -- these are NEVER
         cropped into the training manifest

The `clip` split is the part people usually get wrong. Clip-level calibration
needs whole videos the model has never seen; if those videos also contributed
crops to training, the calibrated confidence is fitted on data the model has
memorised and reads far better than it is.

Splitting is by GROUP, not by video. For FF++ a group is a source/target pair
merged with union-find, so a fake and the original it was made from always land
together. See data_sources.py for the per-dataset group key.

  python plan_splits.py --dataset ffpp --root /content/data/ffpp \\
      --compression c23 --out /content/data/splits.json

Optionally reserve a whole manipulation method as an unseen-generator test:

  python plan_splits.py ... --holdout-method NeuralTextures

That method's fakes are excluded from train/val entirely and evaluated
separately, which gives you a within-dataset generalization number in addition
to the cross-dataset one.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import config
import data_sources as ds

SPLIT_NAMES = ["train", "val", "test", "clip"]


def assign_groups(groups, fractions, seed: int) -> dict:
    """Shuffle groups once, then slice.

    val/test/clip are sized first and train takes the remainder. Letting the
    last split absorb rounding error would starve it, and with a modest number
    of groups that silently produced an empty `clip` split -- leaving nothing
    to calibrate on, which is the one split that must not be empty.
    """
    groups = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(groups)

    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"fractions must sum to 1.0, got {total:.4f}")

    n = len(groups)
    sized = ["val", "test", "clip"]
    counts = {}
    for name in sized:
        # At least one group each, when the fraction asks for any at all.
        want = int(round(fractions[name] * n))
        counts[name] = max(1, want) if fractions[name] > 0 else 0
    counts["train"] = n - sum(counts[k] for k in sized)

    if counts["train"] < 1:
        raise SystemExit(
            f"Only {n} groups available, too few for a "
            f"{'/'.join(f'{fractions[s]:.0%}' for s in SPLIT_NAMES)} split.\n"
            "Download more videos, or lower --val/--test/--clip.")

    out, start = {}, 0
    for name in SPLIT_NAMES:
        out[name] = groups[start:start + counts[name]]
        start += counts[name]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ffpp", choices=ds.DATASETS)
    ap.add_argument("--root", type=Path, default=config.FFPP_ROOT)
    ap.add_argument("--compression", default=config.FFPP_COMPRESSION,
                    choices=ds.COMPRESSIONS)
    ap.add_argument("--methods", nargs="*", default=None,
                    help=f"FF++ methods to include (default all: {ds.FFPP_METHODS})")
    ap.add_argument("--holdout-method", default=None,
                    help="reserve one manipulation method as an unseen-generator test")
    ap.add_argument("--train", type=float, default=0.60)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out", type=Path, default=config.SPLITS_JSON)
    args = ap.parse_args(argv)

    if args.holdout_method and args.holdout_method not in ds.FFPP_METHODS:
        raise SystemExit(f"--holdout-method must be one of {ds.FFPP_METHODS}")

    records = ds.scan(args.dataset, args.root, args.compression, args.methods)
    print(ds.summarize(records))

    groups = {r.group_id for r in records}
    fractions = {"train": args.train, "val": args.val,
                 "test": args.test, "clip": args.clip}
    assignment = assign_groups(groups, fractions, args.seed)

    # Group -> split, and a report of what each split actually contains.
    g2s = {g: s for s, gs in assignment.items() for g in gs}
    per_split = {s: Counter() for s in SPLIT_NAMES}
    per_split_videos = Counter()
    for r in records:
        s = g2s[r.group_id]
        per_split_videos[s] += 1
        per_split[s]["fake" if r.label else "real"] += 1

    payload = {
        "dataset": args.dataset,
        "root": str(args.root),
        "compression": args.compression,
        "methods": args.methods or ds.FFPP_METHODS,
        "holdout_method": args.holdout_method,
        "seed": args.seed,
        "fractions": fractions,
        "groups": assignment,
        "counts": {s: {"groups": len(assignment[s]),
                       "videos": per_split_videos[s],
                       "real": per_split[s]["real"],
                       "fake": per_split[s]["fake"]} for s in SPLIT_NAMES},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"split plan written: {args.out}")
    for s in SPLIT_NAMES:
        c = payload["counts"][s]
        print(f"  {s:<6} groups={c['groups']:<5} videos={c['videos']:<6} "
              f"real={c['real']:<5} fake={c['fake']}")
    print("=" * 62)

    # Every group appears exactly once, by construction -- verify it anyway.
    seen = [g for gs in assignment.values() for g in gs]
    assert len(seen) == len(set(seen)) == len(groups), "group assignment is not a partition"
    print("verified: group assignment is a partition (no group in two splits)")

    if args.holdout_method:
        print(f"\nheld-out method: {args.holdout_method}")
        print("  Excluded from train/val by extract_crops.py. Evaluate it "
              "separately for an unseen-generator number.")
    if payload["counts"]["clip"]["videos"] < 60:
        print(f"\nWARNING: only {payload['counts']['clip']['videos']} videos reserved "
              "for clip calibration. Raise --clip if the calibration set looks thin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
