"""Face-crop dataset and the leak-free group split.

Manifest schema (produced by extract_crops.py):

    crop_path,label,video_id,group_id[,split,dataset,method,compression]

  label     0 = real, 1 = fake
  video_id  the video the crop came from
  group_id  the SOURCE identity the split groups on
  split     optional; if present it is used verbatim

group_id is not the same as video_id, and the difference matters. In FF++ a
fake named `000_003.mp4` is built from originals `000.mp4` and `003.mp4`. If
you group by video_id alone, that fake can land in train while the real video
it was generated from lands in test -- same person, same scene, same framing.
The model then recognises the scene rather than the manipulation, and the
reported AUC is meaningless. data_sources.py merges each source/target pair
with union-find so the whole family moves together.

Two paths into a split:

  * If the manifest carries a `split` column (the normal case, written by
    extract_crops.py from plan_splits.py), it is used directly. The split was
    decided once, over whole groups, and recorded -- so every script in the
    project agrees and the decision is auditable.

  * Otherwise a group split is computed here as a fallback.

Either way verify_splits() asserts group AND video disjointness before
anything trains.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

import config
from preprocess import train_tf, eval_tf, build_train_transform, build_eval_transform  # noqa: F401

REQUIRED_COLUMNS = ["crop_path", "label", "video_id", "group_id"]


class FaceCropDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, return_index: bool = False):
        self.df = df.reset_index(drop=True)
        self.tf = transform
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row.crop_path).convert("RGB")
        x = self.tf(img)
        y = int(row.label)
        return (x, y, i) if self.return_index else (x, y)


def load_manifest(csv_path=None) -> pd.DataFrame:
    csv_path = Path(csv_path or config.MANIFEST_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {csv_path}\nRun extract_crops.py first.")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")
    return df


def load_splits(csv_path=None, val_size: float = None, test_size: float = None,
                seed: int = None, group_col: str = "group_id"
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-disjoint train/val/test split.

    Uses the manifest's `split` column when present; otherwise falls back to a
    two-stage GroupShuffleSplit. val is for model selection, temperature
    fitting and threshold tuning. test is evaluated once, at the end.
    """
    val_size = config.VAL_SIZE if val_size is None else val_size
    test_size = config.TEST_SIZE if test_size is None else test_size
    seed = config.SEED if seed is None else seed

    df = load_manifest(csv_path)

    if "split" in df.columns and df["split"].notna().any() and (df["split"] != "").any():
        train_df = df[df.split == "train"].copy()
        val_df = df[df.split == "val"].copy()
        test_df = df[df.split == "test"].copy()
        missing = [n for n, d in (("train", train_df), ("val", val_df),
                                  ("test", test_df)) if d.empty]
        if missing:
            raise ValueError(
                f"Manifest has a `split` column but these splits are empty: "
                f"{missing}. Re-run plan_splits.py and extract_crops.py, or pass "
                "a manifest without a split column to fall back to an automatic "
                "split.")
        # 'clip' rows, if any leaked in, must never be trained on.
        n_clip = int((df.split == "clip").sum())
        if n_clip:
            print(f"[dataset] ignoring {n_clip} crops marked split=clip "
                  "(reserved for clip-level calibration)")
        verify_splits(train_df, val_df, test_df, group_col=group_col)
        return train_df, val_df, test_df

    groups = df[group_col].astype(str)

    holdout = val_size + test_size
    if not 0 < holdout < 1:
        raise ValueError(f"val_size + test_size must be in (0, 1), got {holdout}")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=groups))
    train_df, rest_df = df.iloc[train_idx].copy(), df.iloc[rest_idx].copy()

    # Split the holdout into val/test, again by group.
    test_frac_of_rest = test_size / holdout
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac_of_rest, random_state=seed + 1)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df[group_col].astype(str)))
    val_df, test_df = rest_df.iloc[val_idx].copy(), rest_df.iloc[test_idx].copy()

    verify_splits(train_df, val_df, test_df, group_col=group_col)
    return train_df, val_df, test_df


def verify_splits(train_df, val_df, test_df, group_col: str = "group_id") -> None:
    """Assert group AND video disjointness. Raises rather than warns."""
    names = ["train", "val", "test"]
    frames = [train_df, val_df, test_df]
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            for col in (group_col, "video_id"):
                a = set(frames[i][col].astype(str))
                b = set(frames[j][col].astype(str))
                overlap = a & b
                if overlap:
                    sample = sorted(overlap)[:5]
                    raise AssertionError(
                        f"LEAK: {len(overlap)} shared {col} values between "
                        f"{names[i]} and {names[j]} (e.g. {sample}). "
                        "Any metric computed from this split is invalid.")


def split_summary(train_df, val_df, test_df) -> str:
    rows = []
    for name, d in (("train", train_df), ("val", val_df), ("test", test_df)):
        n = len(d)
        n_fake = int((d.label == 1).sum())
        n_vid = d.video_id.nunique()
        n_grp = d.group_id.nunique()
        pct = 100.0 * n_fake / n if n else 0.0
        rows.append(f"  {name:<6} crops={n:<7} videos={n_vid:<6} groups={n_grp:<6} "
                    f"fake={n_fake} ({pct:.1f}%)")
    return "split summary:\n" + "\n".join(rows)


def class_weights(df: pd.DataFrame, n_classes: int = 2) -> list:
    """Inverse-frequency weights, robust to a class being absent.

    Caveat worth carrying into calibration: class weighting shifts the prior
    the model encodes. Temperature scaling rescales sharpness only and will not
    undo a prior shift, and the deployment prior for government communications
    -- where almost everything is authentic -- is nothing like the balance of
    any training set here. See calibrate.adjust_prior().
    """
    counts = df.label.value_counts().to_dict()
    total = len(df)
    weights = []
    for c in range(n_classes):
        n = counts.get(c, 0)
        weights.append(total / (n_classes * n) if n > 0 else 0.0)
    return weights


def effective_sample_size(df: pd.DataFrame) -> dict:
    """Crops are near-duplicates; independent sample count is closer to groups."""
    return {
        "crops": len(df),
        "videos": int(df.video_id.nunique()),
        "groups": int(df.group_id.nunique()),
        "crops_per_video": round(len(df) / max(df.video_id.nunique(), 1), 2),
    }
