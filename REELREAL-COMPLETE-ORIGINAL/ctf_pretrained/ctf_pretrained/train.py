"""Fine-tune a face-crop classifier, with validation, model selection and
frame-level temperature fitting.

Run:
  python train.py --manifest /content/data/crops_manifest.csv \
      --arch resnet18 --epochs 10 --batch-size 64

Nothing here trains automatically on import. Every knob has a flag.

Design points worth knowing before you read the loop:

  * Three-way group split. val drives model selection, temperature fitting
    and threshold tuning; test is not touched by this script at all.
  * Model selection is on val ROC-AUC, not training loss. With near-duplicate
    crops, training loss falls steadily while the model memorises scenes.
  * A leakage tripwire fires if epoch-1 val AUC is implausibly high. On a
    correct group split that does not happen, and the likeliest explanation
    is that group_id is not doing its job.
  * The frame-level temperature is fitted on val at the end and written into
    the checkpoint. That is the frame calibration only -- the clip-level
    calibration the interface displays comes from fit_clip_calibration.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from calibrate import collect_logits, fit_temperature, softmax_fake_prob
from dataset import (FaceCropDataset, class_weights, effective_sample_size,
                     load_splits, split_summary)
from model import (amp_autocast, build_model, get_device, make_grad_scaler,
                   save_checkpoint)
from preprocess import eval_tf, train_tf

LEAK_TRIPWIRE_AUC = 0.98


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, transform, batch_size, workers, shuffle):
    ds = FaceCropDataset(df, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False, persistent_workers=bool(workers))


def evaluate_split(model, loader, device, criterion) -> dict:
    logits, labels = collect_logits(model, loader, device)
    probs = softmax_fake_prob(logits, 1.0)
    preds = (probs >= 0.5).astype(int)
    loss = float(criterion(torch.tensor(logits), torch.tensor(labels)).item())
    out = {"loss": loss, "accuracy": float((preds == labels).mean())}
    if len(np.unique(labels)) > 1:
        out["roc_auc"] = float(roc_auc_score(labels, probs))
        out["pr_auc"] = float(average_precision_score(labels, probs))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=config.MANIFEST_CSV)
    ap.add_argument("--arch", default=config.ARCH,
                    choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop after this many epochs without val AUC improvement")
    ap.add_argument("--ckpt-dir", type=Path, default=config.CKPT_DIR)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap training crops; use for a fast smoke test")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    device = get_device()
    device_type = "cuda" if device == "cuda" else "cpu"
    use_amp = (not args.no_amp) and device == "cuda"
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  amp={use_amp}  arch={args.arch}")
    if device == "cpu":
        print("WARNING: no GPU visible. On Colab, Runtime > Change runtime type > GPU.")

    # --- data ------------------------------------------------------------
    train_df, val_df, test_df = load_splits(args.manifest, seed=args.seed)
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)),
                                   random_state=args.seed)
    print(split_summary(train_df, val_df, test_df))
    ess = effective_sample_size(train_df)
    print(f"effective sample size (train): {ess}")
    if ess["groups"] < 100:
        print("WARNING: few independent groups. Expect noisy held-out metrics; "
              "prefer more source videos over more crops per video.")

    workers = args.num_workers
    train_dl = make_loader(train_df, train_tf, args.batch_size, workers, True)
    val_dl = make_loader(val_df, eval_tf, args.batch_size, workers, False)

    # --- model -----------------------------------------------------------
    model = build_model(args.arch).to(device)
    w = class_weights(train_df)
    print(f"class weights (real, fake) = ({w[0]:.3f}, {w[1]:.3f})")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32).to(device))
    cpu_criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = make_grad_scaler(device_type, use_amp)

    best_auc, best_epoch, since_improve = -1.0, -1, 0
    history = []
    last_path = args.ckpt_dir / "model_last.pt"
    best_path = args.ckpt_dir / "model_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        t0 = time.time()
        for x, y in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with amp_autocast(device_type, use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        val = evaluate_split(model, val_dl, device, cpu_criterion)
        secs = time.time() - t0
        print(f"epoch {epoch:>2}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_auc={val['roc_auc']:.4f}  "
              f"val_ap={val['pr_auc']:.4f}  val_acc={val['accuracy']:.4f}  "
              f"({secs:.0f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, "seconds": secs, **val})

        # Checkpoint every epoch: a disconnect costs one epoch, not the run.
        save_checkpoint(last_path, model, args.arch, epoch=epoch, history=history)

        if epoch == 1 and np.isfinite(val["roc_auc"]) and val["roc_auc"] > LEAK_TRIPWIRE_AUC:
            print("\n" + "!" * 62)
            print(f"TRIPWIRE: val ROC-AUC {val['roc_auc']:.4f} after one epoch.")
            print("On a correct group split this does not happen. Assume leakage,")
            print("not success. Check that data_sources.py merged each fake with")
            print("the originals it was built from (look for a 'union-find over")
            print("source/target pairs' line when scanning), and that plan_splits.py")
            print("assigned whole groups rather than individual videos.")
            print("!" * 62 + "\n")

        if np.isfinite(val["roc_auc"]) and val["roc_auc"] > best_auc:
            best_auc, best_epoch, since_improve = val["roc_auc"], epoch, 0
            save_checkpoint(best_path, model, args.arch, epoch=epoch,
                            val_metrics=val, history=history)
            print(f"           new best (val_auc={best_auc:.4f}) -> {best_path.name}")
        else:
            since_improve += 1
            if since_improve >= args.patience:
                print(f"early stop: no val AUC improvement in {args.patience} epochs")
                break

    if best_epoch < 0:
        print("No epoch produced a usable val AUC. Check that val contains both classes.")
        return 1

    # --- frame-level temperature, fitted on val --------------------------
    print(f"\nfitting frame-level temperature on val (best epoch {best_epoch})")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    val_logits, val_labels = collect_logits(model, val_dl, device)
    T = fit_temperature(val_logits, val_labels)
    print(f"temperature T = {T:.4f}"
          + ("  (T>1: model was overconfident, as expected)" if T > 1
             else "  (T<1: model was underconfident -- unusual, check the val split)"))

    np.savez(args.ckpt_dir / "val_logits.npz",
             logits=val_logits, labels=val_labels, temperature=T)
    save_checkpoint(best_path, model, args.arch, epoch=best_epoch,
                    temperature=T, val_metrics=ckpt.get("val_metrics"),
                    history=history)

    (args.ckpt_dir / "history.json").write_text(
        json.dumps({"history": history, "best_epoch": best_epoch,
                    "best_val_auc": best_auc, "temperature": T,
                    "effective_sample_size": ess}, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"best epoch      {best_epoch}  (val ROC-AUC {best_auc:.4f})")
    print(f"checkpoint      {best_path}")
    print(f"temperature     {T:.4f}")
    print("=" * 62)
    print("\nNext: python evaluate.py --checkpoint %s" % best_path)
    print("Then: python fit_clip_calibration.py  (clip-level calibration + thresholds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
