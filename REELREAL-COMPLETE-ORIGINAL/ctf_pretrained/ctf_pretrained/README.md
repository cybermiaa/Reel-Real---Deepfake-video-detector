# Video Deepfake Detection — Government Communications

Frame-level face-crop classification with temporal aggregation. Faces are
detected and cropped from sampled video frames, each crop is scored by a
fine-tuned CNN, and per-frame scores are aggregated into one clip verdict with
a calibrated confidence.

**Scope:** video, faces only. Audio-only manipulation, full-body/scene
manipulation, and provenance verification are explicitly out of scope and are
stated as such in the interface.

---

## Datasets

Train on **FaceForensics++ (c23)**, test on three independent sets. This is the
standard protocol — [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)
trains on FF++ c23 and cross-tests elsewhere — so results are comparable to
published work, and all of it fits on Colab.

| Role | Dataset | Why |
|---|---|---|
| Train | FF++ c23, 5 manipulation methods | Standard protocol; already lightly compressed, matching the threat model |
| Codec robustness | FF++ **c40** | Same videos, heavier H.264. Real compression, not a simulated proxy |
| Cross-dataset | **Celeb-DF v2** | Hardest standard cross-test; celebrity interview footage is closer to broadcast conditions than staged actor datasets |
| Cross-dataset | **DFD** | Free — ships with the same FF++ downloader |
| Unseen generator | One FF++ method held out | Optional, via `--holdout-method` |

**Access:** FF++ needs a signed form at
[github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) —
they email you `faceforensics_download_v4.py`. Celeb-DF v2 needs the form at
[github.com/yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics),
or use a Kaggle mirror.

DFDC is deliberately not used: 470 GB, paid actors in domestic settings
(furthest from the target domain), and its pre-extracted-crop workaround is what
introduces crop-geometry mismatch risk in the first place.

---

## Quick start (Colab)

Open [colab_pipeline.ipynb](colab_pipeline.ipynb) and run top to bottom.

Local install (for editing, not training):

```
pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0
```

The `--no-deps` is not optional. facenet-pytorch pins an older torch, and
without the flag pip silently downgrades Colab's torch and breaks CUDA.

---

## Run order

| # | Command | Produces |
|---|---|---|
| 0 | `smoke_test.py` | 30-second wiring check, no data needed |
| 1 | `download_data.py ffpp --script <downloader>` | FF++ videos |
| 2 | `download_data.py celebdf --via kaggle` | Celeb-DF videos |
| 3 | `plan_splits.py --dataset ffpp --root ...` | `splits.json` |
| 4 | `extract_crops.py --dataset ffpp --splits splits.json` | crops + manifest |
| 5 | `train.py --manifest ffpp_c23_manifest.csv` | `model_best.pt` + temperature |
| 6 | `evaluate.py --by-method --robustness` | metrics, reliability, sweep |
| 7 | `evaluate.py --cross-manifest celebdf_manifest.csv` | generalization drop |
| 8 | `fit_clip_calibration.py --splits splits.json` | clip calibrator + thresholds |
| 9 | `app.py --checkpoint model_best.pt --share` | the interface |

Every script takes `--help`. Nothing trains on import.

---

## Files

**Core pipeline**

- [config.py](config.py) — every path and hyperparameter. The preprocessing
  block is written into checkpoints and re-checked at load.
- [data_sources.py](data_sources.py) — FF++ / DFD / Celeb-DF scanners and the
  group keys.
- [preprocess.py](preprocess.py) — face detection, cropping, transforms.
  Shared by extraction and inference.
- [video_io.py](video_io.py) — robust frame sampling.
- [model.py](model.py) — architectures, checkpoint I/O, AMP compatibility.
- [dataset.py](dataset.py) — dataset class and split loading.
- [aggregate.py](aggregate.py) — per-frame scores → clip decision.
- [calibrate.py](calibrate.py) — temperature scaling, clip calibrator,
  reliability, ECE, prior adjustment.
- [gradcam.py](gradcam.py) — Grad-CAM and landmark-based region naming.
- [evidence.py](evidence.py) — sentence templates, each filled from a measurement.
- [infer_pipeline.py](infer_pipeline.py) — video → verdict.

**Scripts**

- [download_data.py](download_data.py) — FF++, DFD, Celeb-DF fetching.
- [plan_splits.py](plan_splits.py) — assigns groups to train/val/test/clip, once.
- [extract_crops.py](extract_crops.py) — videos → crops + manifest.
- [train.py](train.py) — training with validation and model selection.
- [evaluate.py](evaluate.py) — frame metrics, per-method, robustness, cross-dataset.
- [fit_clip_calibration.py](fit_clip_calibration.py) — clip calibration and thresholds.
- [degrade.py](degrade.py) — image- and video-level degradation.
- [app.py](app.py) — Gradio interface.
- [smoke_test.py](smoke_test.py) — end-to-end wiring check on synthetic data.

Run `smoke_test.py` first. It verifies the plumbing, not the model.

---

## The five things this pipeline is careful about

### 1. Training crops and inference crops come from identical code

Both call `preprocess.crops_from_frame`. Because we extract our own crops from
video rather than using a downloaded crop mirror, the two cannot diverge —
there is no crop-geometry mismatch to detect, because none is possible.

Two traps avoided inside that function: `MTCNN.__call__` with the default
`post_process=True` returns tensors standardised as `(x - 127.5) / 128`, a
different space from ImageNet normalization, so we only call `.detect()` and
crop ourselves; and `MTCNN(margin=N)` is a fixed pixel margin against the
output size that does not scale with the detected box, so we expand by
`config.CROP_SCALE` instead.

### 2. The group key merges each fake with the videos it was built from

An FF++ fake named `000_003.mp4` is built from originals `000.mp4` (target) and
`003.mp4` (source). All four of `000`, `003`, `000_003` and `003_000` must move
through the split together, or the model sees the same person in the same scene
on both sides and learns the scene instead of the manipulation.

`data_sources.py` merges source/target pairs with union-find. FF++ pairs its
1000 videos into 500 disjoint couples, giving ~500 groups — fine granularity.

The per-identity fallback is **leaky** and fires only when union-find collapses
everything into one group, which makes splitting otherwise impossible (DFD
swaps its 28 actors densely; Celeb-DF swaps all 59). It prints a warning when
it does. For FF++ it never fires.

`train.py` carries a tripwire: epoch-1 validation AUC above 0.98 prints a
leakage warning instead of looking like success.

### 3. The split is decided once and written down

`plan_splits.py` assigns every group to exactly one of train / val / test /
**clip**, and saves it. Every downstream script reads that file, so they cannot
disagree, and the decision is auditable.

The `clip` split is whole videos reserved for clip-level calibration.
`extract_crops.py` refuses to crop them. Without that reservation, the
calibrator gets fitted on videos the model has already memorised and reads far
better than it is.

### 4. Calibration happens at the level the interface displays

Frame-level temperature scaling does not transfer to a clip aggregate — "22 of
30 frames flagged" is a count, not a probability. Two stages:

- `train.py` fits a temperature on held-out crops (frame level).
- `fit_clip_calibration.py` runs the full pipeline over the reserved `clip`
  videos, fits a logistic regression over aggregate features, tunes both
  thresholds against an explicit error cost, and produces a **clip-level**
  reliability diagram.

The confidence word shown to the user is a function of the calibrated clip
probability's distance from the tuned decision threshold. With no clip
calibration fitted, the interface says "Uncalibrated" rather than presenting a
raw fraction as a probability.

### 5. Every evidence sentence is backed by a measurement

`evidence.py` has no free-text path. Each template takes measured values and
either fills them in or returns `None`. Grad-CAM region naming uses MTCNN's
five landmarks rather than fixed image coordinates, and reports the share of
attention mass near each anchor. When no region reaches 40% of the mass, the
sentence is omitted instead of invented.

---

## Known limitations to state in the writeup

- **Academic benchmarks overstate real-world performance.** Detectors scoring
  well on FF++ and Celeb-DF still fail on deepfakes actually circulating online
  ([Fit for Purpose?, 2025](https://arxiv.org/pdf/2510.16556)). Name this
  rather than hoping nobody asks.
- **FF++ fakes are mostly 2019-era methods.** Current generators are better.
  [DF40](https://github.com/YZY-stack/DF40) covers 40 modern techniques if you
  can obtain even a slice.
- **Robustness levels are not interchangeable.** `evaluate.py --robustness` is
  image-level requantisation and rescaling of crops. The c23→c40 comparison and
  `fit_clip_calibration.py --video-robustness` are real codec re-encoding.
  Report them separately and say which is which.
- **Class prior.** Crops as extracted here are roughly balanced, but the
  deployment base rate for government communications is overwhelmingly
  authentic. Temperature scaling rescales sharpness and will not fix a prior
  shift; `calibrate.adjust_prior()` applies the standard logit correction. Say
  whether you applied it.
- **Effective sample size** is the number of source groups, not crops. Crops
  from one video are near-duplicates. `train.py` prints both.
- **Celeb-DF grouping is by target identity**, not union-find, because it swaps
  all 59 subjects and merging would produce a single group. It is used as a
  test set, where this does not matter.
- **Grad-CAM shows attention, not manipulation.** On a 7×7 final feature map
  each cell covers roughly 32×32 input pixels.
- **Domain gap.** FF++ is YouTube talking-heads; Celeb-DF is celebrity
  interview footage; the target is officials in press-conference and broadcast
  conditions.
- **Dataset terms.** The FF++ and Celeb-DF licences both restrict use to
  research. For a government-facing brief, include a compliance line.

---

## Troubleshooting

**FF++ downloader hangs** — it blocks on `input()` waiting for TOS
acceptance. `download_data.py` feeds it a newline. Run it through that script
rather than directly.

**`No FF++ videos found`** — check `--root` and `--compression`. The expected
layout is `original_sequences/youtube/c23/videos/`.

**Celeb-DF layout not recognised** — expects `Celeb-real/`, `YouTube-real/` and
`Celeb-synthesis/` under `--root`. An extra nesting level from unzipping is
tolerated automatically.

**`[ffpp] WARNING: ... collapsed to a single group`** — you have too small a
subset for pair-merging to work. Download more videos; on the full release this
never happens.

**Epoch-1 val AUC above 0.98** — assume leakage, not success. Check the group key.

**`load_checkpoint` raises about preprocessing** — `config.py` changed since
training. Restore the old values or retrain; the mismatch would not raise on
its own at inference, which is why it raises here.

**App says "Model unavailable"** — no checkpoint at the path. Train one, or
pass `--checkpoint`.

**App says "not calibrated"** — run `fit_clip_calibration.py`.

**Colab session died mid-training** — `model_last.pt` is written every epoch to
Drive.
