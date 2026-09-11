"""End-to-end wiring check. No dataset, no training, ~30 seconds.

Run this first on Colab, straight after installing dependencies:

    python smoke_test.py

It builds synthetic crops and a synthetic video, then exercises every stage the
real pipeline uses: the group split and its leak assertion, class weighting,
transforms and augmentation, checkpoint round-trip including the preprocessing
guard, temperature fitting, clip calibration, aggregation and verdict logic,
evidence templates, degradation, video decoding, and a full VideoAnalyzer pass
with an untrained model.

It does not tell you whether the model is any good -- it tells you the plumbing
is connected, which is the thing worth knowing before you spend GPU time or
download 4 GB.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path

PASS, FAIL, SKIP = [], [], []


def check(name):
    def deco(fn):
        def run():
            try:
                result = fn()
                if result == "skip":
                    SKIP.append(name)
                    print(f"  SKIP  {name}")
                else:
                    PASS.append(name)
                    print(f"  ok    {name}" + (f"  ({result})" if result else ""))
            except Exception as exc:
                FAIL.append((name, exc))
                print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
        run.__name__ = fn.__name__
        return run
    return deco


TMP = Path(tempfile.mkdtemp(prefix="dfd_smoke_"))


# --------------------------------------------------------------------------
@check("imports")
def t_imports():
    import config, preprocess, model, dataset, aggregate, calibrate  # noqa
    import evidence, degrade, video_io, gradcam, data_sources  # noqa
    return f"config.CROP_SCALE={config.CROP_SCALE}"


@check("FF++ group key merges each fake with its source videos")
def t_ffpp_grouping():
    import data_sources as ds
    # FF++ layout: originals 000/003, fakes 000_003 and 003_000.
    root = TMP / "ffpp"
    orig = root / "original_sequences" / "youtube" / "c23" / "videos"
    orig.mkdir(parents=True, exist_ok=True)
    for vid in ("000", "003", "010", "017"):
        (orig / f"{vid}.mp4").write_bytes(b"x")
    for method in ("Deepfakes", "Face2Face"):
        d = root / "manipulated_sequences" / method / "c23" / "videos"
        d.mkdir(parents=True, exist_ok=True)
        for name in ("000_003", "003_000", "010_017", "017_010"):
            (d / f"{name}.mp4").write_bytes(b"x")

    recs = ds.scan_ffpp(root, "c23", verbose=False)
    by_vid = {r.video_id: r for r in recs}

    # The whole 000/003 family must share one group, and must not share it
    # with the 010/017 family.
    fam_a = {by_vid["000"].group_id, by_vid["003"].group_id,
             by_vid["Deepfakes__000_003"].group_id,
             by_vid["Deepfakes__003_000"].group_id,
             by_vid["Face2Face__000_003"].group_id}
    fam_b = {by_vid["010"].group_id, by_vid["017"].group_id,
             by_vid["Deepfakes__010_017"].group_id}
    assert len(fam_a) == 1, f"000/003 family split across groups: {fam_a}"
    assert len(fam_b) == 1, f"010/017 family split across groups: {fam_b}"
    assert fam_a != fam_b, "unrelated families were merged into one group"

    assert sum(r.label for r in recs) == 8, "wrong fake count"
    assert sum(1 for r in recs if r.label == 0) == 4, "wrong real count"
    return f"4 real + 8 fake -> {len({r.group_id for r in recs})} groups"


@check("Celeb-DF labels come from directories, not the list file")
def t_celebdf_scan():
    import data_sources as ds
    root = TMP / "celebdf"
    for d in ("Celeb-real", "YouTube-real", "Celeb-synthesis"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "Celeb-real" / "id0_0000.mp4").write_bytes(b"x")
    (root / "Celeb-real" / "id1_0000.mp4").write_bytes(b"x")
    (root / "YouTube-real" / "00000.mp4").write_bytes(b"x")
    (root / "Celeb-synthesis" / "id0_id1_0000.mp4").write_bytes(b"x")
    (root / "Celeb-synthesis" / "id1_id0_0000.mp4").write_bytes(b"x")
    # Deliberately inverted numeric column: directories must still win.
    (root / "List_of_testing_videos.txt").write_text(
        "0 Celeb-real/id0_0000.mp4\n1 Celeb-synthesis/id0_id1_0000.mp4\n",
        encoding="utf-8")

    recs = ds.scan_celebdf(root, verbose=False)
    lab = {r.video_id: r.label for r in recs}
    assert lab["Celeb-real__id0_0000"] == 0
    assert lab["YouTube-real__00000"] == 0
    assert lab["Celeb-synthesis__id0_id1_0000"] == 1, "fake mislabelled as real"

    # Fakes group by target identity, so id0_id1 groups with real id0.
    g = {r.video_id: r.group_id for r in recs}
    assert g["Celeb-synthesis__id0_id1_0000"] == g["Celeb-real__id0_0000"]

    subset = ds.scan_celebdf(root, testing_list_only=True, verbose=False)
    assert len(subset) == 2, f"official-test filter returned {len(subset)}"
    return "directory labels win over an inverted list column"


@check("split plan is a partition over groups")
def t_plan_splits():
    from plan_splits import assign_groups
    groups = [f"g{i}" for i in range(100)]
    fracs = {"train": 0.6, "val": 0.15, "test": 0.15, "clip": 0.10}
    a = assign_groups(groups, fracs, seed=0)
    flat = [g for gs in a.values() for g in gs]
    assert len(flat) == len(set(flat)) == 100, "not a partition"
    assert all(len(a[k]) > 0 for k in fracs), a
    # Deterministic for a fixed seed.
    assert assign_groups(groups, fracs, seed=0) == a
    return " ".join(f"{k}={len(v)}" for k, v in a.items())


@check("manifest split column is honoured verbatim")
def t_split_column():
    import numpy as np
    import pandas as pd
    from PIL import Image
    from dataset import load_splits

    crops = TMP / "splitcrops"
    crops.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(40):
        split = ["train", "val", "test", "clip"][i % 4]
        p = crops / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(p)
        rows.append({"crop_path": str(p), "label": i % 2,
                     "video_id": f"v{i}", "group_id": f"g{i}", "split": split})
    csv = TMP / "split_manifest.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    tr, va, te = load_splits(csv)
    assert set(tr.split) == {"train"} and set(va.split) == {"val"}
    assert set(te.split) == {"test"}
    # Clips reserved for calibration must never appear in a training split.
    for d in (tr, va, te):
        assert "clip" not in set(d.split)
    return f"train={len(tr)} val={len(va)} test={len(te)}, clip excluded"


@check("synthetic crop dataset + manifest")
def t_manifest():
    import numpy as np
    import pandas as pd
    from PIL import Image

    crops = TMP / "crops"
    rows = []
    rng = np.random.default_rng(0)
    # 20 source groups, 2 videos each, 5 crops each.
    for g in range(20):
        for v in range(2):
            vid = f"vid_{g}_{v}"
            d = crops / vid
            d.mkdir(parents=True, exist_ok=True)
            label = g % 2
            for k in range(5):
                arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
                p = d / f"{k}.png"
                Image.fromarray(arr).save(p)
                rows.append({"crop_path": str(p), "label": label,
                             "video_id": vid, "group_id": f"grp_{g}"})
    df = pd.DataFrame(rows)
    df.to_csv(TMP / "manifest.csv", index=False)
    return f"{len(df)} crops, {df.group_id.nunique()} groups"


@check("three-way group split is disjoint")
def t_split():
    from dataset import effective_sample_size, load_splits, split_summary
    tr, va, te = load_splits(TMP / "manifest.csv", seed=0)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not set(a.group_id) & set(b.group_id)
        assert not set(a.video_id) & set(b.video_id)
    assert len(tr) and len(va) and len(te)
    ess = effective_sample_size(tr)
    assert ess["groups"] <= ess["videos"] <= ess["crops"]
    return f"train={len(tr)} val={len(va)} test={len(te)}"


@check("leak assertion actually fires")
def t_leak_detection():
    import pandas as pd
    from dataset import verify_splits
    df = pd.read_csv(TMP / "manifest.csv")
    a, b = df.iloc[:50], df.iloc[:50]  # deliberate overlap
    try:
        verify_splits(a, b, df.iloc[50:100])
    except AssertionError:
        return "raised as expected"
    raise RuntimeError("verify_splits did NOT catch an overlapping split")


@check("class weights handle a missing class")
def t_class_weights():
    import pandas as pd
    from dataset import class_weights
    df = pd.read_csv(TMP / "manifest.csv")
    w = class_weights(df)
    assert len(w) == 2 and all(x >= 0 for x in w)
    only_fake = df[df.label == 1]
    w2 = class_weights(only_fake)
    assert w2[0] == 0.0  # absent class must not KeyError
    return f"balanced={w[0]:.2f},{w[1]:.2f}"


@check("transforms + augmentation are picklable and shaped right")
def t_transforms():
    import pickle
    import numpy as np
    from PIL import Image
    import config
    from preprocess import eval_tf, train_tf
    img = Image.fromarray(np.random.default_rng(1).integers(
        0, 255, (300, 250, 3), dtype=np.uint8))
    for tf in (train_tf, eval_tf):
        x = tf(img)
        assert tuple(x.shape) == (3, config.IMAGE_SIZE, config.IMAGE_SIZE), x.shape
    pickle.dumps(train_tf)  # DataLoader workers must be able to ship this
    return "train+eval ok, picklable"


@check("checkpoint round-trip and preprocessing guard")
def t_checkpoint():
    import config
    from model import build_model, load_checkpoint, save_checkpoint, update_checkpoint
    p = TMP / "ck.pt"
    m = build_model("resnet18", pretrained=False)
    save_checkpoint(p, m, "resnet18", temperature=1.7)
    m2, ck = load_checkpoint(p, "cpu")
    assert abs(ck["temperature"] - 1.7) < 1e-6
    update_checkpoint(p, decision_thresh=0.42)
    _, ck2 = load_checkpoint(p, "cpu")
    assert abs(ck2["decision_thresh"] - 0.42) < 1e-6

    # The guard must refuse a checkpoint trained under different crop settings.
    old = config.CROP_SCALE
    config.CROP_SCALE = old + 0.5
    try:
        load_checkpoint(p, "cpu", strict_preproc=True)
        raise RuntimeError("preprocessing mismatch was NOT caught")
    except ValueError:
        pass
    finally:
        config.CROP_SCALE = old
    return "save/load/update + mismatch guard"


@check("temperature fitting is stable and positive")
def t_temperature():
    import numpy as np
    from calibrate import expected_calibration_error, fit_temperature, softmax_fake_prob
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    # Overconfident logits: true signal, scaled up. T should come back > 1.
    base = np.where(y == 1, 1.0, -1.0) + rng.normal(0, 0.8, 2000)
    logits = np.stack([-base * 3.0, base * 3.0], axis=1)
    T = fit_temperature(logits, y)
    assert T > 0 and np.isfinite(T)
    e_before = expected_calibration_error(softmax_fake_prob(logits, 1.0), y)
    e_after = expected_calibration_error(softmax_fake_prob(logits, T), y)
    # Temperature scaling minimises NLL, not ECE, so allow a little slack.
    assert e_after <= e_before + 0.02, (e_before, e_after)
    return f"T={T:.3f}, ECE {e_before:.3f} -> {e_after:.3f}"


@check("clip calibrator fit + apply round-trip")
def t_clip_calibrator():
    import numpy as np
    import aggregate
    from calibrate import fit_clip_calibrator
    rng = np.random.default_rng(4)
    recs, labels = [], []
    for i in range(120):
        lab = i % 2
        probs = rng.beta(5, 2, 30) if lab else rng.beta(2, 5, 30)
        recs.append(probs.tolist())
        labels.append(lab)
    X = np.stack([aggregate.feature_vector(p, 0.8) for p in recs])
    cal = fit_clip_calibrator(X, np.array(labels), aggregate.FEATURE_NAMES, 0.8)
    assert len(cal["coef"]) == len(aggregate.FEATURE_NAMES)
    p_fake, is_cal = aggregate.apply_clip_calibrator(recs[1], cal)
    p_real, _ = aggregate.apply_clip_calibrator(recs[0], cal)
    assert is_cal and 0 <= p_fake <= 1 and 0 <= p_real <= 1
    assert p_fake > p_real, (p_fake, p_real)
    return f"P(fake clip)={p_fake:.3f} > P(real clip)={p_real:.3f}"


@check("aggregation and verdict logic")
def t_aggregate():
    import aggregate
    high = [0.95] * 25 + [0.1] * 5
    low = [0.05] * 30

    d = aggregate.decide(high, coverage=1.0, decision_thresh=0.5)
    assert d["verdict"] == "SYNTHETIC", d["verdict"]
    assert d["is_calibrated"] is False  # no calibrator supplied
    assert d["confidence_word"] == "Uncalibrated"

    d2 = aggregate.decide(low, coverage=1.0, decision_thresh=0.5)
    assert d2["verdict"] == "NO MANIPULATION DETECTED", d2["verdict"]

    d3 = aggregate.decide(high, coverage=0.1, decision_thresh=0.5)
    assert d3["verdict"] == "INSUFFICIENT EVIDENCE", d3["verdict"]

    d4 = aggregate.decide([], coverage=0.0)
    assert d4["verdict"] == "INSUFFICIENT EVIDENCE"

    assert aggregate.longest_true_run([1, 1, 0, 1, 1, 1, 0]) == 3
    scattered = aggregate.clip_features([0.9, 0.1] * 15, 0.8)
    assert scattered["longest_run"] == 1
    return "synthetic / clean / low-coverage / empty all correct"


@check("confidence word tracks the calibrated probability")
def t_confidence():
    from aggregate import confidence_word
    words = [confidence_word(p, 0.5, True) for p in (0.99, 0.75, 0.52)]
    assert words == ["Strong", "Moderate", "Weak"], words
    # Must not be a constant looked up from the verdict.
    assert confidence_word(0.51, 0.5, True) != confidence_word(0.99, 0.5, True)
    return " / ".join(words)


@check("evidence sentences carry measurements")
def t_evidence():
    import evidence as ev
    r = {"duration_sec": 10.0, "n_scored": 30, "n_flagged": 22, "high_thresh": 0.8,
         "first_flagged_t": 4.0, "last_flagged_t": 7.0, "longest_run": 2,
         "mean": 0.6, "max": 0.97, "n_faces": 12, "n_sampled": 30,
         "min_coverage": 1 / 3, "decode_errors": [], "n_decoded": 30,
         "is_calibrated": True, "region": None}
    lines = ev.build_evidence(r)
    assert lines and any("22 of 30" in s for s in lines)
    assert not any("attention" in s.lower() for s in lines)  # no region measured
    assert ev.build_evidence({}) == []  # nothing measured -> nothing claimed
    return f"{len(lines)} sentences, none unbacked"


@check("image degradation conditions")
def t_degrade():
    import numpy as np
    from PIL import Image
    from degrade import IMAGE_CONDITIONS, DegradeTransform, apply_image
    from preprocess import eval_tf
    img = Image.fromarray(np.random.default_rng(5).integers(
        0, 255, (224, 224, 3), dtype=np.uint8))
    for name, params in IMAGE_CONDITIONS.items():
        out = apply_image(img, **params)
        assert out.size == img.size, name
        x = DegradeTransform(name, eval_tf)(img)
        assert x.shape[0] == 3
    return f"{len(IMAGE_CONDITIONS)} conditions"


@check("synthetic video decode + timestamps")
def t_video():
    import cv2
    import numpy as np
    import video_io
    path = TMP / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(path), fourcc, 25.0, (320, 240))
    if not w.isOpened():
        return "skip"
    rng = np.random.default_rng(6)
    for i in range(250):  # 10 seconds
        frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        w.write(frame)
    w.release()

    frames, meta = video_io.sample_video(path, n_frames=30)
    assert len(frames) >= 20, f"only decoded {len(frames)}"
    assert meta.n_requested == 30, meta.n_requested
    assert meta.decode_mode == "sequential", meta.decode_mode
    ts = [f.t_sec for f in frames]
    assert ts == sorted(ts) and ts[-1] > 5.0, ts[:3]
    assert len({f.index for f in frames}) == len(frames), "duplicate frames returned"
    return f"{len(frames)} frames, {meta.duration_sec:.1f}s, mode={meta.decode_mode}"


@check("missing/corrupt video fails safely")
def t_bad_video():
    import video_io
    frames, meta = video_io.sample_video(TMP / "does_not_exist.mp4")
    assert frames == [] and meta.errors
    bad = TMP / "corrupt.mp4"
    bad.write_bytes(b"not a video at all")
    frames2, meta2 = video_io.sample_video(bad)
    assert frames2 == []
    return "no exception, empty result + errors recorded"


@check("full VideoAnalyzer pass on a face-free video")
def t_analyzer():
    path = TMP / "test.mp4"
    # The writer test may have skipped if no mp4 encoder was available.
    if not path.exists() or path.stat().st_size < 1000:
        return "skip"
    try:
        from infer_pipeline import VideoAnalyzer
    except ImportError:
        return "skip"
    import config
    an = VideoAnalyzer.untrained()
    r = an.analyze(path, n_frames=8, want_gradcam=True)
    # No faces in noise, so the guard must fire rather than inventing a verdict.
    assert r["verdict"] == "INSUFFICIENT EVIDENCE", r["verdict"]
    assert r["coverage"] < config.MIN_FACE_COVERAGE, r["coverage"]
    assert r["n_sampled"] == 8, r["n_sampled"]
    assert r["headline"] and r["guidance"] and r["limitations"]
    assert isinstance(r["evidence"], list)
    assert r["is_calibrated"] is False
    return f"verdict={r['verdict']}, coverage={r['coverage']:.2f}"


@check("coverage uses requested frames as denominator")
def t_coverage_denominator():
    import aggregate
    # 8 of 30 requested frames decoded, all with faces. Coverage must be 8/30,
    # not 8/8 -- otherwise the insufficient-evidence guard never fires.
    d = aggregate.decide([0.9] * 8, coverage=8 / 30, decision_thresh=0.5)
    assert d["verdict"] == "INSUFFICIENT EVIDENCE", d["verdict"]
    return "8/30 -> INSUFFICIENT EVIDENCE"


@check("Grad-CAM region naming from landmarks")
def t_gradcam_regions():
    import numpy as np
    from gradcam import region_report
    S = 224
    lm = np.array([[80, 90], [144, 90], [112, 130], [90, 165], [134, 165]],
                  dtype=np.float32)
    cam = np.zeros((S, S), dtype=np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    mouth = (lm[3] + lm[4]) / 2
    cam += np.exp(-((xx - mouth[0]) ** 2 + (yy - mouth[1]) ** 2) / (2 * 18 ** 2))
    rep = region_report(cam, lm)
    assert rep and rep["region"] == "mouth and jaw", rep
    # Flat attention must not produce a claim.
    flat = region_report(np.ones((S, S), dtype=np.float32), lm)
    assert flat is not None and flat["region"] is None, flat
    return f"peak->mouth ({rep['share']:.0%}), flat->no claim"


def main() -> int:
    print(f"scratch dir: {TMP}\n")
    for fn in (t_imports, t_ffpp_grouping, t_celebdf_scan, t_plan_splits,
               t_manifest, t_split, t_split_column, t_leak_detection,
               t_class_weights, t_transforms, t_checkpoint, t_temperature,
               t_clip_calibrator, t_aggregate, t_confidence, t_evidence,
               t_degrade, t_video, t_bad_video, t_analyzer,
               t_coverage_denominator, t_gradcam_regions):
        fn()

    print("\n" + "=" * 62)
    print(f"passed {len(PASS)}   failed {len(FAIL)}   skipped {len(SKIP)}")
    if FAIL:
        print("\nfailures:")
        for name, exc in FAIL:
            print(f"  - {name}: {type(exc).__name__}: {exc}")
    print("=" * 62)

    shutil.rmtree(TMP, ignore_errors=True)
    if FAIL:
        return 1
    print("\nPlumbing is connected. This says nothing about model quality --\n"
          "it says the stages talk to each other correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
