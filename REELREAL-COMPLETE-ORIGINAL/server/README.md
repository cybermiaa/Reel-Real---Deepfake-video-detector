# server/ — the middle layer

The frontend is HTML/CSS/JS in a browser. The detector is Python and PyTorch.
A browser cannot run Python, so they cannot talk directly. This folder is the
small HTTP server that sits between them.

```
browser  ──POST video──▶  server/app.py  ──▶  ctf_pretrained/infer_pipeline.py
browser  ◀──JSON result──  server/app.py  ◀──  (result dict)
```

Two files do the work:

- **`app.py`** — receives the upload, runs the pipeline, deletes the video.
- **`adapter.py`** — translates the pipeline's result dict into the shape the
  frontend already reads. This is the only place the two vocabularies meet.

Nothing in `ctf_pretrained/` was modified.

---

## Running it

Everything goes in a virtual environment so it can't collide with anything else
on the machine. One time only:

```bash
cd server
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Then the pipeline's own dependencies. Note the deviations from
`ctf_pretrained/requirements.txt`:

- **`gradio` and `pandas` are skipped.** They are only used by her standalone
  `app.py` demo, which we don't run. Worse, `gradio` pins its own FastAPI and
  would downgrade ours.
- **PyTorch comes from the CPU index.** No NVIDIA GPU here, so the CUDA build
  would be a 2.5 GB download that never gets used. The CPU wheels are ~250 MB.
- **`facenet-pytorch` must be `--no-deps`**, or it silently downgrades torch.

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install "opencv-python-headless==4.10.0.84" "scikit-learn==1.5.2" "matplotlib==3.9.2"
.venv/Scripts/python.exe -m pip install --no-deps facenet-pytorch==2.6.0
```

Point it at the pipeline, then start it:

```bash
PIPELINE_DIR=/c/Users/Zobia/Downloads/ctf_pretrained .venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

(`uvicorn` is invoked through `python -m` because the venv's scripts are not on
PATH. On macOS/Linux the interpreter is `.venv/bin/python` instead.)

Check it's alive: <http://127.0.0.1:8000/health>

The **first** analysis is slow — the ViT weights download from Hugging Face and
load into memory. Every request after that reuses the loaded model. Measured on
this machine with a 25-second clip: **4m51s cold, 11s warm** (CPU only).

The frontend needs no configuration; `site/js/detector.js` already points at
`http://127.0.0.1:8000`. To aim it somewhere else, set
`window.REELREAL_API_BASE` before that script loads.

---

## What happens when the server isn't running

The site still works. `detector.js` falls back to the original mocked
generator and stamps the report **`SIMULATED RESULT`**. That fallback only
triggers when the server cannot be reached at all — if the server answers with
an error, the error is shown. A real failure is never replaced by an invented
result.

Sample chips are always mocked. They're filenames with no video behind them,
so there is nothing to upload.

---

## Endpoints

| Method | Path | Returns |
|---|---|---|
| `GET` | `/health` | whether the model is loaded |
| `POST` | `/v1/analyze` | `AnalysisResult` (contract at the top of `site/js/detector.js`) |

`POST /v1/analyze` takes one multipart field, `video`. Max 200 MB.
Allowed: `.mp4 .mov .webm .mkv .avi .m4v`.

---

## How the translation works

Most of the report maps across cleanly:

| Report shows | Comes from |
|---|---|
| Synthetic score | `clip_prob` |
| Verdict badge | `verdict`, mapped to synthetic / authentic / inconclusive |
| Timeline bars | `frames[].prob`, resampled onto a one-bar-per-second grid |
| Flagged segment | `first_flagged_t` … `last_flagged_t` |
| Manipulated duration | derived: `n_flagged × (duration ÷ n_frames)` |
| Temporal flicker row | `longest_run` — sustained vs scattered |

### The timeline resample

The pipeline doesn't score every second. It samples frames across the clip, so
a 24-second video might have scores at 0.4s, 1.9s, 3.1s… The chart wants one
bar per second. Per second:

- if sampled frames land inside it, take the **maximum** (averaging would
  dilute exactly the short edits this tool exists to catch),
- if none do, **hold** the nearest sampled frame's score.

The hold repeats a real measurement. It does not interpolate a new one.

### Rows that stay empty

Four evidence rows read *"Not measured by this model"*, greyed and italic:

- **Blink rate** — not measured
- **Compression trace** — not measured
- **Lip-sync alignment** — not measured
- **C2PA provenance** — nothing in the pipeline reads file signatures

**Face boundary blending** will fill itself in automatically as soon as
`_explain()` in `infer_pipeline.py` is un-stubbed — it currently returns
`{"region": None}`, so the Grad-CAM region name never arrives. The adapter
already handles the populated case.

When detectors for the other four exist, add them in `adapter.py::_artifacts`.
No frontend change needed.

---

## ⚠️ Calibration is not done

`clip_calibrator` is `None`, so **every result is uncalibrated**.

Two consequences you can see in the UI:

1. **"Calibrated band" reads `Uncalibrated`.** No interval is sent, because
   none exists. Showing a range here would claim an accuracy that has never
   been checked.
2. **The clip score is the single highest-scoring frame.** That's the
   uncalibrated fallback in `aggregate.apply_clip_calibrator`. One blurry
   frame, one half-turned face, and the whole video gets called synthetic.

To fix, on the pipeline side:

1. train a model → `model_best.pt`
2. run `fit_clip_calibration.py --splits splits.json`
3. load that checkpoint in `VideoAnalyzer.load` instead of the hard-coded
   `dummy_ckpt` (which sets `clip_calibrator: None`)

Until then, treat the number as a ranking, not a probability.

---

## Before this is exposed to anyone else

- `allow_origins=["*"]` in `app.py` → replace with the real site origin.
- `host_permissions` in `extension/manifest.json` → replace the two localhost
  entries with the deployed `https://` origin. Do not use `https://*/*`; Chrome
  will warn users the extension can read data on every site.
- The site now says videos are sent to a server over an encrypted connection.
  That is only true once this is actually served over HTTPS.
- Uploads are deleted immediately after the verdict, and nothing is logged. If
  that changes, the copy on the upload panel has to change with it.
