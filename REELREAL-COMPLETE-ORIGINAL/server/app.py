"""The middle layer between the browser and the detection pipeline.

WHY THIS EXISTS
---------------
The frontend is HTML/CSS/JS running in a browser. The detector is Python and
PyTorch. A browser cannot run Python, so the two cannot talk directly. This is
a small HTTP server that sits between them:

    browser  --POST video-->  this server  --calls-->  infer_pipeline.py
    browser  <--JSON result--  this server  <--dict--  infer_pipeline.py

It does three jobs and nothing else:
  1. accept an uploaded video over HTTP,
  2. run the existing pipeline on it (unmodified),
  3. translate the pipeline's dict into the shape the frontend already reads
     (see adapter.py).

Run it:
    pip install -r requirements.txt
    set PIPELINE_DIR=C:\\path\\to\\ctf_pretrained     (Windows)
    export PIPELINE_DIR=/path/to/ctf_pretrained       (macOS/Linux)
    uvicorn app:app --port 8000

The first request is slow: the model weights download from Hugging Face and
load into memory. Every request after that reuses the loaded model.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import adapter

# --------------------------------------------------------------------------
# Locating the pipeline
# --------------------------------------------------------------------------
# The detection code lives in its own folder (ctf_pretrained). Rather than
# copying it in here — which would immediately drift out of date — we add it to
# the import path. Set PIPELINE_DIR to wherever that folder actually is.
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", "").strip()
if PIPELINE_DIR and PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

# Only these extensions are accepted. The uploaded filename is NEVER used to
# build a path on disk — see _save_upload — this list is just an early reject.
ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024          # 200 MB

app = FastAPI(title="REEL/REAL detection API", version="1.0")

# The site runs from file:// or a local static server, and the extension popup
# is a chrome-extension:// origin. All three are different origins from this
# server, so the browser needs explicit permission to read the response.
# allow_origins=["*"] is fine for local development. Before this is exposed to
# anyone else, replace it with the real site origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_analyzer = None          # loaded once, on the first request
_model_version = "unknown"


def _get_analyzer():
    """Load the pipeline's analyzer once and keep it in memory.

    Loading is deferred to the first request rather than done at import time so
    the server starts instantly and a missing PIPELINE_DIR produces a clear
    HTTP error instead of a crash at boot.
    """
    global _analyzer, _model_version
    if _analyzer is not None:
        return _analyzer

    try:
        import infer_pipeline
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=("Cannot import the detection pipeline. Set PIPELINE_DIR to "
                    "the folder containing infer_pipeline.py. (%s)" % exc),
        )

    _analyzer = infer_pipeline.VideoAnalyzer.load()
    calibrated = bool(getattr(_analyzer, "clip_calibrator", None))
    _model_version = "ViT-DFv2" + ("" if calibrated else " (uncalibrated)")
    return _analyzer


def _save_upload(upload: UploadFile) -> Path:
    """Stream the upload to a temp file and return its path.

    The client-supplied filename is deliberately not used for the path — only
    its extension is read, and even that is validated against an allow-list.
    A filename like "../../etc/passwd" therefore cannot influence where this
    writes. The name is passed through to the report separately, as text.
    """
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type '%s'. Allowed: %s" % (
                suffix or "(none)", ", ".join(sorted(ALLOWED_SUFFIXES))),
        )

    tmp_dir = Path(tempfile.gettempdir()) / "reelreal_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / ("%s%s" % (uuid.uuid4().hex, suffix))

    written = 0
    with dest.open("wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="File larger than %d MB." % (MAX_UPLOAD_BYTES // (1024 * 1024)),
                )
            out.write(chunk)

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    return dest


def _probe_resolution(path: Path) -> str:
    """Read the pixel height so the report can say "720p". Best effort only."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        cap.release()
        if h > 0:
            return "%dp" % h
        if w > 0:
            return "%dx?" % w
    except Exception:
        pass
    return "unknown"


@app.get("/health")
def health():
    """Cheap check the frontend uses to decide whether a real backend is up."""
    return {
        "ok": True,
        "modelLoaded": _analyzer is not None,
        "modelVersion": _model_version,
        "pipelineDir": PIPELINE_DIR or None,
    }


@app.post("/v1/analyze")
async def analyze(video: UploadFile = File(...)):
    """Analyse one uploaded video and return an AnalysisResult."""
    started = time.time()
    path = _save_upload(video)

    try:
        analyzer = _get_analyzer()
        try:
            raw = analyzer.analyze(str(path))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Analysis failed: %s: %s" % (type(exc).__name__, exc),
            )

        return adapter.to_analysis_result(
            raw,
            file_name=video.filename or path.name,
            file_size=path.stat().st_size,
            resolution=_probe_resolution(path),
            processing_ms=int((time.time() - started) * 1000),
            model_version=_model_version,
            analysed_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        # The video is deleted as soon as the verdict is produced. Nothing is
        # retained on disk between requests.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@app.on_event("shutdown")
def _cleanup():
    shutil.rmtree(Path(tempfile.gettempdir()) / "reelreal_uploads", ignore_errors=True)
