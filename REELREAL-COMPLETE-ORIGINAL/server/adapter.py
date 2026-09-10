"""Translate the detection pipeline's output into the shape the frontend reads.

The frontend was built against a mocked detector whose result shape is
documented at the top of site/js/detector.js ("AnalysisResult"). The pipeline
in ctf_pretrained/ produces a different, richer dict. This module is the only
place the two vocabularies meet, so neither side has to know about the other.

THE RULE THIS MODULE ENFORCES
-----------------------------
Every field below is one of three things:

  1. copied straight from a measurement the pipeline produced,
  2. derived from one by a formula written down here in the code,
  3. explicitly marked NOT_MEASURED.

Nothing is invented to fill a slot. If a UI row has no measurement behind it,
it says so. Four of the six evidence rows are in that state today because the
model does not measure blink rate, lip-sync, compression history or C2PA
provenance at all. When detectors for those are added, fill them in here and
the UI lights up with no frontend change.
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional

# Shown in any evidence row the current model has no measurement for. Kept as
# one constant so the UI wording stays consistent and is trivial to grep.
NOT_MEASURED = "Not measured by this model"

# The pipeline's three verdicts -> the frontend's three verdict strings.
# "INSUFFICIENT EVIDENCE" has no equivalent in the original mock, which only
# ever returned synthetic/authentic; app.js has been taught the third value.
VERDICT_MAP = {
    "SYNTHETIC": "synthetic",
    "NO MANIPULATION DETECTED": "authentic",
    "INSUFFICIENT EVIDENCE": "inconclusive",
}


def _per_second_timeline(frames: List[Dict], duration_sec: float,
                         high_thresh: float, decision_thresh: float
                         ) -> List[Dict]:
    """Resample per-frame scores onto the one-bar-per-second grid the UI draws.

    The pipeline does not score every second. It samples a fixed number of
    frames spread across the clip, so a 24-second video might yield scores at
    0.4s, 1.9s, 3.1s and so on. The timeline widget wants one value per second.

    Resampling rule, applied per second:
      - if one or more sampled frames land inside that second, take the MAX
        (a manipulated frame inside an otherwise clean second still matters,
        and averaging would dilute exactly the short edits this tool exists
        to catch),
      - if no sampled frame lands there, hold the value of the nearest sampled
        frame in time.

    The hold is a step function, not an interpolation: it repeats a real
    measurement rather than inventing an intermediate one.
    """
    seconds = max(1, int(round(duration_sec)))
    if not frames:
        return []

    points = sorted(
        ({"t": float(f["t_sec"]), "p": float(f["prob"])} for f in frames),
        key=lambda d: d["t"],
    )

    timeline = []
    for sec in range(seconds):
        inside = [d["p"] for d in points if sec <= d["t"] < sec + 1]
        if inside:
            score = max(inside)
        else:
            nearest = min(points, key=lambda d: abs(d["t"] - (sec + 0.5)))
            score = nearest["p"]

        # Colour by the pipeline's own thresholds, not the mock's, so a red bar
        # means exactly "this frame was flagged" by the same rule the verdict
        # used. Otherwise the chart and the verdict could visibly disagree.
        if score >= high_thresh:
            label = "synthetic"
        elif score >= decision_thresh:
            label = "uncertain"
        else:
            label = "authentic"

        timeline.append({"second": sec, "score": round(score, 4), "label": label})

    return timeline


def _artifacts(result: Dict) -> List[Dict]:
    """Fill the six evidence rows that already exist in the site's markup.

    Row ids e1..e6 are hard-coded in index.html; app.js writes each artifact's
    detail into the matching element. Only rows with a real measurement behind
    them get a value.
    """
    n_scored = int(result.get("n_scored") or 0)
    n_flagged = int(result.get("n_flagged") or 0)
    longest_run = int(result.get("longest_run") or 0)
    region = result.get("region") or {}

    # e1 — Grad-CAM region naming. The pipeline has the machinery but the
    # _explain() hook in infer_pipeline.py is currently stubbed to return
    # {"region": None}, so this stays empty until that is switched back on.
    if region.get("region"):
        e1 = {
            "detail": "Attention concentrated on %s (%.0f%% of the map)" % (
                region.get("phrase", region["region"]),
                100 * float(region.get("share", 0.0)),
            ),
            "severity": "bad",
        }
    else:
        e1 = {"detail": NOT_MEASURED, "severity": "na"}

    # e3 — Temporal consistency. This one IS measured: the pipeline records the
    # longest run of consecutive flagged frames. Sustained runs look like
    # manipulation; scattered single frames look like noise. Same test the
    # pipeline's own evidence.consistency_sentence() applies.
    if n_scored == 0 or n_flagged == 0:
        e3 = {"detail": "No frames flagged", "severity": "ok"}
    elif longest_run >= max(3, int(0.5 * n_flagged)):
        e3 = {
            "detail": "Sustained — %d consecutive flagged frames" % longest_run,
            "severity": "bad",
        }
    else:
        e3 = {
            "detail": "Scattered, not sustained (longest run: %d)" % longest_run,
            "severity": "warn",
        }

    return [
        {"id": "e1", "label": "Face boundary blending", **e1},
        {"id": "e2", "label": "Blink rate & frequency",
         "detail": NOT_MEASURED, "severity": "na"},
        {"id": "e3", "label": "Temporal flicker", **e3},
        {"id": "e4", "label": "Compression trace",
         "detail": NOT_MEASURED, "severity": "na"},
        {"id": "e5", "label": "Lip-sync alignment",
         "detail": NOT_MEASURED, "severity": "na"},
        {"id": "e6", "label": "C2PA Provenance",
         "detail": NOT_MEASURED, "severity": "na"},
    ]


def _manipulated_seconds(result: Dict) -> Optional[float]:
    """Estimate how much of the clip was flagged, in seconds.

    Derivation, stated plainly because this is the one number on the report
    that is computed rather than measured:

        seconds_per_sampled_frame = duration / frames_scored
        manipulated = frames_flagged * seconds_per_sampled_frame

    It is an estimate over a sample, not a frame-exact measurement. Returns
    None when there is nothing to divide by, and the UI then shows a dash.
    """
    n_frames = int(result.get("n_frames") or 0)
    duration = float(result.get("duration_sec") or 0.0)
    if n_frames <= 0 or duration <= 0:
        return None
    return round(int(result.get("n_flagged") or 0) * (duration / n_frames), 1)


def to_analysis_result(result: Dict, *, file_name: str, file_size: int,
                       resolution: str, processing_ms: int,
                       model_version: str, analysed_at: str) -> Dict:
    """pipeline result dict -> AnalysisResult (site/js/detector.js)."""
    verdict = VERDICT_MAP.get(result.get("verdict", ""), "inconclusive")
    is_calibrated = bool(result.get("is_calibrated"))

    first_t = result.get("first_flagged_t")
    last_t = result.get("last_flagged_t")
    flagged_segment = (
        {"startSecond": float(first_t), "endSecond": float(last_t)}
        if first_t is not None and last_t is not None else None
    )

    return {
        "id": "an_" + uuid.uuid4().hex[:10],
        "fileName": file_name,
        "fileSizeBytes": file_size,
        "durationSeconds": float(result.get("duration_sec") or 0.0),
        "resolution": resolution,

        "verdict": verdict,
        "confidence": round(float(result.get("clip_prob") or 0.0), 4),

        # No interval is produced. A logistic clip calibrator gives a point
        # probability, not a 90% band, and clip calibration has not been fitted
        # at all yet (clip_calibrator is None). Sending null makes the UI print
        # "Uncalibrated" instead of a made-up range.
        "calibratedBand": None,

        "manipulatedDurationSeconds": _manipulated_seconds(result),
        "flaggedSegment": flagged_segment,

        "timeline": _per_second_timeline(
            result.get("frames") or [],
            float(result.get("duration_sec") or 0.0),
            float(result.get("high_thresh") or 0.8),
            float(result.get("decision_thresh") or 0.5),
        ),

        "artifacts": _artifacts(result),

        # Nothing in this pipeline inspects file signatures or C2PA manifests,
        # so provenance is reported as unchecked rather than as unsigned.
        "provenance": {"signed": False, "summary": "provenance not checked"},

        "processingTimeMs": processing_ms,
        "modelVersion": model_version,
        "analysedAt": analysed_at,

        # Everything the pipeline measured that the current UI has no slot for.
        # Not rendered today; kept small on purpose because this object is
        # base64'd into a URL fragment when the extension hands off to the site.
        "pipeline": {
            "confidenceWord": result.get("confidence_word"),
            "isCalibrated": is_calibrated,
            "coverage": round(float(result.get("coverage") or 0.0), 3),
            "framesSampled": int(result.get("n_sampled") or 0),
            "framesScored": int(result.get("n_scored") or 0),
            "framesFlagged": int(result.get("n_flagged") or 0),
            "longestRun": int(result.get("longest_run") or 0),
            "highThresh": float(result.get("high_thresh") or 0.0),
            "decisionThresh": float(result.get("decision_thresh") or 0.0),
            "headline": result.get("headline"),
            "guidance": result.get("guidance"),
            "evidence": result.get("evidence") or [],
        },
    }
