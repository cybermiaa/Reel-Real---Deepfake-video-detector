"""Gradio interface for a non-technical user."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gradio as gr
import numpy as np

import config
import evidence as ev
from infer_pipeline import VideoAnalyzer

ANALYZER = None
LOAD_ERROR = None

BANNER_OK = """
### Deepfake & Synthetic Media Detection — Government Communications
Upload a video. The system analyses **faces only** and returns a verdict with
supporting evidence.
"""

def load_analyzer(ckpt_path):
    global ANALYZER, LOAD_ERROR
    try:
        # Explicitly check pipeline directory environment or fallback
        pipeline_dir = os.environ.get("PIPELINE_DIR", str(Path(__file__).resolve().parent.parent / "ctf_pretrained"))
        os.environ["PIPELINE_DIR"] = pipeline_dir
        
        ANALYZER = VideoAnalyzer.load(ckpt_path)
        
        # Look for calibration json in common locations if not automatically loaded
        if not ANALYZER.clip_calibrator:
            for potential_path in [
                Path(pipeline_dir) / "clip_calibration.json",
                Path("reports/clip_calibration.json"),
                Path("../reports/clip_calibration.json")
            ]:
                if potential_path.exists():
                    try:
                        ANALYZER.load_clip_calibration(str(potential_path))
                        break
                    except Exception:
                        pass

        if not ANALYZER.clip_calibrator:
            LOAD_ERROR = ("Model loaded, but no clip-level calibration was found. "
                          "Scores shown are raw flagged-frame fractions, not "
                          "probabilities. Run fit_clip_calibration.py.")
    except Exception as exc:
        ANALYZER = None
        LOAD_ERROR = f"{type(exc).__name__}: {exc}"


def timeline_plot(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = result.get("frames") or []
    if not frames:
        return None
    ts = [f["t_sec"] for f in frames]
    ps = [f["prob"] for f in frames]
    ht = result.get("high_thresh", config.DEFAULT_HIGH_THRESH)

    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(ts, ps, "-", color="#3b6ea5", lw=1.2, zorder=2)
    flagged = [(t, p) for t, p in zip(ts, ps) if p > ht]
    if flagged:
        ax.scatter(*zip(*flagged), s=26, color="#c1440e", zorder=3, label="flagged")
    ax.axhline(ht, ls="--", lw=1, color="0.5", label=f"frame threshold {ht:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (seconds)")
    ax.set_ylabel("P(manipulated)")
    ax.set_title("Per-frame scores", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def analyze(video_file):
    if video_file is None:
        return ("<p style='font-size:1.15em;'>Upload a video to begin.</p>",
                "", "", "", None, None, "")
    if ANALYZER is None:
        return ("<div style='padding:14px;border-left:5px solid #c1440e;'>"
                "<b>Model unavailable</b><br><code>" + str(LOAD_ERROR) + "</code>"
                "<br><br>Train a model with <code>train.py</code>, or start the app "
                "with <code>--checkpoint &lt;path&gt;</code>.</div>",
                "", "", "", None, None, "")

    try:
        r = ANALYZER.analyze(video_file)
    except Exception as exc:
        return ("<div style='padding:14px;border-left:5px solid #c1440e;'>"
                f"<b>Could not analyse this file</b><br><code>"
                f"{type(exc).__name__}: {exc}</code></div>",
                "", "", "", None, None, "")

    colour = {"SYNTHETIC": "#c1440e",
              "NO MANIPULATION DETECTED": "#2d6a4f",
              "INSUFFICIENT EVIDENCE": "#8a6d3b"}.get(r["verdict"], "#333")
    headline = (f"<div style='padding:14px;border-left:5px solid {colour};'>"
                f"<div style='font-size:1.35em;font-weight:600;'>{r['headline']}</div>"
                f"<div style='margin-top:6px;opacity:0.85;'>{r['guidance']}</div></div>")

    conf = r["confidence_word"]
    cal_note = "" if r["is_calibrated"] else " — not calibrated"
    confidence = f"## {conf}{cal_note}\n_Confidence in this verdict_"

    ev_lines = "\n".join(f"- {s}" for s in r["evidence"])
    evidence_md = f"**What the system noticed**\n\n{ev_lines}"

    details = {
        "verdict": r["verdict"],
        "clip_probability": round(r["clip_prob"], 4),
        "is_calibrated": r["is_calibrated"],
        "decision_threshold": round(r["decision_thresh"], 4),
        "decision_margin": round(r["decision_margin"], 4),
        "frame_threshold": round(r["high_thresh"], 4),
        "frames_flagged": r["n_flagged"],
        "frames_scored": r["n_scored"],
        "frames_sampled": r["n_sampled"],
        "frames_decoded": r["n_decoded"],
        "face_coverage": round(r["coverage"], 4),
        "longest_flagged_run": r["longest_run"],
        "mean_frame_prob": round(r["mean"], 4),
        "max_frame_prob": round(r["max"], 4),
        "temperature": round(r["temperature"], 4),
        "duration_sec": round(r["duration_sec"], 2),
        "decode_mode": r["decode_mode"],
        "decode_notes": r["decode_errors"],
        "per_frame_probabilities": r["probs"],
    }

    region = r.get("region") or {}
    cam_img = None
    if region.get("overlay") is not None:
        cam_img = (np.clip(region["overlay"], 0, 1) * 255).astype("uint8")
    cam_caption = ""
    if cam_img is not None:
        t = r.get("gradcam_t_sec")
        where = f" at {ev.fmt_timestamp(t)}" if t is not None else ""
        cam_caption = (f"Attention map for the highest-scoring frame{where}. "
                       "This shows **where the model looked**, which is not "
                       "proof of manipulation.")

    return (headline, confidence, evidence_md,
            json.dumps(details, indent=2), timeline_plot(r), cam_img, cam_caption)


def build_ui():
    with gr.Blocks(title="Deepfake Detection — Government Communications",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(BANNER_OK)
        if LOAD_ERROR:
            gr.Markdown(f"> **Notice:** {LOAD_ERROR}")

        with gr.Tab("Analyse a video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid = gr.Video(label="Upload a video")
                    btn = gr.Button("Analyse", variant="primary")
                with gr.Column(scale=2):
                    headline = gr.HTML()
                    confidence = gr.Markdown()
                    evidence_md = gr.Markdown()

            with gr.Accordion("Attention map", open=False):
                cam = gr.Image(label="Grad-CAM overlay", type="numpy", height=280)
                cam_caption = gr.Markdown()

            with gr.Accordion("Per-frame scores", open=False):
                plot = gr.Plot()

            with gr.Accordion("Numeric details", open=False):
                details = gr.Code(language="json", label="All measured values")

            gr.Markdown(f"---\n**Limitations**\n\n{ev.LIMITATIONS}")

            btn.click(analyze, inputs=vid,
                      outputs=[headline, confidence, evidence_md, details, plot,
                               cam, cam_caption])

        with gr.Tab("Scope"):
            gr.Markdown(f"""
### What this system does
Detects AI-manipulated **faces** in video -- face swaps and lip-sync
manipulation. Faces are detected in sampled frames, each crop is scored, and
the per-frame scores are aggregated into one clip verdict with a calibrated
confidence.

### What it does not do
{ev.LIMITATIONS}

### How to read the verdict
- **SYNTHETIC** -- calibrated clip probability above the tuned decision threshold.
- **NO MANIPULATION DETECTED** -- below it. Not a certificate of authenticity.
- **INSUFFICIENT EVIDENCE** -- a face was found in fewer than
  {config.MIN_FACE_COVERAGE:.0%} of sampled frames.

Confidence words come from how far the calibrated probability sits from the
decision threshold, not from the verdict itself.
""")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(config.BEST_CKPT))
    ap.add_argument("--share", action="store_true",
                    help="public link (expires after 72h)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    load_analyzer(args.checkpoint)
    if LOAD_ERROR:
        print(f"[app] {LOAD_ERROR}")

    build_ui().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()