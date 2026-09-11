"""Fixed sentence templates, each generated from a measured value.

The rule this module enforces: no sentence is produced that is not backed by a
number the pipeline actually computed. Every function takes measurements and
either returns a sentence containing them, or returns None. There is no
free-text path and nothing here paraphrases a model output into a claim.

If you add a template, it must fill from a measurement. A sentence that would
read the same regardless of what the pipeline measured does not belong here.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def fmt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def flagged_frames_sentence(n_flagged: int, n_scored: int, high_thresh: float,
                            first_t: Optional[float], last_t: Optional[float]
                            ) -> Optional[str]:
    if not n_scored:
        return None
    s = (f"{n_flagged} of {n_scored} analysed frames scored above the "
         f"{high_thresh:.2f} per-frame threshold")
    if n_flagged > 0 and first_t is not None and last_t is not None:
        span = (f", all at {fmt_timestamp(first_t)}" if abs(last_t - first_t) < 0.5
                else f", between {fmt_timestamp(first_t)} and {fmt_timestamp(last_t)}")
        s += span
    return s + "."


def consistency_sentence(longest_run: int, n_flagged: int, n_scored: int
                         ) -> Optional[str]:
    if not n_scored or n_flagged == 0:
        return None
    if longest_run >= max(3, int(0.5 * n_flagged)):
        return (f"The flagged frames were sustained -- {longest_run} in a row -- "
                "which is more consistent with a manipulation than with noise.")
    return (f"The flagged frames were scattered rather than sustained "
            f"(longest run: {longest_run} consecutive), which lowers confidence "
            "in the result.")


def coverage_sentence(n_faces: int, n_sampled: int, min_coverage: float
                      ) -> Optional[str]:
    if not n_sampled:
        return None
    frac = n_faces / n_sampled
    if frac < min_coverage:
        return (f"A face was detectable in only {n_faces} of {n_sampled} sampled "
                f"frames ({frac:.0%}), below the {min_coverage:.0%} needed for a "
                "verdict.")
    if frac < 0.75:
        return (f"A face was detectable in {n_faces} of {n_sampled} sampled frames "
                f"({frac:.0%}); the rest could not be assessed.")
    return (f"A face was detectable in {n_faces} of {n_sampled} sampled frames "
            f"({frac:.0%}).")


def region_sentence(region: Optional[Dict]) -> Optional[str]:
    """Only speaks when one region actually dominated the attention map."""
    if not region or not region.get("region"):
        return None
    return (f"Attention concentrated on {region['phrase']} "
            f"({region['share']:.0%} of the attention map).")


def score_spread_sentence(mean_p: float, max_p: float, n_scored: int
                          ) -> Optional[str]:
    if not n_scored:
        return None
    return (f"Per-frame scores averaged {mean_p:.2f} and peaked at {max_p:.2f}.")


def calibration_caveat(is_calibrated: bool) -> Optional[str]:
    if is_calibrated:
        return None
    return ("This score is NOT calibrated: no clip-level calibration has been "
            "fitted, so the number shown is a raw flagged-frame fraction and "
            "should not be read as a probability.")


def decode_caveat(meta_errors: List[str], n_decoded: int, n_requested: int
                  ) -> Optional[str]:
    if n_requested and n_decoded < n_requested:
        return (f"Only {n_decoded} of {n_requested} requested frames could be "
                "decoded from this file.")
    if meta_errors:
        return f"Decoder notes: {'; '.join(meta_errors)}."
    return None


def duration_sentence(duration_sec: float, n_sampled: int) -> Optional[str]:
    if not duration_sec or duration_sec <= 0:
        return None
    return (f"Sampled {n_sampled} frames across {fmt_timestamp(duration_sec)} "
            "of video.")


def guidance_line(verdict: str) -> str:
    return {
        "SYNTHETIC": ("Treat as suspect. Verify against the official channel "
                      "before sharing."),
        "NO MANIPULATION DETECTED": ("No facial manipulation was detected. This is "
                                     "not a guarantee of authenticity -- verify "
                                     "against the official channel before relying "
                                     "on it."),
        "INSUFFICIENT EVIDENCE": ("Not enough usable face imagery to judge. Try a "
                                  "clearer or longer clip of the same source."),
    }.get(verdict, "Verify against the official channel before sharing.")


def headline(verdict: str) -> str:
    return {
        "SYNTHETIC": "This video shows strong signs of AI manipulation.",
        "NO MANIPULATION DETECTED": ("This video does not show signs of AI facial "
                                     "manipulation."),
        "INSUFFICIENT EVIDENCE": "This video could not be reliably analysed.",
    }.get(verdict, "Result unavailable.")


LIMITATIONS = (
    "Scope: this tool analyses faces in video only.\n"
    "- A real video with a cloned voice will NOT be caught (audio is not analysed).\n"
    "- Full-body or scene manipulation is NOT analysed.\n"
    "- It detects manipulation artifacts, not authorship or provenance.\n"
    "- Attention maps show where the model looked. That is not proof of "
    "manipulation."
)


def build_evidence(result: Dict) -> List[str]:
    """Assemble the evidence panel. Drops any sentence lacking its measurement."""
    sentences = [
        duration_sentence(result.get("duration_sec", 0.0), result.get("n_scored", 0)),
        flagged_frames_sentence(
            result.get("n_flagged", 0), result.get("n_scored", 0),
            result.get("high_thresh", 0.8),
            result.get("first_flagged_t"), result.get("last_flagged_t")),
        consistency_sentence(result.get("longest_run", 0),
                             result.get("n_flagged", 0), result.get("n_scored", 0)),
        score_spread_sentence(result.get("mean", 0.0), result.get("max", 0.0),
                              result.get("n_scored", 0)),
        region_sentence(result.get("region")),
        coverage_sentence(result.get("n_faces", 0), result.get("n_sampled", 0),
                          result.get("min_coverage", 1 / 3)),
        decode_caveat(result.get("decode_errors", []), result.get("n_decoded", 0),
                      result.get("n_sampled", 0)),
        calibration_caveat(result.get("is_calibrated", True)),
    ]
    return [s for s in sentences if s]