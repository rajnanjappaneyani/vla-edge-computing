"""VLM verifier: looks at a view and returns a raw anomaly logit plus a one-sentence reason.

The logit is log P("Yes") - log P("No") for the first generated token, read from
the model's own distribution rather than parsed from text, so it is a continuous
score that the calibration layer can work with. The reply is prefilled with
"Verdict:" so that first token really is the verdict; without the prefill the
model opens with "The image shows..." and the yes/no ratio is noise.

The model runs in float32. Its shipped bfloat16 weights round log-probabilities
to steps of 0.25, which is too coarse to calibrate on.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

# What counts as an anomaly is site policy, so it is a named preset, not a constant.
ANOMALY_SETS = {
    "general": (
        "fighting or assault, a person falling or lying on the ground, climbing a fence or wall, "
        "theft or break-in, vandalism, a weapon, fire or smoke, or an abandoned bag",
        "walking, standing, talking, hugging, sitting",
    ),
    # CHAD's own definition (parking lot): anything not listed there is normal.
    "chad": (
        "fighting, punching, kicking, pushing, pulling, slapping, strangling, body hitting, theft, "
        "pick-pocketing, tripping someone, chasing, playing with a ball or racket, throwing something, "
        "riding a bicycle or scooter or skateboard, littering, hopping, running, falling, jumping, "
        "or sleeping or lying on the ground",
        "walking, standing, talking, carrying things, using a phone, getting in or out of a car",
    ),
}
ANOMALIES = ANOMALY_SETS["general"][0]
PREFILL = "Verdict:"


def build_question(anomaly_set: str = "general", with_context: bool = False, n_frames: int = 1, gap_s: float = 0.4) -> str:
    anomalies, normal = ANOMALY_SETS[anomaly_set]
    if n_frames > 1:
        seq = f"consecutive frames, {gap_s:g} seconds apart and in time order"
        intro = (
            f"The first {n_frames} images are zoomed-in close-ups from a security camera: {seq}. "
            "The last image is the earlier wide view of the same scene, for context. Judge the close-ups. "
            if with_context
            else f"These {n_frames} images are {seq}, from a security camera. "
        ) + "Compare the frames to see how the people are moving. "
    else:
        intro = (
            "The first image is a zoomed-in close-up from a security camera. "
            "The second image is the earlier wide view of the same scene, for context. Judge the close-up. "
            if with_context
            else "This is a security camera view. "
        )
    return (
        intro + "Is there an anomaly that a guard should be alerted to? "
        f"Anomalies include: {anomalies}. "
        f"Ordinary behaviour ({normal}) is not an anomaly.\n"
        "Reply in exactly this format:\n"
        "Verdict: Yes or No\n"
        "Reason: one sentence of at most 20 words describing what you see."
    )


Frames = "np.ndarray | list[np.ndarray]"  # one view, or a short sequence, oldest first


@dataclass
class Reading:
    logit: float  # raw log-odds of "anomaly"; calibrate before thresholding
    said: str  # "Yes" / "No" as generated
    yes_no_mass: float  # P(Yes) + P(No) at the verdict token; near 1 when the format held
    reason: str
    latency_s: float


class Verifier(Protocol):
    name: str

    def read(self, view_bgr: Frames, context_bgr: np.ndarray | None = None, gap_s: float = 0.4) -> Reading: ...


def _logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + float(np.log(sum(np.exp(v - m) for v in vals)))


class MLXVerifier:
    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        max_tokens: int = 40,
        anomaly_set: str | None = None,
    ):
        from mlx_vlm import apply_chat_template, load, stream_generate

        self.name = model_id.split("/")[-1]
        import mlx.core as mx

        self.model, self.processor = load(model_id)
        self.model.set_dtype(mx.float32)
        self._stream = stream_generate
        self._template = apply_chat_template
        self.max_tokens = max_tokens
        self.anomaly_set = anomaly_set or os.environ.get("LOOKCLOSER_ANOMALIES", "general")
        tok = self.processor.tokenizer

        def single_token_ids(words: list[str]) -> list[int]:
            ids = [tok.encode(w, add_special_tokens=False) for w in words]
            return sorted({i[0] for i in ids if len(i) == 1})

        self.yes_ids = single_token_ids(["Yes", " Yes", "yes", " yes"])
        self.no_ids = single_token_ids(["No", " No", "no", " no"])

    def read(self, view_bgr: Frames, context_bgr: np.ndarray | None = None, gap_s: float = 0.4) -> Reading:
        from PIL import Image

        views = view_bgr if isinstance(view_bgr, list) else [view_bgr]
        images = [Image.fromarray(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in views]
        if context_bgr is not None:
            images.append(Image.fromarray(cv2.cvtColor(context_bgr, cv2.COLOR_BGR2RGB)))
        question = build_question(self.anomaly_set, context_bgr is not None, len(views), gap_s)
        prompt = self._template(self.processor, self.model.config, question, num_images=len(images)) + PREFILL

        t0 = time.time()
        logit, mass, text = 0.0, 0.0, ""
        for i, r in enumerate(
            self._stream(self.model, self.processor, prompt, image=images, max_tokens=self.max_tokens, temperature=0.0)
        ):
            if i == 0:
                lp = r.logprobs.reshape(-1)
                ly = _logsumexp([float(lp[j]) for j in self.yes_ids])
                ln = _logsumexp([float(lp[j]) for j in self.no_ids])
                logit, mass = ly - ln, float(np.exp(ly) + np.exp(ln))
            text += r.text
        head, _, rest = text.strip().partition("\n")
        said = head.strip(" ,.:;").capitalize()
        if said not in ("Yes", "No"):
            said, rest = ("Yes" if logit > 0 else "No"), text
        reason = rest.replace("Reason:", "").strip().split("\n")[0]
        return Reading(
            logit=logit, said=said, yes_no_mass=mass, reason=reason[:1].upper() + reason[1:], latency_s=time.time() - t0
        )


class MockVerifier:
    """Stand-in for tests and UI work: `fn(view, context)` returns (logit, reason)."""

    name = "mock"

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray | None], tuple[float, str]]):
        self.fn = fn

    def read(self, view_bgr: Frames, context_bgr: np.ndarray | None = None, gap_s: float = 0.4) -> Reading:
        logit, reason = self.fn(view_bgr, context_bgr)
        return Reading(logit=logit, said="Yes" if logit > 0 else "No", yes_no_mass=1.0, reason=reason, latency_s=0.0)
