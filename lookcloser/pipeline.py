"""The loop: watch wide, score candidates, zoom only when unsure, verify, decide.

`run` is a generator of `Tick`s (one per displayed frame), so the CLI renderer and
the web demo consume the same thing. Verifier calls block; a tick with the phase
"scoring" or "verifying" is yielded just before each call so a UI can show it.

Modes
  active   zoom only when the calibrated gate says "unsure"   (this project)
  passive  never zoom; unsure cases are settled by p >= 0.5   (digital-crop baseline)
  always   zoom on every candidate, whatever the gate says    (rule-based PTZ baseline)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

from . import policy
from .calibration import Gate, GateResult
from .detector import Box, Candidate
from .sim_camera import PTZState, SimPTZCamera
from .verifier import Verifier


@dataclass
class Config:
    mode: str = "active"
    check_interval_s: float = 2.0  # how often wide-view candidates are scored
    cooldown_s: float = 8.0  # a decided region is not re-scored for this long
    detect_every: int = 3  # run the detector on every n-th frame
    max_candidates: int = 3  # per scoring round
    dwell_s: float = 0.4  # time on target before the close-up is read (at least the frame span)
    frames: int = 3  # frames per look, so the verifier can see motion; 1 = single still
    frame_gap_s: float = 0.4  # time between those frames
    crop_margin: float = 0.4
    paired_fixed_camera: bool = False  # a second, fixed camera keeps the wide view
    charge_inference_time: bool = False  # let scene time pass while the VLM thinks
    lam: float = 0.1
    mu: float = 0.5


@dataclass
class Tick:
    t: float
    phase: str  # watching | scoring | moving | verifying | returning
    view: np.ndarray  # what the PTZ sees
    wide: np.ndarray  # whole scene, for display; unseen by the system while zoomed
    wide_is_seen: bool
    fov: tuple[float, float, float, float]
    zoom: float
    boxes: list[tuple[Box, str]] = field(default_factory=list)  # scene boxes with a status
    events: list[dict] = field(default_factory=list)


def _iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def digital_crop(frame: np.ndarray, box: Box, margin: float, min_side: int = 336) -> np.ndarray:
    """Crop a region of the frame and enlarge it. Bigger, but no new detail."""
    h, w = frame.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = int(max(0, (box[0] - margin * bw) * w))
    x1 = int(min(w, (box[2] + margin * bw) * w))
    y0 = int(max(0, (box[1] - margin * bh) * h))
    y1 = int(min(h, (box[3] + margin * bh) * h))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return frame
    scale = max(1.0, min_side / max(crop.shape[:2]))
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def region_crop(frame: np.ndarray, fov: tuple[float, float, float, float], size: tuple[int, int]) -> np.ndarray:
    """The part of the wide frame a zoom would show, enlarged to `size`. Same region, no new detail."""
    h, w = frame.shape[:2]
    x0, x1 = int(round(fov[0] * w)), max(int(round(fov[2] * w)), int(round(fov[0] * w)) + 2)
    y0, y1 = int(round(fov[1] * h)), max(int(round(fov[3] * h)), int(round(fov[1] * h)) + 2)
    return cv2.resize(frame[y0:y1, x0:x1], size, interpolation=cv2.INTER_CUBIC)


def spaced(frames: list, step: int, k: int) -> list:
    """Up to `k` items ending at the newest, `step` apart, oldest first."""
    idx = [len(frames) - 1 - i * step for i in range(k)]
    return [frames[i] for i in reversed(idx) if i >= 0]


class Pipeline:
    def __init__(
        self,
        cam: SimPTZCamera,
        detector,
        verifier: Verifier,
        gate: Gate,
        cfg: Config | None = None,
        gate_zoom: Gate | None = None,
    ):
        self.cam, self.detector, self.verifier, self.gate = cam, detector, verifier, gate
        # Close-ups have their own score distribution, so they get their own calibration.
        self.gate_zoom = gate_zoom or gate
        self.cfg = cfg or Config()
        self.cleared: list[tuple[Box, float, str]] = []  # (scene box, expires at, status)
        self.log: list[dict] = []
        self.snaps: list[np.ndarray] = []  # evidence images; events refer to them by index
        self._pending: list[dict] = []

    # -- helpers ------------------------------------------------------------
    def _emit(self, kind: str, **data) -> dict:
        ev = {"t": round(self.cam.t, 2), "kind": kind, **data}
        self.log.append(ev)
        self._pending.append(ev)
        return ev

    def _snap(self, img: np.ndarray) -> int:
        self.snaps.append(img.copy())
        return len(self.snaps) - 1

    def _status(self, box: Box) -> str | None:
        now = self.cam.t
        self.cleared = [c for c in self.cleared if c[1] > now]
        for b, _, status in self.cleared:
            if _iou(b, box) > 0.3:
                return status
        return None

    def _tick(self, phase: str, view: np.ndarray, boxes: list[tuple[Box, str]]) -> Tick:
        zoomed = not self.cam.state.is_wide
        events, self._pending = self._pending, []
        return Tick(
            t=self.cam.t,
            phase=phase,
            view=view,
            wide=self.cam.oracle_wide() if zoomed else view,
            wide_is_seen=(not zoomed) or self.cfg.paired_fixed_camera,
            fov=self.cam.fov(),
            zoom=self.cam.state.zoom,
            boxes=boxes,
            events=events,
        )

    def _read(self, view, context: np.ndarray | None = None) -> tuple[GateResult, object]:
        reading = self.verifier.read(view, context, self.cfg.frame_gap_s)
        if self.cfg.charge_inference_time:
            self.cam.skip(reading.latency_s)
        gate = self.gate if context is None else self.gate_zoom
        return gate.decide(reading.logit), reading

    def _decide(self, box: Box, outcome: str, via: str, res: GateResult, reason: str, **extra) -> None:
        self.cleared.append((box, self.cam.t + self.cfg.cooldown_s, outcome))
        self._emit("decision", outcome=outcome, via=via, p=round(res.p, 3), reason=reason, box=box, **extra)

    # -- main loop ----------------------------------------------------------
    def run(self) -> Iterator[Tick]:
        cam, cfg = self.cam, self.cfg
        cands: list[Candidate] = []
        last_check = -1e9
        n = 0
        multi = cfg.frames > 1
        step = max(1, int(round(cfg.frame_gap_s * cam.fps)))
        recent: deque = deque(maxlen=step * (cfg.frames - 1) + 1)  # wide frames since the camera got home
        while True:
            frame = cam.read()
            if frame is None:
                return
            recent.append(frame)
            if n % cfg.detect_every == 0:
                cands = self.detector.detect(frame)
            n += 1
            boxes = [(c.box, self._status(c.box) or "candidate") for c in cands]
            fresh = [c for c, (_, s) in zip(cands, boxes) if s == "candidate"][: cfg.max_candidates]

            warming_up = multi and len(recent) < recent.maxlen  # not enough history for a sequence yet
            if not fresh or warming_up or cam.t - last_check < cfg.check_interval_s:
                yield self._tick("watching", frame, boxes)
                continue

            # ---- tier 2: score each candidate on the wide view's own pixels ----
            last_check = cam.t
            yield self._tick("scoring", frame, boxes)
            unsure: list[tuple[Candidate, GateResult, int]] = []
            for c in fresh:
                if multi:  # the region a zoom would show, over the last few frames
                    region = cam.fov(policy.zoom_target(cam, c.box, cfg.crop_margin))
                    seq = [region_crop(f, region, (cam.wide_w, cam.wide_h)) for f in spaced(list(recent), step, cfg.frames)]
                    crop = seq[-1]
                    res, reading = self._read(seq)
                else:
                    crop = digital_crop(frame, c.box, cfg.crop_margin)
                    res, reading = self._read(crop)
                snap = self._snap(crop)
                self._emit(
                    "first_pass",
                    box=c.box,
                    snap=snap,
                    p=round(res.p, 3),
                    label=res.label,
                    uncertainty=round(res.uncertainty, 3),
                    reason=reading.reason,
                    latency_s=round(reading.latency_s, 2),
                )
                needs_look = res.label == "unsure" or cfg.mode == "always"
                if needs_look and cfg.mode != "passive":
                    unsure.append((c, res, snap))
                elif res.label == "unsure":  # passive: nothing better to do than guess
                    self._decide(c.box, "alarm" if res.p >= 0.5 else "reject", "guess", res, reading.reason, snap=snap)
                else:
                    outcome = "alarm" if res.label == "anomaly" else "reject"
                    self._decide(c.box, outcome, "wide", res, reading.reason, snap=snap)

            # ---- tier 3: look closer at the most informative unsure candidate ----
            plan = policy.choose(cam, [u[0].box for u in unsure], [u[1].uncertainty for u in unsure]) if unsure else None
            if unsure and plan is None:
                for c, res, snap in unsure:  # already fills the view: zooming cannot add detail
                    self._decide(c.box, "escalate", "wide", res, "Still unsure and too close to zoom further.", snap=snap)
            if plan is None:
                yield self._tick("watching", frame, boxes)
                continue

            cand, before, before_snap = unsure[plan.index]
            for c, res, _ in unsure:
                if c is not cand:  # left for the next round; the camera can only look at one place
                    self._emit("deferred", box=c.box, p=round(res.p, 3))
            self._emit("move", box=cand.box, zoom=round(plan.target.zoom, 2), score=round(plan.score, 3))
            context = frame.copy()
            t_start = cam.t
            missed = 0
            for view, _ in cam.move_iter(plan.target):
                yield self._tick("moving", view, [(cand.box, "target")])
            t_move = cam.t - t_start

            view, views = None, []
            n_dwell = max(1, int(cfg.dwell_s * cam.fps))
            if multi:  # stay long enough to collect the whole sequence
                n_dwell = max(n_dwell, step * (cfg.frames - 1) + 1)
            for _ in range(n_dwell):
                view = cam.read()
                if view is None:
                    return
                views.append(view)
                missed = max(missed, self._count_unseen(cam, n))
                n += 1
                yield self._tick("verifying", view, [(cand.box, "target")])

            # ---- tier 4: verify on the new pixels, with the wide view as context ----
            after, reading = self._read(spaced(views, step, cfg.frames) if multi else view, context)
            if cfg.paired_fixed_camera:
                missed = 0
            r = policy.reward(before.uncertainty, after.uncertainty, t_move, missed, cfg.lam, cfg.mu)
            outcome = {"anomaly": "alarm", "normal": "reject", "unsure": "escalate"}[after.label]
            self._decide(
                cand.box,
                outcome,
                "zoom",
                after,
                reading.reason,
                snap=self._snap(view),
                snap_before=before_snap,
                p_before=round(before.p, 3),
                u_before=round(before.uncertainty, 3),
                u_after=round(after.uncertainty, 3),
                t_move=round(t_move, 2),
                n_missed=missed,
                reward=round(r, 3),
                latency_s=round(reading.latency_s, 2),
            )
            yield self._tick("verifying", view, [(cand.box, outcome)])

            for view, _ in cam.move_iter(PTZState()):
                yield self._tick("returning", view, [(cand.box, outcome)])
            self._emit("home", away_s=round(cam.t - t_start, 2))
            cands = []
            recent.clear()

    def _count_unseen(self, cam: SimPTZCamera, n: int) -> int:
        """Blind-spot proxy: undecided people the wide view would show, outside the zoomed view.

        Uses the oracle wide view, so this is a metric, never an input to a decision.
        """
        if n % self.cfg.detect_every:
            return 0
        x0, y0, x1, y1 = cam.fov()
        count = 0
        for c in self.detector.detect(cam.oracle_wide()):
            cx, cy = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1) and self._status(c.box) is None:
                count += 1
        return count


def summarize(log: list[dict]) -> dict:
    dec = [e for e in log if e["kind"] == "decision"]
    zooms = [e for e in dec if e["via"] == "zoom"]
    first = [e for e in log if e["kind"] == "first_pass"]
    return {
        "candidates_scored": len(first),
        "decisions": len(dec),
        "alarms": sum(e["outcome"] == "alarm" for e in dec),
        "rejected": sum(e["outcome"] == "reject" for e in dec),
        "escalated": sum(e["outcome"] == "escalate" for e in dec),
        "guessed": sum(e["via"] == "guess" for e in dec),
        "zooms": len(zooms),
        "zoom_rate": round(len(zooms) / len(first), 3) if first else 0.0,
        "mean_uncertainty_drop": round(float(np.mean([e["u_before"] - e["u_after"] for e in zooms])), 3) if zooms else None,
        "mean_reward": round(float(np.mean([e["reward"] for e in zooms])), 3) if zooms else None,
        "camera_seconds_away": round(sum(e["away_s"] for e in log if e["kind"] == "home"), 2),
        "unseen_candidates": sum(e["n_missed"] for e in zooms),
        "mean_vlm_latency_s": round(float(np.mean([e["latency_s"] for e in first])), 2) if first else None,
    }
