"""Where to look: a hand-written stand-in for the learned VLA camera policy.

It ranks unsure candidates by expected uncertainty reduction per second of
camera time. The trained policy (SFT, then GRPO) would replace `choose`, and
`reward` is the signal it would be trained on.
"""
from __future__ import annotations

from dataclasses import dataclass

from .detector import Box
from .sim_camera import PTZState, SimPTZCamera

MIN_USEFUL_ZOOM = 1.3  # below this a zoom adds too little detail to be worth the blind spot


def zoom_target(cam: SimPTZCamera, box: Box, margin: float = 0.4) -> PTZState:
    """Pose that fills the view with `box` (scene fractions) plus a margin of context."""
    bw, bh = (box[2] - box[0]) * (1 + 2 * margin), (box[3] - box[1]) * (1 + 2 * margin)
    zoom = 1.0 / max(bw, bh, 1e-6)
    return cam.clamp(PTZState(cx=(box[0] + box[2]) / 2, cy=(box[1] + box[3]) / 2, zoom=zoom))


@dataclass
class Plan:
    index: int  # which candidate
    target: PTZState
    score: float  # expected bits of uncertainty removed per second
    t_round_trip: float


def choose(cam: SimPTZCamera, boxes: list[Box], uncertainties: list[float]) -> Plan | None:
    """Pick the unsure candidate with the best expected information gain per second."""
    best: Plan | None = None
    for i, (box, u) in enumerate(zip(boxes, uncertainties)):
        target = zoom_target(cam, box)
        if target.zoom < MIN_USEFUL_ZOOM:
            continue
        # A zoom of z shows z^2 times the pixels; treat the uncovered fraction as the
        # share of the current uncertainty a closer look can be expected to remove.
        gain = u * (1 - 1 / target.zoom**2)
        t = cam.motion.travel_time(cam.state, target) + cam.motion.travel_time(target, PTZState())
        plan = Plan(index=i, target=target, score=gain / max(t, 1e-6), t_round_trip=t)
        if best is None or plan.score > best.score:
            best = plan
    return best


def reward(u_before: float, u_after: float, t_move: float, n_missed: int, lam: float = 0.1, mu: float = 0.5) -> float:
    """R = (U_before - U_after) - lambda * t_move - mu * n_missed."""
    return (u_before - u_after) - lam * t_move - mu * n_missed
