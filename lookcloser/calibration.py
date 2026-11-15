"""Turn the verifier's raw yes/no logit into a calibrated three-way gate.

Two pieces, fitted on a held-out labelled set:

1. Platt scaling, p = sigmoid(a * logit + b). The bias b matters here: zero-shot
   VLMs lean towards "normal", and a temperature alone cannot shift that.
2. Class-conditional (Mondrian) split conformal prediction. Each class gets its
   own threshold, so the miss rate on true anomalies is bounded by `alpha_miss`
   independently of how rare anomalies are. The prediction set is {normal},
   {anomaly}, or both; both (or neither) means "unsure", which triggers a zoom.

Without calibration data the gate falls back to a fixed probability band and
reports itself as uncalibrated.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -50, 50)))


def entropy_bits(p: float) -> float:
    """Binary entropy: 0 = certain, 1 = coin flip. This is the U in the reward."""
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error of P(anomaly) against 0/1 labels."""
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def recall_at_far(p, y, far: float = 0.05) -> float:
    """Recall on anomalies at the threshold whose false-alarm rate on normals is <= far."""
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(bool)
    neg = np.sort(p[~y])[::-1]
    if len(neg) == 0 or not y.any():
        return float("nan")
    k = int(math.floor(far * len(neg)))  # normals allowed above the threshold
    thr = neg[k] if k < len(neg) else -np.inf
    return float((p[y] > thr).mean())


def fit_platt(logits, y, iters: int = 100) -> tuple[float, float]:
    """Fit p = sigmoid(a * logit + b) by damped Newton steps on the log-loss.

    Uses Platt's smoothed targets instead of hard 0/1 labels, which keeps the fit
    finite when a small calibration set happens to be perfectly separable.
    """
    x, y = np.asarray(logits, dtype=float), np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    t = np.where(y, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def loss(a: float, b: float) -> float:
        p = np.clip(sigmoid(a * x + b), 1e-12, 1 - 1e-12)
        return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())

    a, b = 1.0, 0.0
    cur = loss(a, b)
    for _ in range(iters):
        p = sigmoid(a * x + b)
        g = np.array([((p - t) * x).mean(), (p - t).mean()])
        w = p * (1 - p)
        h = np.array([[(w * x * x).mean(), (w * x).mean()], [(w * x).mean(), w.mean()]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g)
        scale = 1.0
        while scale > 1e-6 and loss(a - scale * step[0], b - scale * step[1]) > cur:
            scale /= 2  # backtrack: a full Newton step can overshoot
        a, b = a - scale * step[0], b - scale * step[1]
        new = loss(a, b)
        if cur - new < 1e-10:
            break
        cur = new
    return float(a), float(b)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return 1.0  # too few examples for this alpha: never exclude the class
    return float(np.sort(scores)[k - 1])


@dataclass
class GateResult:
    p: float  # calibrated P(anomaly)
    label: str  # "normal" | "anomaly" | "unsure"
    uncertainty: float  # entropy in bits


@dataclass
class Gate:
    a: float = 1.0
    b: float = 0.0
    # Conformal thresholds on the nonconformity score 1 - p(class).
    q_anomaly: float = 0.75
    q_normal: float = 0.75
    calibrated: bool = False
    n_cal: int = 0
    alpha_miss: float | None = None
    alpha_false_alarm: float | None = None

    @classmethod
    def uncalibrated(cls, low: float = 0.25, high: float = 0.75) -> "Gate":
        """Raw confidence with a fixed band: unsure when low < p < high."""
        return cls(q_anomaly=1 - low, q_normal=high)

    @classmethod
    def fit(cls, logits, y, alpha_miss: float = 0.05, alpha_false_alarm: float = 0.10) -> "Gate":
        """Fit on held-out (logit, label) pairs; label 1 = anomaly.

        Platt and conformal should use disjoint data, so the set is split in half.
        """
        x, y = np.asarray(logits, dtype=float), np.asarray(y).astype(int)
        if min((y == 1).sum(), (y == 0).sum()) < 4:
            raise ValueError("need at least 4 examples of each class to calibrate")
        rng = np.random.default_rng(0)
        order = rng.permutation(len(x))
        fit_idx, conf_idx = order[::2], order[1::2]
        a, b = fit_platt(x[fit_idx], y[fit_idx])
        p = sigmoid(a * x[conf_idx] + b)
        yc = y[conf_idx]
        return cls(
            a=a,
            b=b,
            q_anomaly=_conformal_quantile(1 - p[yc == 1], alpha_miss),
            q_normal=_conformal_quantile(p[yc == 0], alpha_false_alarm),
            calibrated=True,
            n_cal=len(x),
            alpha_miss=alpha_miss,
            alpha_false_alarm=alpha_false_alarm,
        )

    def prob(self, logit: float) -> float:
        return float(sigmoid(self.a * logit + self.b))

    def decide(self, logit: float) -> GateResult:
        p = self.prob(logit)
        in_anomaly = (1 - p) <= self.q_anomaly
        in_normal = p <= self.q_normal
        if in_anomaly and not in_normal:
            label = "anomaly"
        elif in_normal and not in_anomaly:
            label = "normal"
        else:
            label = "unsure"
        return GateResult(p=p, label=label, uncertainty=entropy_bits(p))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Gate":
        with open(path) as f:
            return cls(**json.load(f))
