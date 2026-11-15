"""Fit the gate on labelled clips and report calibration before and after.

    python -m lookcloser.calibrate labels.csv --out gate.json

labels.csv has one clip per line: `path,label` with label 1 = anomaly, 0 = normal.
Every scored candidate in a clip inherits the clip's label, so trim anomaly clips
to the part where the anomaly is visible. Use clips that the demo and the
evaluation will not use.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from .calibration import Gate, ece, recall_at_far, sigmoid
from .detector import PersonDetector
from .pipeline import digital_crop
from .sim_camera import SimPTZCamera
from .verifier import MLXVerifier


def collect(path: str, detector, verifier, every_s: float, max_per_clip: int) -> list[float]:
    cam = SimPTZCamera(path)
    logits: list[float] = []
    while len(logits) < max_per_clip:
        frame = cam.read()
        if frame is None:
            break
        for c in detector.detect(frame)[:2]:
            logits.append(verifier.read(digital_crop(frame, c.box, 0.4)).logit)
        if not cam.skip(every_s):
            break
    cam.close()
    return logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--out", default="gate.json")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between scored frames")
    ap.add_argument("--max-per-clip", type=int, default=6)
    ap.add_argument("--alpha-miss", type=float, default=0.05, help="bound on anomalies dismissed without a look")
    ap.add_argument("--alpha-false-alarm", type=float, default=0.10)
    args = ap.parse_args()

    detector, verifier = PersonDetector(), MLXVerifier()
    x: list[float] = []
    y: list[int] = []
    with open(args.labels) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            logits = collect(row[0].strip(), detector, verifier, args.every, args.max_per_clip)
            x += logits
            y += [int(row[1])] * len(logits)
            print(f"{row[0]}: {len(logits)} candidates, label {row[1]}")

    xa, ya = np.array(x), np.array(y)
    gate = Gate.fit(xa, ya, args.alpha_miss, args.alpha_false_alarm)
    gate.save(args.out)
    raw, cal = sigmoid(xa), sigmoid(gate.a * xa + gate.b)
    labels = [gate.decide(v).label for v in xa]
    print(f"\n{len(xa)} candidates ({int(ya.sum())} anomaly, {int((1 - ya).sum())} normal)")
    print(f"ECE            raw {ece(raw, ya):.3f}  ->  calibrated {ece(cal, ya):.3f}   (in-sample: optimistic)")
    print(f"recall @5% FAR raw {recall_at_far(raw, ya):.3f}      (ranking is unchanged by Platt scaling)")
    print(f"gate labels    unsure {labels.count('unsure') / len(labels):.0%}, anomaly {labels.count('anomaly') / len(labels):.0%}, normal {labels.count('normal') / len(labels):.0%}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
