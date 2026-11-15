"""Offline evaluation on clips with frame-level labels (CHAD layout).

    python -m lookcloser.evaluate collect data/chad --out runs/chad_scores.csv
    python -m lookcloser.evaluate report runs/chad_scores.csv --save-gate gate.json

`collect` samples frames, and for each one scores the candidates twice: on an
enlarged crop of the wide view, and on a zoomed view of the same region. A frame's
score is the maximum over its candidates, compared with the frame's 0/1 label.

`report` cross-fits by clip: every frame is judged by gates fitted on other clips.

Two simplifications to keep in mind when reading the numbers:
  * the zoom is instantaneous here (same frame), so motor travel and the blind spot
    are not charged; the live pipeline charges them;
  * the label is per frame, not per person, so a frame with a fight in one corner
    labels every candidate in it.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from . import policy
from .calibration import Gate, ece, recall_at_far, sigmoid
from .pipeline import digital_crop, region_crop
from .sim_camera import PTZState, SimPTZCamera

FIELDS = ["clip", "frame", "label", "n_candidates", "logit_wide", "logit_zoom", "zoom"]


def collect(
    root: str,
    out: str,
    every_s: float,
    max_candidates: int,
    wide_size: tuple[int, int] = (640, 360),
    frames: int = 1,
    gap_s: float = 0.4,
) -> None:
    """Score sampled frames. With `frames` > 1 each look is a short sequence ending at the
    sampled frame, and the wide crop covers the same region as the zoom (fewer pixels)."""
    from .detector import PersonDetector
    from .verifier import MLXVerifier

    detector, verifier = PersonDetector(), MLXVerifier()
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {r["clip"] for r in csv.DictReader(f)}
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            w.writeheader()
        for path in sorted(glob.glob(os.path.join(root, "videos", "*.mp4"))):
            clip = os.path.splitext(os.path.basename(path))[0]
            if clip in done:
                continue
            labels = np.load(os.path.join(root, "labels", clip + ".npy"), allow_pickle=False)
            cam = SimPTZCamera(path, wide_size=wide_size)
            every = int(round(every_s * cam.fps))
            step = max(1, int(round(gap_s * cam.fps)))
            want = {i * step for i in range(frames)}  # offsets before the sampled frame to keep
            held: dict[int, np.ndarray] = {}
            rows = 0
            while True:
                frame = cam.read()
                if frame is None or cam.frame_idx >= len(labels):
                    break
                to_sample = -cam.frame_idx % every  # frames until the next sampled one
                if to_sample in want:
                    held[to_sample] = cam.hold()
                if to_sample or len(held) < frames:
                    if not to_sample:
                        held.clear()  # the clip's first sample has no history
                    continue
                history = [held[k] for k in sorted(held, reverse=True)]  # oldest first
                held.clear()
                # Groups first, then the largest: the camera would look there first.
                cands = sorted(detector.detect(frame), key=lambda c: (-c.n_people, -(c.box[3] - c.box[1])))
                wide, zoomed, zooms = [], [], []
                for c in cands[:max_candidates]:
                    target = policy.zoom_target(cam, c.box)
                    zooms.append(target.zoom)
                    if frames == 1:
                        wide.append(verifier.read(digital_crop(frame, c.box, 0.4)).logit)
                        zoomed.append(verifier.read(cam.render_held(history[0], target), frame).logit)
                    else:
                        region = cam.fov(target)
                        seq = [region_crop(cam.render_held(h, PTZState()), region, wide_size) for h in history]
                        wide.append(verifier.read(seq, None, gap_s).logit)
                        seq = [cam.render_held(h, target) for h in history]
                        zoomed.append(verifier.read(seq, frame, gap_s).logit)
                if wide:
                    w.writerow(
                        dict(
                            clip=clip,
                            frame=cam.frame_idx,
                            label=int(labels[cam.frame_idx]),
                            n_candidates=len(cands),
                            logit_wide=round(max(wide), 4),
                            logit_zoom=round(max(zoomed), 4),
                            zoom=round(float(np.mean(zooms)), 2),
                        )
                    )
                    rows += 1
                    f.flush()
            cam.close()
            print(f"{clip}: {rows} scored frames", flush=True)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def _bootstrap_gain(by_clip: dict[str, list[dict]], n: int = 3000) -> tuple[float, float, float]:
    """Zoom-minus-wide AUC, resampling whole clips (frames within a clip are not independent)."""
    rng = np.random.default_rng(0)
    clips = sorted(by_clip)
    gains = []
    for _ in range(n):
        sel = [r for c in rng.choice(clips, len(clips)) for r in by_clip[c]]
        y = np.array([int(r["label"]) for r in sel])
        if y.min() == y.max():
            continue
        w = np.array([float(r["logit_wide"]) for r in sel])
        z = np.array([float(r["logit_zoom"]) for r in sel])
        gains.append(auc(z, y) - auc(w, y))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.array(gains) > 0))


def report(path: str, save_gate: str | None, alpha_miss: float, alpha_fa: float, folds: int = 5) -> None:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    clips = sorted(by_clip)

    def arr(rs, key):
        return np.array([float(r[key]) for r in rs])

    y = arr(rows, "label").astype(int)
    xw, xz = arr(rows, "logit_wide"), arr(rows, "logit_zoom")
    n_anom_clips = sum(any(r["label"] == "1" for r in by_clip[c]) for c in clips)
    print(f"{len(rows)} scored frames ({y.sum()} anomalous) from {len(clips)} clips ({n_anom_clips} with anomalies); mean zoom {arr(rows, 'zoom').mean():.1f}x\n")

    lo, hi, share = _bootstrap_gain(by_clip)
    print("Ranking quality, all clips (threshold-free, needs no calibration):")
    print(f"  {'':14s}{'AUC':>7s}{'recall@10%FAR':>15s}")
    print(f"  {'wide crop':14s}{auc(xw, y):7.3f}{recall_at_far(xw, y, 0.10):15.3f}")
    print(f"  {'zoomed view':14s}{auc(xz, y):7.3f}{recall_at_far(xz, y, 0.10):15.3f}")
    print(f"  AUC gain from zoom {auc(xz, y) - auc(xw, y):+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}], positive in {share:.0%} of resamples")

    # Cross-fit by clip: every frame is judged by gates that never saw its clip.
    # Anomaly and normal clips are dealt into folds separately so each fold has both.
    fold_of = {}
    for group in (sorted(c for c in clips if c.endswith("_1")), sorted(c for c in clips if not c.endswith("_1"))):
        for k, c in enumerate(group):
            fold_of[c] = k % folds
    pw, pz = np.zeros(len(rows)), np.zeros(len(rows))
    lw = np.empty(len(rows), dtype=object)
    f_id = np.array([fold_of[r["clip"]] for r in rows])
    for k in range(folds):
        tr, te = f_id != k, f_id == k
        gw = Gate.fit(xw[tr], y[tr], alpha_miss, alpha_fa)
        gz = Gate.fit(xz[tr], y[tr], alpha_miss, alpha_fa)
        pw[te] = [gw.prob(v) for v in xw[te]]
        pz[te] = [gz.prob(v) for v in xz[te]]
        lw[te] = [gw.decide(v).label for v in xw[te]]

    print(f"\nCalibration error, {folds}-fold by clip (ECE, lower is better):")
    print(f"  wide crop    raw {ece(sigmoid(xw), y):.3f}  ->  calibrated {ece(pw, y):.3f}")
    print(f"  zoomed view  raw {ece(sigmoid(xz), y):.3f}  ->  calibrated {ece(pz, y):.3f}")

    def outcome(name: str, alarm: np.ndarray, zoomed: np.ndarray) -> None:
        print(f"  {name:38s}{alarm[y == 1].mean():8.3f}{alarm[y == 0].mean():13.3f}{zoomed.mean():11.0%}")

    # Threshold "never zoom" at the base rate rather than 0.5: anomalies are the minority,
    # so a calibrated score rarely crosses 0.5 and that baseline would be a straw man.
    base = y.mean()
    unsure = lw == "unsure"
    never = np.zeros(len(y), bool)
    print(f"\nDecisions, {folds}-fold by clip (gate: miss bound {alpha_miss:.0%}, false-alarm bound {alpha_fa:.0%}):")
    print(f"  {'':38s}{'recall':>8s}{'false alarms':>13s}{'zoom rate':>11s}")
    outcome("raw VLM, never zoom (logit > 0)", xw > 0, never)
    outcome(f"calibrated, never zoom (p > {base:.2f})", pw > base, never)
    outcome(f"always zoom (p_zoom > {base:.2f})", pz > base, ~never)
    outcome("zoom when unsure (this project)", np.where(unsure, pz > base, lw == "anomaly"), unsure)
    print(f"\n  gate on wide view: {np.mean(lw == 'normal'):.0%} normal, {np.mean(lw == 'anomaly'):.0%} anomaly, {unsure.mean():.0%} unsure")
    print(f"  anomalous frames dismissed without a look: {(lw[y == 1] == 'normal').mean():.1%} (bound {alpha_miss:.0%})")
    print(f"  normal frames alarmed without a look:      {(lw[y == 0] == 'anomaly').mean():.1%} (bound {alpha_fa:.0%})")

    if save_gate:  # the deployed gates use every clip
        Gate.fit(xw, y, alpha_miss, alpha_fa).save(save_gate)
        Gate.fit(xz, y, alpha_miss, alpha_fa).save(save_gate.replace(".json", "_zoom.json"))
        print(f"\nwrote {save_gate} and its _zoom twin, fitted on all {len(clips)} clips")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("root", help="folder with videos/*.mp4 and labels/*.npy")
    c.add_argument("--out", default="runs/chad_scores.csv")
    c.add_argument("--every", type=float, default=1.0, help="seconds between scored frames")
    c.add_argument("--max-candidates", type=int, default=3)
    c.add_argument("--wide", default="640x360", help="resolution the system sees; smaller simulates a farther scene")
    c.add_argument("--frames", type=int, default=1, help="frames per look; more than 1 lets the verifier see motion")
    c.add_argument("--gap", type=float, default=0.4, help="seconds between those frames")
    r = sub.add_parser("report")
    r.add_argument("scores")
    r.add_argument("--save-gate", default=None)
    r.add_argument("--alpha-miss", type=float, default=0.05)
    r.add_argument("--alpha-false-alarm", type=float, default=0.10)
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.root, args.out, args.every, args.max_candidates, tuple(int(v) for v in args.wide.split("x")), args.frames, args.gap)
    else:
        report(args.scores, args.save_gate, args.alpha_miss, args.alpha_false_alarm)


if __name__ == "__main__":
    main()
