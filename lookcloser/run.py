"""Run the pipeline over a clip and write a side-by-side video plus an event log.

    python -m lookcloser.run clip.mp4 --mode active --out runs/clip_active
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2

from .calibration import Gate
from .detector import PersonDetector
from .pipeline import Config, Pipeline, summarize
from .render import compose
from .sim_camera import SimPTZCamera
from .verifier import MLXVerifier


def build_gate(path: str | None) -> Gate:
    if path and os.path.exists(path):
        return Gate.load(path)
    return Gate.uncalibrated()


def build_zoom_gate(path: str | None) -> Gate | None:
    """Separate calibration for close-ups, written by `evaluate report --save-gate`."""
    zoom_path = path.replace(".json", "_zoom.json") if path else None
    return Gate.load(zoom_path) if zoom_path and os.path.exists(zoom_path) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--mode", choices=["active", "passive", "always"], default="active")
    ap.add_argument("--out", default=None, help="output prefix; writes <out>.mp4 and <out>.json")
    ap.add_argument("--gate", default="gate.json", help="calibration file from lookcloser.calibrate")
    ap.add_argument("--wide", default="640x360", help="resolution the system is allowed to see")
    ap.add_argument("--paired", action="store_true", help="fixed camera + PTZ pair (no blind spot)")
    ap.add_argument("--charge-inference", action="store_true", help="scene time passes while the VLM runs")
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args()

    ww, wh = (int(v) for v in args.wide.split("x"))
    cam = SimPTZCamera(args.video, wide_size=(ww, wh))
    gate = build_gate(args.gate)
    cfg = Config(mode=args.mode, paired_fixed_camera=args.paired, charge_inference_time=args.charge_inference)
    pipe = Pipeline(cam, PersonDetector(), MLXVerifier(), gate, cfg, build_zoom_gate(args.gate))
    print(
        f"{args.video}: {cam.native_w}x{cam.native_h} @ {cam.fps:.0f} fps, wide view {ww}x{wh}, "
        f"max zoom {cam.max_zoom:.1f}x, gate {'calibrated' if gate.calibrated else 'UNCALIBRATED (fixed band)'}"
    )

    out = args.out or f"runs/{os.path.splitext(os.path.basename(args.video))[0]}_{args.mode}"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    writer, last_decision, t0 = None, None, time.time()
    for tick in pipe.run():
        for ev in tick.events:
            if ev["kind"] == "decision":
                last_decision = ev
            print(json.dumps(ev))
        frame = compose(tick, last_decision)
        if writer is None:
            writer = cv2.VideoWriter(f"{out}.mp4", cv2.VideoWriter_fourcc(*"avc1"), cam.fps, frame.shape[1::-1])
        writer.write(frame)
        if args.max_seconds and tick.t >= args.max_seconds:
            break
    if writer:
        writer.release()

    summary = summarize(pipe.log)
    summary.update(mode=args.mode, scene_seconds=round(cam.t, 1), wall_seconds=round(time.time() - t0, 1), gate_calibrated=gate.calibrated)
    with open(f"{out}.json", "w") as f:
        json.dump({"summary": summary, "events": pipe.log}, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}.mp4 and {out}.json")


if __name__ == "__main__":
    main()
