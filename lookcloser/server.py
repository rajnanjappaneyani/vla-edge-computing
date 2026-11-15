"""Live demo server.

    python -m lookcloser.server            # then open http://localhost:8000

One worker thread owns the models and runs the pipeline; the page shows the scene,
what the PTZ sees, and the decisions as they happen. Playback pauses while the VLM
runs (its real latency is shown on each decision).
"""
from __future__ import annotations

import glob
import os
import queue
import threading
import time

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from .pipeline import Config, Pipeline, summarize
from .render import draw_view, draw_wide
from .run import build_gate, build_zoom_gate
from .sim_camera import SimPTZCamera
from .verifier import ANOMALY_SETS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class StartReq(BaseModel):
    clip: str
    mode: str = "active"
    paired: bool = False


class Session:
    """Shared between the worker thread (writes) and the HTTP handlers (read)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.commands: queue.Queue = queue.Queue()
        self.ready = False
        self.error: str | None = None
        self.verifier_name = ""
        self.run_id = 0
        self.running = False
        self.info: dict = {}
        self.status: dict = {"phase": "idle"}
        self.events: list[dict] = []
        self.snaps: list = []
        self.jpeg = {"wide": b"", "view": b""}
        self.frame_no = 0

    def worker(self) -> None:
        try:
            from .detector import PersonDetector
            from .verifier import MLXVerifier

            detector, verifier = PersonDetector(), MLXVerifier()
            self.verifier_name = verifier.name
            self.ready = True
        except Exception as e:  # surfaced on the page rather than lost in a thread
            self.error = f"{type(e).__name__}: {e}"
            return
        req = self.commands.get()
        while True:
            req = self._play(req, detector, verifier) or self.commands.get()

    def _play(self, req: StartReq, detector, verifier) -> StartReq | None:
        """Run one clip. Returns the next request if it was interrupted by one."""
        cam = SimPTZCamera(os.path.join(DATA, req.clip))
        # A dataset folder under data/ can carry its own calibration and anomaly definition.
        folder = req.clip.split(os.sep)[0] if os.sep in req.clip else ""
        gate_path = os.path.join(DATA, folder, "gate.json") if folder else os.path.join(ROOT, "gate.json")
        gate = build_gate(gate_path)
        verifier.anomaly_set = folder if folder in ANOMALY_SETS else "general"
        cfg = Config(mode=req.mode, paired_fixed_camera=req.paired)
        pipe = Pipeline(cam, detector, verifier, gate, cfg, build_zoom_gate(gate_path))
        with self.lock:
            self.run_id += 1
            self.running = True
            self.events, self.snaps = pipe.log, pipe.snaps
            self.info = {
                "clip": req.clip,
                "mode": req.mode,
                "paired": req.paired,
                "native": f"{cam.native_w}×{cam.native_h}",
                "wide": f"{cam.wide_w}×{cam.wide_h}",
                "max_zoom": round(cam.max_zoom, 1),
                "calibrated": gate.calibrated,
                "verifier": verifier.name,
                "anomalies": verifier.anomaly_set,
            }
        wall0, scene0 = time.time(), None
        for tick in pipe.run():
            if not self.commands.empty():
                cam.close()
                return self.commands.get()
            enc = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            wide = cv2.imencode(".jpg", draw_wide(tick), enc)[1].tobytes()
            view = cv2.imencode(".jpg", draw_view(tick), enc)[1].tobytes()
            with self.lock:
                self.jpeg = {"wide": wide, "view": view}
                self.frame_no += 1
                self.status = {
                    "phase": tick.phase,
                    "t": round(tick.t, 1),
                    "zoom": round(tick.zoom, 2),
                    "wide_is_seen": tick.wide_is_seen,
                }
            # Pace to scene time; a blocking VLM call simply resets the clock.
            if scene0 is None or tick.phase in ("scoring", "verifying"):
                wall0, scene0 = time.time(), tick.t
            delay = (tick.t - scene0) - (time.time() - wall0)
            if delay > 0:
                time.sleep(min(delay, 0.2))
        cam.close()
        with self.lock:
            self.running = False
            self.status = {**self.status, "phase": "finished"}
        return None


session = Session()
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


@app.get("/api/clips")
def clips():
    files = sorted(f for f in glob.glob(os.path.join(DATA, "**", "*"), recursive=True) if f.lower().endswith(VIDEO_EXT))
    return [os.path.relpath(f, DATA) for f in files]


@app.post("/api/start")
def start(req: StartReq):
    if req.mode not in ("active", "passive", "always"):
        raise HTTPException(400, "unknown mode")
    if req.clip not in clips():
        raise HTTPException(404, "clip not found in data/")
    session.commands.put(req)
    return {"ok": True}


@app.get("/api/state")
def state(since: int = 0):
    with session.lock:
        events = session.events[since:]
        return {
            "ready": session.ready,
            "error": session.error,
            "running": session.running,
            "run_id": session.run_id,
            "info": session.info,
            "status": session.status,
            "events": events,
            "n_events": len(session.events),
            "summary": summarize(session.events) if session.events else None,
        }


@app.get("/api/snap/{run_id}/{idx}")
def snap(run_id: int, idx: int):
    with session.lock:
        if run_id != session.run_id or not 0 <= idx < len(session.snaps):
            raise HTTPException(404)
        img = session.snaps[idx]
    return Response(cv2.imencode(".jpg", img)[1].tobytes(), media_type="image/jpeg")


def _mjpeg(key: str):
    last = -1
    while True:
        with session.lock:
            n, data = session.frame_no, session.jpeg[key]
        if n != last and data:
            last = n
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        else:
            time.sleep(0.01)


@app.get("/stream/{key}")
def stream(key: str):
    if key not in ("wide", "view"):
        raise HTTPException(404)
    return StreamingResponse(_mjpeg(key), media_type="multipart/x-mixed-replace; boundary=frame")


def main() -> None:
    threading.Thread(target=session.worker, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), log_level="warning")


if __name__ == "__main__":
    main()
