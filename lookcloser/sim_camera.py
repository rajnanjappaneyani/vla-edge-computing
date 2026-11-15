"""Simulated PTZ camera built on a high-resolution video.

The system only ever sees `wide_size` pixels per frame. At zoom 1 that is the
whole scene downsampled; at zoom z it is a (1/z) window of the full-resolution
frame resampled to the same size, so zooming reveals detail that was not in the
wide view. Zoom is capped at native_width / wide_width: past that point a crop
would only stretch pixels, which is exactly what optical zoom is not.

Moving costs time. `move_to` advances the video by the modelled travel time, so
an event can end before the camera arrives, and everything outside the zoomed
window is unseen until the camera returns.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class PTZState:
    cx: float = 0.5  # view centre, as a fraction of scene width
    cy: float = 0.5  # view centre, as a fraction of scene height
    zoom: float = 1.0

    @property
    def is_wide(self) -> bool:
        return self.zoom <= 1.0 + 1e-6


@dataclass(frozen=True)
class MotionModel:
    """Timing of the simulated motor. Replace with measured values from a real PTZ."""

    command_delay_s: float = 0.13  # command-to-motion latency
    pan_speed: float = 1.0  # scene widths per second
    zoom_speed: float = 2.0  # zoom factors per second
    settle_s: float = 0.3  # autofocus settle after zoom changes

    def travel_time(self, a: PTZState, b: PTZState) -> float:
        pan = max(abs(a.cx - b.cx), abs(a.cy - b.cy)) / self.pan_speed
        zoom = abs(a.zoom - b.zoom) / self.zoom_speed
        if pan < 1e-6 and zoom < 1e-6:
            return 0.0
        settle = self.settle_s if zoom > 1e-6 else 0.0
        return self.command_delay_s + max(pan, zoom) + settle


class SimPTZCamera:
    def __init__(
        self,
        source: str,
        wide_size: tuple[int, int] = (640, 360),
        motion: MotionModel | None = None,
        loop: bool = False,
    ):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {source}")
        self.source = source
        self.wide_w, self.wide_h = wide_size
        self.native_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.max_zoom = max(1.0, self.native_w / self.wide_w)
        self.motion = motion or MotionModel()
        self.loop = loop
        self.state = PTZState()
        self.frame_idx = -1
        self._full: np.ndarray | None = None

    # -- time ---------------------------------------------------------------
    @property
    def t(self) -> float:
        return max(self.frame_idx, 0) / self.fps

    def _advance(self) -> bool:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return False
        self._full = frame
        self.frame_idx += 1
        return True

    def skip(self, seconds: float) -> bool:
        """Let scene time pass without returning frames. False if the video ended."""
        for _ in range(int(round(seconds * self.fps))):
            if not self._advance():
                return False
        return True

    # -- views --------------------------------------------------------------
    def clamp(self, s: PTZState) -> PTZState:
        zoom = float(np.clip(s.zoom, 1.0, self.max_zoom))
        half = 0.5 / zoom
        return PTZState(
            cx=float(np.clip(s.cx, half, 1 - half)),
            cy=float(np.clip(s.cy, half, 1 - half)),
            zoom=zoom,
        )

    def fov(self, s: PTZState | None = None) -> tuple[float, float, float, float]:
        """Field of view as (x0, y0, x1, y1) in scene fractions."""
        s = self.clamp(s or self.state)
        half = 0.5 / s.zoom
        return (s.cx - half, s.cy - half, s.cx + half, s.cy + half)

    def _render(self, s: PTZState, full: np.ndarray | None = None) -> np.ndarray:
        full = self._full if full is None else full
        x0, y0, x1, y1 = self.fov(s)
        px0, px1 = int(round(x0 * self.native_w)), int(round(x1 * self.native_w))
        py0, py1 = int(round(y0 * self.native_h)), int(round(y1 * self.native_h))
        crop = full[py0:py1, px0:px1]
        return cv2.resize(crop, (self.wide_w, self.wide_h), interpolation=cv2.INTER_AREA)

    def read(self) -> np.ndarray | None:
        """Advance one frame and return what the camera sees in its current pose."""
        if not self._advance():
            return None
        return self._render(self.state)

    def view(self) -> np.ndarray:
        """Re-render the current frame in the current pose (no time passes)."""
        return self._render(self.state)

    def hold(self) -> np.ndarray:
        """The current full-resolution frame, to be re-rendered later with `render_held`.

        Offline evaluation only: it lets a sequence of past frames be viewed in a pose
        the camera was not in at the time, i.e. a zoom with no travel time.
        """
        return self._full

    def render_held(self, full: np.ndarray, s: PTZState) -> np.ndarray:
        return self._render(s, full)

    def oracle_wide(self) -> np.ndarray:
        """Wide view of the current frame regardless of pose.

        For metrics and the demo display only (what the camera is missing while
        zoomed). The decision pipeline must not read this while zoomed, unless
        it is modelling the fixed-camera + PTZ pair setup.
        """
        return self._render(PTZState())

    # -- control ------------------------------------------------------------
    def move_to(self, target: PTZState) -> float:
        """Drive to `target`; scene time advances by the travel time, which is returned."""
        target = self.clamp(target)
        dt = self.motion.travel_time(self.state, target)
        self.skip(dt)
        self.state = target
        return dt

    def move_iter(self, target: PTZState):
        """Like `move_to`, but yields (view, settled) for every frame of the journey.

        `settled` is False until the motor has stopped and focus has settled;
        those frames should be displayed but not used as evidence.
        """
        target = self.clamp(target)
        start = self.state
        m = self.motion
        total = m.travel_time(start, target)
        if total == 0.0:
            return
        settle = m.settle_s if abs(start.zoom - target.zoom) > 1e-6 else 0.0
        travel = total - m.command_delay_s - settle
        n = max(1, int(round(total * self.fps)))
        for i in range(1, n + 1):
            elapsed = i / self.fps
            a = float(np.clip((elapsed - m.command_delay_s) / travel, 0, 1)) if travel > 0 else 1.0
            self.state = PTZState(
                cx=start.cx + a * (target.cx - start.cx),
                cy=start.cy + a * (target.cy - start.cy),
                zoom=start.zoom + a * (target.zoom - start.zoom),
            )
            if not self._advance():
                break
            yield self._render(self.state), False
        self.state = target

    def home(self) -> float:
        return self.move_to(PTZState())

    def region_in_view(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert a scene-fraction box to fractions of the current view."""
        x0, y0, x1, y1 = self.fov()
        w, h = x1 - x0, y1 - y0
        return ((box[0] - x0) / w, (box[1] - y0) / h, (box[2] - x0) / w, (box[3] - y0) / h)

    def view_to_scene(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Convert a box in fractions of the current view to scene fractions."""
        x0, y0, x1, y1 = self.fov()
        w, h = x1 - x0, y1 - y0
        return (x0 + box[0] * w, y0 + box[1] * h, x0 + box[2] * w, y0 + box[3] * h)

    def close(self) -> None:
        self.cap.release()


def with_zoom(s: PTZState, zoom: float) -> PTZState:
    return replace(s, zoom=zoom)
