"""Drawing: overlays for the wide view and a side-by-side frame for recorded runs."""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import Tick

# BGR
COLORS = {
    "candidate": (0, 200, 255),
    "target": (255, 200, 0),
    "alarm": (60, 60, 235),
    "reject": (90, 190, 90),
    "escalate": (0, 140, 255),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_wide(tick: Tick, scale: int = 2) -> np.ndarray:
    """Wide view with candidate boxes and the PTZ's current field of view."""
    img = cv2.resize(tick.wide, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = img.shape[:2]
    if tick.zoom > 1.001:
        x0, y0, x1, y1 = tick.fov
        p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
        if not tick.wide_is_seen:  # dim what the camera is not looking at
            dim = (img * 0.35).astype(np.uint8)
            dim[p0[1] : p1[1], p0[0] : p1[0]] = img[p0[1] : p1[1], p0[0] : p1[0]]
            img = dim
        cv2.rectangle(img, p0, p1, (255, 255, 255), 2)
    for box, status in tick.boxes:
        color = COLORS.get(status, COLORS["candidate"])
        p0, p1 = (int(box[0] * w), int(box[1] * h)), (int(box[2] * w), int(box[3] * h))
        cv2.rectangle(img, p0, p1, color, 2)
        if status != "candidate":
            cv2.putText(img, status, (p0[0], max(14, p0[1] - 6)), FONT, 0.5, color, 2, cv2.LINE_AA)
    return img


def draw_view(tick: Tick, scale: int = 2) -> np.ndarray:
    return cv2.resize(tick.view, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if len(cur) + len(word) + 1 > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + [cur] if cur else lines


def compose(tick: Tick, last_decision: dict | None, scale: int = 1) -> np.ndarray:
    """Side-by-side frame: scene on the left, the PTZ's view on the right, status below."""
    left, right = draw_wide(tick, scale), draw_view(tick, scale)
    h, w = left.shape[:2]
    bar = np.full((70, 2 * w + 8, 3), 24, np.uint8)
    gap = np.full((h, 8, 3), 24, np.uint8)
    seen = "" if tick.wide_is_seen else "  (rest of scene unseen)"
    cv2.putText(bar, f"t={tick.t:5.1f}s  {tick.phase}  zoom {tick.zoom:.1f}x{seen}", (10, 20), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    if last_decision:
        d = last_decision
        color = COLORS.get(d["outcome"], (230, 230, 230))
        head = f"{d['outcome'].upper()} via {d['via']}  p={d['p']:.2f}"
        cv2.putText(bar, head, (10, 42), FONT, 0.5, color, 1, cv2.LINE_AA)
        for i, line in enumerate(_wrap(d["reason"], 150)[:1]):
            cv2.putText(bar, line, (10, 62 + 18 * i), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([np.hstack([left, gap, right]), bar])
