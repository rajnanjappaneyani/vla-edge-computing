"""Always-on tier: person detection on the low-resolution wide view, grouped into candidates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 as fractions of the view


@dataclass
class Candidate:
    box: Box  # union box of the group, fractions of the view it was detected in
    n_people: int
    det_conf: float  # mean detector confidence of the members


def _expand(b: Box, m: float) -> Box:
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - m * w, b[1] - m * h, b[2] + m * w, b[3] + m * h)


def _overlap(a: Box, b: Box) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def group_boxes(boxes: list[Box], confs: list[float], margin: float = 0.5) -> list[Candidate]:
    """Merge people standing close together: interactions are what need verifying."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(_expand(boxes[i], margin), _expand(boxes[j], margin)):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = []
    for members in groups.values():
        bs = np.array([boxes[i] for i in members])
        out.append(
            Candidate(
                box=(float(bs[:, 0].min()), float(bs[:, 1].min()), float(bs[:, 2].max()), float(bs[:, 3].max())),
                n_people=len(members),
                det_conf=float(np.mean([confs[i] for i in members])),
            )
        )
    return sorted(out, key=lambda c: c.box[0])


class PersonDetector:
    """YOLO person detector. A low confidence floor, because far-field people are tiny."""

    def __init__(self, weights: str = "yolo11n.pt", conf: float = 0.2, imgsz: int = 640):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> list[Candidate]:
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, classes=[0], conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        boxes = [(x0 / w, y0 / h, x1 / w, y1 / h) for x0, y0, x1, y1 in res.boxes.xyxy.tolist()]
        return group_boxes(boxes, res.boxes.conf.tolist())
