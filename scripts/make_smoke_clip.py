"""Build a synthetic 1080p clip for smoke tests: two small photo insets on a plain scene.

This only checks that the plumbing works (people are small in the wide view and
sharp when zoomed). It is not evidence about anomaly detection; use real footage
for that.
"""
import sys

import cv2
import numpy as np
from ultralytics.utils import ASSETS

out = sys.argv[1] if len(sys.argv) > 1 else "data/smoke.mp4"
W, H, FPS, SECONDS = 1920, 1080, 25, 14

yy = np.linspace(95, 150, H, dtype=np.float32)[:, None]
bg = np.repeat(np.repeat(yy, W, axis=1)[..., None], 3, axis=2).astype(np.uint8)

bus = cv2.imread(str(ASSETS / "bus.jpg"))  # pedestrians beside a bus
bus = cv2.resize(bus, (255, 340), interpolation=cv2.INTER_AREA)
zid = cv2.imread(str(ASSETS / "zidane.jpg"))  # two men, close-up
zid = cv2.resize(zid, (400, 225), interpolation=cv2.INTER_AREA)

vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
for i in range(FPS * SECONDS):
    frame = bg.copy()
    dx = int(6 * np.sin(i / 9))  # slight sway so frames differ
    frame[520:860, 250 + dx : 505 + dx] = bus
    if i > FPS * 5:  # second group appears later, far from the first
        frame[300:525, 1350:1750] = zid
    vw.write(frame)
vw.release()
print("wrote", out)
