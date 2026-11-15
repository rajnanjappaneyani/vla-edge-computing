"""Lay MSAD out like data/chad: videos/<clip>.mp4 (symlinks) and labels/<clip>.npy (0/1 per frame).

    python -I scripts/prepare_msad.py "<MSAD root>"... --out data/msad

Takes the crime categories and a spread of normal test clips, skipping videos too small to zoom into.
MSAD is licensed for private academic use: data/msad is git-ignored and must stay that way.
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

CRIME = ("Assault", "Fighting", "Robbery", "Shooting", "Vandalism")

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--out", default="data/msad")
ap.add_argument("--min-width", type=int, default=1280)
ap.add_argument("--normal-minutes", type=float, default=18.0)
args = ap.parse_args()

files, ann = {}, {}
for root in args.roots:
    for p in glob.glob(os.path.join(root, "**", "*.mp4"), recursive=True):
        files[os.path.splitext(os.path.basename(p))[0]] = os.path.abspath(p)
    for p in glob.glob(os.path.join(root, "**", "anomaly_annotation.csv"), recursive=True):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                ann[r["name"].strip()] = r

os.makedirs(os.path.join(args.out, "videos"), exist_ok=True)
os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)


def probe(path):
    c = cv2.VideoCapture(path)
    out = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_COUNT)), c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()
    return out


def add(name, labels):
    link = os.path.join(args.out, "videos", name + ".mp4")
    if not os.path.lexists(link):
        os.symlink(files[name], link)
    np.save(os.path.join(args.out, "labels", name + ".npy"), labels)


n_anom = n_norm = skipped = mismatch = 0
anom_s = norm_s = 0.0
for name in sorted(files):
    if not name.startswith(CRIME) or name not in ann:
        continue
    w, n, fps = probe(files[name])
    if w < args.min_width:
        skipped += 1
        continue
    r = ann[name]
    start, end = int(r["starting frame of anomaly"]), int(r["ending frame of anomaly"])
    if abs(int(r["total frames"]) - n) > 5:
        mismatch += 1
    labels = np.zeros(n, dtype=np.uint8)
    labels[max(start - 1, 0) : min(end, n)] = 1  # annotation frames are 1-based, inclusive
    add(name, labels)
    n_anom += 1
    anom_s += n / fps

# Normal test clips, taken evenly across the numbering so scenes are mixed.
normals = sorted((k for k in files if k.startswith("MSAD_normal_testing_")), key=lambda k: int(k.rsplit("_", 1)[1]))
order = normals[0::3] + normals[1::3] + normals[2::3]
for name in order:
    if norm_s >= args.normal_minutes * 60:
        break
    w, n, fps = probe(files[name])
    if w < args.min_width:
        continue
    add(name, np.zeros(n, dtype=np.uint8))
    n_norm += 1
    norm_s += n / fps

print(f"anomaly clips {n_anom} ({anom_s / 60:.1f} min), skipped as too small {skipped}, frame-count mismatches {mismatch}")
print(f"normal clips {n_norm} ({norm_s / 60:.1f} min)")
