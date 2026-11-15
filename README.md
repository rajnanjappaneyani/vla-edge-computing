# Look Closer Before You Alarm — laptop testbed

A simulated-PTZ version of the research plan that runs on one Mac: watch a low-resolution wide view, score each group of people with a small VLM, zoom only when the gate is unsure, verify on the close-up, then alarm, reject or escalate.

## Setup

Needs an Apple-silicon Mac (the verifier runs on MLX) and Python 3.12.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The verifier weights download on first run. The CHAD subset used here (56 clips and their frame labels, about 2.5 GB) is included under `data/chad/`; see `data/chad/NOTICE.md` for its source and licence. `yolo11n.pt` is the Ultralytics YOLO11n detector (AGPL-3.0).

## Run

```bash
.venv/bin/python -m lookcloser.server
```

Open http://localhost:8000, pick a clip from `data/`, choose a mode and press Start.

```bash
.venv/bin/python -m lookcloser.run data/clip.mp4 --mode active
```

Writes a side-by-side video and an event log to `runs/`. Modes: `active` (zoom when unsure), `passive` (never zoom), `always` (rule-based: zoom on everything). Add `--paired` for the fixed camera + PTZ setup.

```bash
.venv/bin/python -m lookcloser.calibrate labels.csv
```

Fits the gate from labelled clips (`path,label` per line, 1 = anomaly) and writes `gate.json`, which the server and CLI pick up automatically.

```bash
.venv/bin/python -m pytest tests -q
```

## Clips

Put 1080p or 4K clips in `data/`. The system only sees a 640×360 downsample; zoom reads the original pixels, capped at native width ÷ 640 (3× for 1080p, 6× for 4K). `data/smoke.mp4` is a synthetic plumbing test made by `scripts/make_smoke_clip.py`, not evidence of anything.

## CHAD subset

`data/chad/` holds 56 clips from the 1080p cameras of the [CHAD dataset](https://github.com/TeCSAR-UNCC/CHAD) (32 with anomalies, 24 without) and their frame labels. A dataset folder under `data/` can carry its own `gate.json` / `gate_zoom.json`, and uses the anomaly list of the same name in `lookcloser/verifier.py` if there is one.

```bash
LOOKCLOSER_ANOMALIES=chad .venv/bin/python -m lookcloser.evaluate collect data/chad --out runs/chad_scores.csv
```

```bash
.venv/bin/python -m lookcloser.evaluate report runs/chad_scores.csv --save-gate data/chad/gate.json
```

Add `--wide 320x180` to `collect` to simulate a scene twice as far away (6× zoom headroom). Add `--frames 3` to give the verifier three frames per look, 0.4 s apart, so it can see motion; the live pipeline does this by default (`Config.frames`), and `data/chad/gate.json` is fitted for it (`gate_1frame.json` keeps the single-frame gates).

Results on 55 clips (one clip has no detected people), about 800 scored frames, 154 anomalous, 640×360 wide view:

| Frames per look | AUC, wide crop | AUC, zoomed | Gain from zoom (95% interval over clips) |
| --- | --- | --- | --- |
| 1 | 0.807 | 0.784 | −0.023 (−0.075 to +0.029) |
| 3 | 0.822 | 0.834 | +0.012 (−0.013 to +0.038) |

On the same frames, three-frame looks raise the zoomed view's AUC by 0.053 (0.014 to 0.095) and the wide crop's by 0.015 (−0.024 to +0.056). At 320×180 (646 frames, 149 anomalous) one frame gives 0.758 for both views; three frames give 0.794 wide and 0.805 zoomed, a zoom gain of +0.011 (−0.031 to +0.056). More zoom headroom did not make the gain larger.

Decisions with three frames, 5-fold by clip, gate bounds 5% missed / 10% false alarms:

| Method | Recall | False alarms | Zoom rate |
| --- | --- | --- | --- |
| Raw VLM, never zoom | 0.909 | 0.532 | 0% |
| Calibrated, never zoom | 0.779 | 0.291 | 0% |
| Always zoom | 0.786 | 0.291 | 100% |
| Zoom when unsure | 0.799 | 0.302 | 55% |

Calibration works: expected calibration error falls from 0.39 to 0.03 on held-out clips, and the gate dismisses 2.6% of anomalous frames without a look (bound 5%); it alarms on 10.5% of normal frames (bound 10%). Seeing motion helps the close-up, but zooming is still not shown to beat the wide crop: the gain interval includes zero, and zoom-when-unsure matches always-zoom with about half the camera moves rather than beating never-zoom.

## MSAD crime clips

MSAD is licensed for private academic use and is not in this repository. With the dataset on disk, `python -I scripts/prepare_msad.py "<MSAD folder>"... --out data/msad` lays out the five crime categories (assault, fighting, robbery, shooting, vandalism) and a spread of normal test clips, with frame labels built from `anomaly_annotation.csv`.

```bash
LOOKCLOSER_ANOMALIES=msad .venv/bin/python -m lookcloser.evaluate collect data/msad --wide 320x180 --frames 3 --every 2 --out runs/msad_scores_far_mf.csv
```

Results on 103 clips (71 crime, 32 normal; 16 more had no detected people), 908 scored frames, 424 anomalous, 320×180 wide view, three frames per look:

| Frames | AUC, wide crop | AUC, zoomed | Gain from zoom (95% interval over clips) |
| --- | --- | --- | --- |
| All | 0.826 | 0.811 | −0.015 (−0.032 to +0.003) |
| Zoom under 1.3× (47% of frames) | 0.742 | 0.727 | −0.015 (−0.036 to +0.012) |
| Zoom 1.3× to 2.5× | 0.803 | 0.828 | +0.025 (−0.006 to +0.059) |
| Zoom 2.5× and over | 0.946 | 0.909 | −0.037 (−0.092 to −0.006) |

Zooming does not help on real crime footage either, and calibrated never-zoom (recall 0.743, false alarms 0.236) is slightly better than zoom-when-unsure (0.729, 0.273). Calibration still works (ECE 0.37 to 0.06). Nearly half the frames are close-range scenes where people already fill the view.

## Layout

| File | Role |
| --- | --- |
| `lookcloser/sim_camera.py` | Simulated PTZ: views, motor timing, blind spot |
| `lookcloser/detector.py` | YOLO person detector, grouped into candidates |
| `lookcloser/verifier.py` | Qwen3-VL-2B (4-bit, MLX): anomaly logit + one-sentence reason |
| `lookcloser/calibration.py` | Platt scaling, class-conditional conformal gate, ECE, recall at fixed false-alarm rate |
| `lookcloser/policy.py` | Hand-written zoom policy and the reward a learned VLA would train on |
| `lookcloser/pipeline.py` | The loop and the three modes |
| `lookcloser/evaluate.py` | Offline scoring and report on clips with frame labels |
| `lookcloser/server.py`, `web/index.html` | Live demo |

## What is simulated

- Zoom comes from high-resolution video, not optics; motor timing is a model in `MotionModel`, not a measurement.
- The camera policy is a rule (most expected uncertainty reduction per second), not a trained VLA.
- "Unseen while zoomed" counts undecided people outside the zoomed view; it is a proxy until clips have event labels.
- Without `gate.json` the gate is a fixed 25–75% band and is labelled uncalibrated in the UI.
