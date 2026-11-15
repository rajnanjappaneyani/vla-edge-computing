import cv2
import numpy as np
import pytest

from lookcloser.calibration import Gate, ece, entropy_bits, fit_platt, recall_at_far, sigmoid
from lookcloser.detector import group_boxes
from lookcloser.sim_camera import MotionModel, PTZState, SimPTZCamera


@pytest.fixture
def clip(tmp_path):
    """2 s of 1920x1080 with a fine checkerboard that the wide view cannot resolve."""
    path = str(tmp_path / "clip.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    frame = np.full((1080, 1920, 3), 128, np.uint8)
    yy, xx = np.mgrid[0:120, 0:120]
    frame[480:600, 900:1020] = (((xx // 2 + yy // 2) % 2) * 255)[..., None]
    for _ in range(50):
        vw.write(frame)
    vw.release()
    return path


def test_zoom_reveals_detail_the_wide_view_lacks(clip):
    cam = SimPTZCamera(clip)
    assert cam.max_zoom == pytest.approx(3.0)
    wide = cam.read()
    assert wide.shape == (360, 640, 3)
    # Digital crop of the wide view around the pattern, vs an optical-style zoom on it.
    digital = wide[160:200, 300:340]
    cam.move_to(PTZState(cx=0.5, cy=0.5, zoom=3.0))
    zoomed = cam.view()
    assert zoomed.shape == wide.shape
    assert zoomed.std() > 2 * digital.std()


def test_moving_costs_scene_time_and_zoom_is_capped(clip):
    cam = SimPTZCamera(clip, motion=MotionModel(command_delay_s=0.1, pan_speed=1.0, zoom_speed=2.0, settle_s=0.3))
    cam.read()
    dt = cam.move_to(PTZState(cx=0.9, cy=0.5, zoom=10.0))
    assert cam.state.zoom == pytest.approx(3.0)  # capped at native / wide
    assert dt == pytest.approx(0.1 + 1.0 + 0.3)  # zoom 1->3 at 2x/s dominates the pan
    assert cam.t == pytest.approx(dt, abs=1 / cam.fps)
    x0, _, x1, _ = cam.fov()
    assert x1 <= 1.0 + 1e-9 and x1 - x0 == pytest.approx(1 / 3)


def test_view_scene_roundtrip(clip):
    cam = SimPTZCamera(clip)
    cam.read()
    cam.move_to(PTZState(cx=0.4, cy=0.6, zoom=2.0))
    box = (0.35, 0.55, 0.45, 0.65)
    assert cam.view_to_scene(cam.region_in_view(box)) == pytest.approx(box)


def test_grouping_merges_neighbours_only():
    boxes = [(0.10, 0.4, 0.14, 0.6), (0.15, 0.4, 0.19, 0.6), (0.80, 0.4, 0.84, 0.6)]
    groups = group_boxes(boxes, [0.9, 0.7, 0.5])
    assert [g.n_people for g in groups] == [2, 1]
    assert groups[0].box == pytest.approx((0.10, 0.4, 0.19, 0.6))


def test_platt_recovers_a_shifted_model():
    rng = np.random.default_rng(1)
    true_logit = rng.normal(0, 2, 4000)
    y = rng.random(4000) < sigmoid(true_logit)
    raw = 0.5 * true_logit - 1.5  # under-confident and biased towards "normal"
    a, b = fit_platt(raw, y)
    assert a == pytest.approx(2.0, rel=0.15) and b == pytest.approx(3.0, rel=0.15)
    assert ece(sigmoid(a * raw + b), y) < ece(sigmoid(raw), y) / 3


def test_conformal_gate_bounds_missed_anomalies():
    rng = np.random.default_rng(2)

    def sample(n):
        y = rng.random(n) < 0.3
        return rng.normal(np.where(y, 1.0, -2.0), 1.5), y

    gate = Gate.fit(*sample(2000), alpha_miss=0.05, alpha_false_alarm=0.10)
    logits, y = sample(20000)
    labels = np.array([gate.decide(v).label for v in logits])
    assert (labels[y] == "normal").mean() <= 0.07  # anomalies dismissed without a look
    assert (labels[~y] == "anomaly").mean() <= 0.13  # normals alarmed without a look
    assert 0.05 < (labels == "unsure").mean() < 0.95


def test_uncalibrated_band_and_metrics():
    gate = Gate.uncalibrated(0.25, 0.75)
    assert [gate.decide(v).label for v in (-3, 0, 3)] == ["normal", "unsure", "anomaly"]
    assert entropy_bits(0.5) == pytest.approx(1.0)
    p = np.array([0.9, 0.8, 0.3, 0.6, 0.2, 0.1])
    y = np.array([1, 1, 1, 0, 0, 0])
    assert recall_at_far(p, y, far=0.0) == pytest.approx(2 / 3)


def test_spaced_picks_oldest_first_and_tolerates_short_history():
    from lookcloser.pipeline import spaced

    assert spaced(list(range(10)), 4, 3) == [1, 5, 9]
    assert spaced([7, 8], 4, 3) == [8]
