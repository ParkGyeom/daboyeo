"""edge/blackbox/detect.py: daboyeo_ai 결과를 서버로 보낼 모양으로 바꾸는지 확인 (모델 파일 없이)."""

import numpy as np

from edge.blackbox import detect


class FakeOnnx:
    def __init__(self, *a, **k):
        FakeOnnx.kwargs = k

    def raw(self, frame):
        # (class_id, conf, x1, y1, x2, y2) 픽셀 좌표. 0 pothole, 1 crack, 2 guardrail, 3 sign
        return [(1, 0.4, 10.2, 20.7, 50.4, 60.6), (2, 0.9, 0, 0, 5, 5), (0, 0.8, -3.0, 5.0, 130.0, 90.0)]


def test_detector_returns_all_classes_sorted_and_clipped(monkeypatch):
    monkeypatch.setattr(detect, "OnnxDetector", FakeOnnx)
    dets = detect.Detector("x.onnx").detect(np.zeros((100, 120, 3), np.uint8))
    assert [d.type for d in dets] == ["guardrail", "pothole", "crack"]  # 신뢰도 순
    assert dets[1].bbox == (0, 5, 120, 90)  # 프레임 밖은 잘린다
    assert dets[2].bbox == (10, 20, 50, 61)
    assert [d.type for d in dets if d.type in detect.SENT_TYPES] == ["pothole", "crack"]  # 시설물은 안 보낸다


def test_detector_passes_class_conf(monkeypatch):
    monkeypatch.setattr(detect, "OnnxDetector", FakeOnnx)
    conf = {"pothole": 0.27, "crack": 0.29}
    detect.Detector("x.onnx", conf=conf, threads=2)
    assert FakeOnnx.kwargs == {"threads": 2, "conf": conf, "windows": None, "tile_classes": None}  # 조각 추론 끔 (기본)
    detect.Detector("x.onnx", conf=conf, windows=[(0, 0, 1, 1)], tile_classes=("crack",))
    assert FakeOnnx.kwargs["windows"] == [(0, 0, 1, 1)] and FakeOnnx.kwargs["tile_classes"] == ("crack",)


class FakeMeter:
    def measure(self, frame):
        return "r"

    def roi_polygon_px(self, w, h):
        return np.array([[10, 40], [90, 40], [99, 70], [0, 70]], np.int32)


class FakeJudgement:
    def __init__(self, confirmed):
        self.confirmed, self.confidence = confirmed, 0.7


class FakeTracker:
    def __init__(self, seq):
        self.seq, self.calls = list(seq), []

    def update(self, r, t=None):
        self.calls.append((r, t))
        return FakeJudgement(self.seq.pop(0))


def test_lane_wear_sends_only_confirmed_frames():
    img = np.zeros((80, 100, 3), np.uint8)
    tracker = FakeTracker([False, True, False])
    lw = detect.LaneWear(FakeMeter(), tracker)
    out = [lw.check(img, t=float(i)) for i in range(3)]
    assert out[0] is None and out[2] is None
    assert (out[1].type, out[1].confidence, out[1].bbox) == ("lane_wear", 0.7, (0, 40, 100, 71))
    assert tracker.calls == [("r", 0.0), ("r", 1.0), ("r", 2.0)]  # 측정 결과와 시각을 그대로 넘긴다


def test_lane_wear_with_real_tracker_needs_warmup():
    # 진짜 LaneWearMeter·LaneWearTracker: 기준을 잡기 전(warmup)에는 확정하지 않는다
    img = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
    lw = detect.LaneWear()
    assert all(lw.check(img, t=float(i)) is None for i in range(5))
