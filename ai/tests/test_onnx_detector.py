import numpy as np
import pytest

from daboyeo_ai.inference import decode_yolo, letterbox
from daboyeo_ai.inference.onnx_detector import Letterbox


def fake_output(boxes, nc=4, n_noise=50):
    """boxes: [(cls, score, cx, cy, w, h)] (640 입력 기준) → (1, 4+nc, N)"""
    rows = []
    for c, s, cx, cy, w, h in boxes:
        r = np.zeros(4 + nc, np.float32)
        r[:4] = cx, cy, w, h
        r[4 + c] = s
        rows.append(r)
    rng = np.random.default_rng(0)
    for _ in range(n_noise):  # 임계값 아래 잡음
        r = np.zeros(4 + nc, np.float32)
        r[:4] = rng.uniform(50, 600, 4)
        r[4 + rng.integers(nc)] = rng.uniform(0, 0.1)
        rows.append(r)
    return np.stack(rows).T[None]


def test_letterbox_wide_frame():
    img = np.zeros((720, 1280, 3), np.uint8)
    out, lb = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert lb.ratio == pytest.approx(0.5)
    assert (lb.pad_x, lb.pad_y) == (0, 140)
    assert out[0, 0].tolist() == [114, 114, 114] and out[320, 320].tolist() == [0, 0, 0]


def test_decode_maps_back_to_original_pixels():
    lb = Letterbox(ratio=0.5, pad_x=0, pad_y=140)  # 1280×720 → 640
    out = fake_output([(0, 0.9, 320, 320, 100, 50)])  # 입력 기준 중앙
    (c, s, x1, y1, x2, y2), = decode_yolo(out, lb, 1280, 720)
    assert c == 0 and s == pytest.approx(0.9)
    assert (x1, y1, x2, y2) == pytest.approx((540, 310, 740, 410))


def test_decode_nms_is_per_class():
    lb = Letterbox(1.0, 0, 0)
    out = fake_output([
        (0, 0.9, 300, 300, 100, 100),
        (0, 0.8, 305, 305, 100, 100),  # 같은 클래스 겹침 → 제거
        (1, 0.7, 305, 305, 100, 100),  # 다른 클래스 겹침 → 유지
    ])
    res = decode_yolo(out, lb, 640, 640, conf=0.25, iou=0.45)
    assert [(c, round(s, 2)) for c, s, *_ in res] == [(0, 0.9), (1, 0.7)]


def test_decode_clips_and_filters():
    lb = Letterbox(1.0, 0, 0)
    out = fake_output([(2, 0.6, 630, 10, 40, 40), (3, 0.2, 100, 100, 20, 20)])
    res = decode_yolo(out, lb, 640, 640, conf=0.25)
    assert len(res) == 1
    _, _, x1, y1, x2, y2 = res[0]
    assert x2 == 640 and y1 == 0


def test_decode_per_class_conf():
    lb = Letterbox(1.0, 0, 0)
    out = fake_output([
        (0, 0.5, 100, 100, 40, 40),  # pothole 0.5 < 0.6 → 버림
        (1, 0.3, 300, 300, 40, 40),  # crack 0.3 ≥ 0.2 → 유지
        (2, 0.65, 500, 500, 40, 40),  # guardrail 0.65 ≥ 0.6 → 유지
    ])
    res = decode_yolo(out, lb, 640, 640, conf=[0.6, 0.2, 0.6, 0.6])
    assert [(c, round(s, 2)) for c, s, *_ in res] == [(2, 0.65), (1, 0.3)]
    assert decode_yolo(out, lb, 640, 640, conf=0.25) == decode_yolo(out, lb, 640, 640, conf=[0.25] * 4)
    with pytest.raises(ValueError):
        decode_yolo(out, lb, 640, 640, conf=[0.5, 0.5])


def test_decode_low_conf_then_filter_equals_direct():
    """낮은 기준으로 한 번 돌리고 점수로 거른 것 = 그 기준으로 바로 돌린 것 (conf_thresholds 가 기대는 성질)."""
    lb = Letterbox(1.0, 0, 0)
    rng = np.random.default_rng(1)
    boxes = [(int(rng.integers(4)), float(rng.uniform(0.05, 1)), *rng.uniform(100, 500, 2), 80, 80) for _ in range(60)]
    out = fake_output(boxes, n_noise=0)
    low = decode_yolo(out, lb, 640, 640, conf=0.05)
    thr = [0.3, 0.5, 0.4, 0.7]
    assert decode_yolo(out, lb, 640, 640, conf=thr) == [d for d in low if d[1] >= thr[d[0]]]


def test_class_conf_list_and_loader(tmp_path):
    import json

    from daboyeo_ai.inference import load_class_conf
    from daboyeo_ai.inference.onnx_detector import class_conf_list

    assert class_conf_list(0.3) == [0.3] * 4
    assert class_conf_list({"sign": 0.5}) == [0.25, 0.25, 0.25, 0.5]
    with pytest.raises(ValueError):
        class_conf_list({"lane_wear": 0.5})
    (tmp_path / "a.json").write_text(json.dumps({"class_conf": {"pothole": 0.4}, "classes": {}}))
    (tmp_path / "card.json").write_text(json.dumps({"recommended_conf": 0.25, "class_conf_draft": {"sign": 0.5}}))
    assert load_class_conf(tmp_path / "a.json") == {"pothole": 0.4}
    assert load_class_conf(tmp_path / "card.json") == {"sign": 0.5}
    (tmp_path / "b.json").write_text("{}")
    with pytest.raises(KeyError):
        load_class_conf(tmp_path / "b.json")


def test_road_tiles_cover_road_with_overlap():
    from daboyeo_ai.inference import road_tiles
    from daboyeo_ai.inference.onnx_detector import FULL_FRAME

    w = road_tiles(cols=2)
    assert w[0] == FULL_FRAME and len(w) == 3
    (a1, at, a2, ab), (b1, bt, b2, bb) = w[1:]
    assert (a1, b2) == (0.0, 1.0) and at == bt == 0.45 and ab == bb == 1.0
    assert a2 - b1 == pytest.approx(0.1 * (a2 - a1), abs=1e-5)  # 조각 너비의 10% 겹침
    assert len(road_tiles(3, full_frame=False)) == 3 and road_tiles(3, full_frame=False)[-1][2] == 1.0
    with pytest.raises(ValueError):
        road_tiles(cols=0)
    with pytest.raises(ValueError):
        road_tiles(top=0.9, bottom=0.5)


def test_merge_nms_dedups_same_class_across_windows():
    from daboyeo_ai.inference.onnx_detector import merge_nms

    dets = [(0, 0.6, 100, 100, 200, 200), (0, 0.8, 105, 105, 205, 205), (1, 0.5, 100, 100, 200, 200)]
    assert [(c, s) for c, s, *_ in merge_nms(dets)] == [(0, 0.8), (1, 0.5)]
    assert merge_nms([]) == []


class _FakeSession:
    """입력 640 고정 모델 흉내: 입력 이미지 정중앙에 pothole 하나를 낸다 (0.9)."""

    class _IO:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    def get_inputs(self):
        return [self._IO("images", [1, 3, 640, 640])]

    def get_outputs(self):
        return [self._IO("output0", [1, 8, 8400])]

    def run(self, _, feeds):
        return [fake_output([(0, 0.9, 320, 320, 40, 40)], n_noise=0)]


def _fake_detector(monkeypatch, **kw):
    import onnxruntime as ort

    from daboyeo_ai.inference import OnnxDetector

    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **k: _FakeSession())
    return OnnxDetector("fake.onnx", **kw)


def test_windows_default_is_single_full_frame(monkeypatch):
    det = _fake_detector(monkeypatch)
    assert det.windows is None
    (c, s, x1, y1, x2, y2), = det.raw(np.zeros((1080, 1920, 3), np.uint8))
    assert (x1 + x2) / 2 == pytest.approx(960) and (y1 + y2) / 2 == pytest.approx(540)


def test_windows_map_boxes_back_to_frame(monkeypatch):
    from daboyeo_ai.inference import road_tiles

    det = _fake_detector(monkeypatch, windows=road_tiles(cols=2, overlap=0.0, full_frame=False))
    res = det.raw(np.zeros((1080, 1920, 3), np.uint8))
    centers = sorted(((x1 + x2) / 2, (y1 + y2) / 2) for _, _, x1, y1, x2, y2 in res)
    # 조각 (0~960, 486~1080), (960~1920, 486~1080) 의 중앙 → 프레임 좌표
    assert centers == [pytest.approx((480, 783), abs=1), pytest.approx((1440, 783), abs=1)]
    with pytest.raises(ValueError):
        _fake_detector(monkeypatch, windows=[(0.5, 0, 0.4, 1)])
