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
