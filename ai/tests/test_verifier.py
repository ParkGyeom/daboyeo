import cv2
import numpy as np
import pytest

from daboyeo_ai.eval.snapshot import crop_region, crop_snapshot, pixel_box
from daboyeo_ai.inference.verifier import Verifier, central_score


def _decode(jpg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


# ---------- crop_snapshot 사본 (edge/blackbox/sender.py 와 같은 동작) ----------

def test_crop_region_margin_and_clamp():
    # 가운데 박스 100×50 → 여유 50+16, 25+16
    assert crop_region((1080, 1920), (500, 500, 600, 550)) == (434, 459, 666, 591)
    # 왼쪽 위 모서리에 붙은 박스는 이미지 안으로 잘린다
    assert crop_region((1080, 1920), (0, 0, 40, 20)) == (0, 0, 76, 46)


def test_crop_snapshot_no_upscale_and_downscale():
    img = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    small = _decode(crop_snapshot(img, (500, 500, 540, 520)))
    assert small.shape[:2] == (20 + 2 * 26, 40 + 2 * 36)  # 확대하지 않는다
    big = _decode(crop_snapshot(img, (100, 100, 1800, 1000)))
    assert max(big.shape[:2]) == 640  # 긴 변 640 으로 줄인다
    assert big.shape[1] == 640  # 가로가 긴 변 (1920 전체)


def test_crop_snapshot_keeps_box_in_middle():
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[500:600, 800:1000] = 255  # 흰 박스
    crop = _decode(crop_snapshot(img, (800, 500, 1000, 600)))
    h, w = crop.shape[:2]
    ys, xs = np.where(crop[:, :, 0] > 128)
    assert (xs.mean() / w, ys.mean() / h) == pytest.approx((0.5, 0.5), abs=0.02)


def test_pixel_box_rounding_matches_edge():
    assert pixel_box(-3.2, 10.7, 99.5, 2000.0, 1920, 1080) == (0, 10, 100, 1080)


# ---------- V0 점수 규칙 ----------

def test_central_score_same_class_and_center_only():
    raw = [
        (0, 0.9, 0, 0, 30, 30),       # 같은 클래스지만 구석 → 무시
        (1, 0.8, 40, 40, 60, 60),     # 가운데지만 다른 클래스 → 무시
        (0, 0.4, 30, 30, 70, 70),     # 같은 클래스·가운데 → 채택
        (0, 0.3, 45, 45, 55, 55),
    ]
    assert central_score(raw, 0, 100, 100) == pytest.approx(0.4)
    assert central_score(raw, 2, 100, 100) == 0.0
    assert central_score([(0, 0.9, 0, 0, 30, 30)], 0, 100, 100, center=1.0) == pytest.approx(0.9)


class _FakeDetector:
    def __init__(self, raw):
        self._raw = raw
        self.seen = None

    def raw(self, img):
        self.seen = img.shape
        return self._raw


def _verifier(raw) -> Verifier:
    v = Verifier.__new__(Verifier)  # 모델 파일 없이
    v.detector, v.center = _FakeDetector(raw), 0.6
    return v


def test_verify_decodes_jpeg_and_scores():
    img = np.zeros((200, 300, 3), np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    v = _verifier([(3, 0.7, 120, 80, 180, 120)])  # 중심 (0.5, 0.5)
    assert v.verify(buf.tobytes(), "sign") == pytest.approx(0.7)
    assert v.detector.seen[:2] == (200, 300)
    assert v.verify(buf.tobytes(), "pothole") == 0.0


def test_verify_rejects_bad_input():
    v = _verifier([])
    with pytest.raises(ValueError):
        v.verify(b"not a jpeg", "pothole")
    ok, buf = cv2.imencode(".jpg", np.zeros((10, 10, 3), np.uint8))
    with pytest.raises(ValueError):
        v.verify(buf.tobytes(), "lane_wear")
