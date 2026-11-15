"""서버 쪽 검증기 V0 — 블랙박스가 보낸 크롭 JPEG 에 같은 탐지기(detector_v2)를 한 번 더 돌린다.

    v = Verifier("ai/models/detector_v2.onnx", threads=1)
    score = v.verify(jpeg_bytes, "pothole")   # 0~1, 높을수록 '진짜 결함'

onnxruntime 만 쓴다 (서버에 torch·ultralytics 없음).

점수 규칙: 크롭을 탐지기에 넣고 (낮은 기준 0.01 로 거의 전부 받는다), **같은 클래스이면서 박스 중심이
크롭 가운데 영역 안에 있는** 탐지의 최고 신뢰도. 없으면 0.

왜 가운데 영역인가: 블랙박스 크롭(`crop_snapshot`)은 원래 박스 양옆에 박스 크기의 0.5배 + 16px 를 붙여
자른다. 이미지 가장자리에 걸리지 않으면 원래 박스는 크롭의 가운데 50% (0.25~0.75) 를 정확히 차지하고 중심은
0.5 에 온다. 이미지 가장자리에서 여유가 잘리면 중심이 한쪽으로 밀리는데, 한 변이 완전히 잘려도 중심은
0.5·b/(1.5·b+16) ≥ 0.25 언저리에 머문다. 그래서 중심이 가운데 `center` 비율(기본 0.6 → 0.2~0.8) 안이면
'원래 박스를 다시 찾은 것' 으로 본다. 크롭 가장자리에 걸친 다른 결함(옆 균열 등)은 세지 않는다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .onnx_detector import OnnxDetector

VERIFY_LOW_CONF = 0.01
DEFAULT_CENTER = 0.6


def central_score(raw: list[tuple[int, float, float, float, float, float]], class_id: int,
                  w: int, h: int, center: float = DEFAULT_CENTER) -> float:
    """탐지 [(cls, score, x1, y1, x2, y2)] 중 같은 클래스·중심이 가운데 center 비율 안인 것의 최고 점수 (없으면 0)."""
    lo, hi = (1 - center) / 2, (1 + center) / 2
    best = 0.0
    for c, s, x1, y1, x2, y2 in raw:
        if c != class_id:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        if lo <= cx <= hi and lo <= cy <= hi:
            best = max(best, float(s))
    return min(best, 1.0)


class Verifier:
    """V0: 크롭에 같은 탐지기를 다시 돌린 점수. `threads` 는 onnxruntime intra-op 스레드 수."""

    def __init__(self, model_path: str | Path, threads: int = 1, center: float = DEFAULT_CENTER):
        self.detector = OnnxDetector(model_path, threads=threads, conf=VERIFY_LOW_CONF)
        self.center = center

    def raw(self, image_bgr: np.ndarray) -> list[tuple[int, float, float, float, float, float]]:
        return self.detector.raw(image_bgr)

    def verify_image(self, image_bgr: np.ndarray, defect_type: str) -> float:
        if defect_type not in CLASS_NAMES:
            raise ValueError(f"unknown defect_type {defect_type!r} (expected {CLASS_NAMES})")
        h, w = image_bgr.shape[:2]
        return central_score(self.raw(image_bgr), CLASS_NAMES.index(defect_type), w, h, self.center)

    def verify(self, jpeg: bytes, defect_type: str) -> float:
        """크롭 JPEG 바이트 + 블랙박스가 붙인 결함 종류 → 0~1. 디코딩이 안 되면 ValueError."""
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("JPEG 디코딩 실패")
        return self.verify_image(img, defect_type)
