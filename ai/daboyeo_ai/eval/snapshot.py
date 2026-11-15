"""블랙박스가 서버로 보내는 크롭 JPEG 를 똑같이 만든다 — 서버 검증(V0) 평가용 사본.

출처 (정현우 담당, 수정 금지 — 여기는 복사본):
- `crop_snapshot`: `origin/be/port-existing` 커밋 3c62628 의 `edge/blackbox/sender.py`
- `pixel_box`: 같은 커밋 `edge/blackbox/detect.py` 40행 (탐지기 실수 좌표 → 정수 픽셀 박스)

원본이 바뀌면 여기도 맞춰야 평가가 실제 전송 크롭과 같아진다.
"""

from __future__ import annotations

import cv2
import numpy as np

SNAPSHOT_MAX_SIDE = 640
SNAPSHOT_QUALITY = 80


def pixel_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    """edge/blackbox/detect.py 와 같은 반올림: 왼쪽 위는 버림, 오른쪽 아래는 반올림, 이미지 안으로."""
    return max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2)))


def crop_region(image_shape: tuple[int, ...], bbox: tuple[int, int, int, int],
                margin: float = 0.5) -> tuple[int, int, int, int]:
    """crop_snapshot 이 자르는 영역 (x0, y0, x1, y1). 박스 크기 × margin + 16px 여유, 이미지 안으로 자른다."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin) + 16, int((y2 - y1) * margin) + 16
    return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)


def crop_snapshot(image: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.5) -> bytes:
    """결함 주변을 여유를 두고 잘라 긴 변 640픽셀 이하 JPEG(품질 80) 로 만든다. 확대는 하지 않는다."""
    x0, y0, x1, y1 = crop_region(image.shape, bbox, margin)
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = SNAPSHOT_MAX_SIDE / max(ch, cw)
    if scale < 1:
        crop = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
    if not ok:
        raise ValueError("JPEG 인코딩 실패")
    return buf.tobytes()
