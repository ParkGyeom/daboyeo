"""영상 품질 판별 (docs/DESIGN.md 5장). 매 프레임 탐지 전에 부른다.

프레임을 줄인 뒤 노면 영역(가운데 아래쪽)에서 세 값을 잰다.
  밝기: 평균 밝기. 너무 어두우면 노면이 보이지 않는다
  흐림: 라플라시안 분산. 안개·물보라로 뿌예지면 작아진다. 매끈한 노면도 작게 나와서
        대비(밝기 표준편차)도 낮을 때만 흐림으로 본다. 야간 가로등 아래 프레임이 이 덕에 통과한다
  과노출: 하얗게 날아간 픽셀 비율

프레임 전체가 아니라 노면 영역을 보는 이유: 야간에도 헤드라이트가 비추는 노면은 밝고,
낮에는 하늘이 하얗게 날아가도 노면은 멀쩡하다. 파일 이름의 night 를 그대로 "나쁨"으로 보지 않는다.

기준값은 로드마크 3,281장을 조건(맑음·야간·우천·안개)별로 재고 경계 부근 프레임을 눈으로 보고 정했다.
측정 결과는 docs/PLAN.md 5장.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SMALL = (480, 270)
# 노면 영역: 세로 50~85% (아래 15%는 보닛이 들어오는 경우가 많다), 가로 15~85%
ROAD_Y = (0.50, 0.85)
ROAD_X = (0.15, 0.85)

MIN_BRIGHTNESS = 40.0
MIN_SHARPNESS = 200.0
MIN_CONTRAST = 30.0  # 흐림 판정은 선명도와 대비가 둘 다 낮을 때만
MAX_OVEREXPOSED = 0.25
OVEREXPOSED_LEVEL = 250


@dataclass(frozen=True)
class Quality:
    ok: bool
    brightness: float
    sharpness: float
    contrast: float
    overexposed: float
    reason: str  # 나쁠 때 이유. 좋으면 빈 문자열


def road_region(gray_small: np.ndarray) -> np.ndarray:
    h, w = gray_small.shape[:2]
    return gray_small[int(h * ROAD_Y[0]): int(h * ROAD_Y[1]), int(w * ROAD_X[0]): int(w * ROAD_X[1])]


def measure(image: np.ndarray) -> Quality:
    """BGR 프레임의 품질을 잰다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    road = road_region(cv2.resize(gray, SMALL, interpolation=cv2.INTER_AREA))
    brightness = float(road.mean())
    sharpness = float(cv2.Laplacian(road, cv2.CV_64F).var())
    overexposed = float((road >= OVEREXPOSED_LEVEL).mean())
    contrast = float(road.std())

    reasons = []
    if brightness < MIN_BRIGHTNESS:
        reasons.append("어두움")
    if sharpness < MIN_SHARPNESS and contrast < MIN_CONTRAST:
        reasons.append("흐림")
    if overexposed > MAX_OVEREXPOSED:
        reasons.append("과노출")
    return Quality(not reasons, brightness, sharpness, contrast, overexposed, ",".join(reasons))
