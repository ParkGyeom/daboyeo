"""가상 블랙박스와 AI 추론 모듈(daboyeo_ai)을 잇는다.

탐지는 OnnxDetector(클래스별 신뢰도 기준, handoff 4-1장), 차선 마모는 LaneWearMeter + LaneWearTracker 를
그대로 부르고 (docs/ai-handoff.md 1·8장),
결과를 지금 서버(server/schema.py)가 받는 모양(유형, 신뢰도, 픽셀 박스)으로만 바꾼다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from daboyeo_ai.inference import OnnxDetector
from daboyeo_ai.lane_wear import LaneWearMeter, LaneWearTracker
from daboyeo_contracts import CLASS_NAMES

# 서버로 보내는 유형. guardrail·sign 은 탐지·화면 표시는 하지만, 서버로 보낼 유형을 박겸과 정하기 전까지 보내지 않는다
SENT_TYPES = {"pothole", "crack", "lane_wear"}


@dataclass
class Detection:
    type: str  # "pothole" | "crack" | "guardrail" | "sign" | "lane_wear"
    confidence: float  # 0~1
    bbox: tuple[int, int, int, int]  # 원본 프레임 픽셀 좌표 x1, y1, x2, y2


class Detector:
    def __init__(self, model_path: str, conf: float | Mapping[str, float] = 0.25, threads: int = 1):
        """conf: 숫자 하나 또는 클래스별 dict (daboyeo_ai.inference.load_class_conf 로 읽은 것)."""
        self._det = OnnxDetector(model_path, threads=threads, conf=conf)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """프레임 한 장(BGR)에서 4클래스를 모두 찾는다. 신뢰도 높은 순. 보낼지는 SENT_TYPES 로 고른다."""
        h, w = image.shape[:2]
        out = []
        for cls, score, x1, y1, x2, y2 in self._det.raw(image):
            name = CLASS_NAMES[int(cls)]
            box = (max(0, int(x1)), max(0, int(y1)), min(w, int(round(x2))), min(h, int(round(y2))))
            out.append(Detection(name, float(score), box))
        return sorted(out, key=lambda d: d.confidence, reverse=True)


class LaneWear:
    """차선 마모. 주행(기기) 하나에 하나 만든다.

    프레임마다 LaneWearMeter 로 재고 LaneWearTracker(같은 주행의 평소 차선과 비교 + 연속 5장 중 3장 투표)에
    넣는다. 마모 관측이 확정된 프레임에서만 Detection(type="lane_wear") 을 돌려주고, 그 밖에는 None.
    박스는 측정 영역(ROI)을 감싸는 사각형이다. 서버로 보낼 사진을 자르는 데만 쓴다.
    """

    def __init__(self, meter: LaneWearMeter | None = None, tracker: LaneWearTracker | None = None):
        self.meter = meter or LaneWearMeter()
        self.tracker = tracker or LaneWearTracker()

    def check(self, image: np.ndarray, t: float | None = None) -> Detection | None:
        """t: 프레임 시각(초). 주면 tracker 가 구간 끊김(5초 넘는 간격)을 안다."""
        j = self.tracker.update(self.meter.measure(image), t=t)
        if not j.confirmed:
            return None
        h, w = image.shape[:2]
        poly = self.meter.roi_polygon_px(w, h)
        x1, y1 = poly.min(axis=0)
        x2, y2 = poly.max(axis=0)
        return Detection("lane_wear", float(j.confidence), (int(x1), int(y1), int(x2) + 1, int(y2) + 1))
