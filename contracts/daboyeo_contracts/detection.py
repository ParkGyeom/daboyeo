"""탐지기 계약 (박겸 → 정현우).

가상 블랙박스는 `Detector.detect(frame)` 만 호출한다. 전처리(리사이즈·정규화)와 NMS는
탐지기 구현(예: ai/inference/onnx_detector.py) 안에서 끝낸다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .classes import CLASS_NAMES, DETECTOR_TO_DEFECT, ClassName, DefectType

if TYPE_CHECKING:
    import numpy as np

BBox = tuple[float, float, float, float]


class Detection(BaseModel):
    """프레임 한 장에서 나온 탐지 1건."""

    model_config = ConfigDict(frozen=True)

    class_id: int = Field(ge=0, lt=len(CLASS_NAMES))
    class_name: ClassName
    conf: float = Field(ge=0.0, le=1.0)
    bbox: BBox = Field(description="x1, y1, x2, y2 — 프레임 크기로 나눈 0~1 값")
    # 시설물(guardrail·sign)에만 채운다. 파손 분류를 못 하면 None (PRD F1 대체안).
    damage: Literal["normal", "suspected"] | None = None
    damage_conf: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> Detection:
        if CLASS_NAMES[self.class_id] != self.class_name:
            raise ValueError(
                f"class_id {self.class_id} is {CLASS_NAMES[self.class_id]!r}, not {self.class_name!r}"
            )
        x1, y1, x2, y2 = self.bbox
        if not all(0.0 <= v <= 1.0 for v in self.bbox):
            raise ValueError(f"bbox must be normalized to 0~1, got {self.bbox}")
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"bbox must satisfy x1<x2 and y1<y2, got {self.bbox}")
        if self.damage is not None and self.class_name not in ("guardrail", "sign"):
            raise ValueError("damage is only for guardrail/sign detections")
        return self

    @classmethod
    def from_class_id(cls, class_id: int, conf: float, bbox: BBox, **kw) -> Detection:
        return cls(class_id=class_id, class_name=CLASS_NAMES[class_id], conf=conf, bbox=bbox, **kw)

    @property
    def defect_type(self) -> DefectType:
        return DETECTOR_TO_DEFECT[self.class_name]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@runtime_checkable
class Detector(Protocol):
    """모든 탐지기(fake / onnx / pytorch / replay)가 따르는 인터페이스."""

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """BGR uint8 프레임(H×W×3, OpenCV 기본)을 받아 탐지 결과를 돌려준다."""
        ...
