"""탐지 결과 형식. 가상 블랙박스와 서버가 함께 쓴다.

기준은 docs/PRD.md 7장이다. 이 파일만 바꾸지 말고 docs/COLLAB.md 3장의 절차를 따른다.

- 전송은 multipart다: `event` 필드에 이 모델의 JSON, `snapshot` 필드에 JPEG 파일 (docs/DESIGN.md 3장).
  그래서 사진은 이 모델에 담기지 않는다.
- 모르는 필드는 무시한다. 필드를 추가해도 상대 쪽이 깨지지 않게 하기 위해서다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DefectType = Literal["pothole", "crack", "lane_wear"]
DEFECT_TYPES: tuple[str, ...] = ("pothole", "crack", "lane_wear")


class DetectionEvent(BaseModel):
    """가상 블랙박스가 서버로 보내는 탐지 결과 한 건."""

    model_config = ConfigDict(extra="ignore")

    device_id: str = Field(min_length=1)
    detected_at: datetime  # 시간대가 붙은 값이어야 한다 (예: 2026-11-14T10:32:11+09:00)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    type: DefectType
    confidence: float = Field(ge=0, le=1)  # lane_wear는 선명도 점수를 0~1로 바꾼 값
    bbox: tuple[int, int, int, int]  # 원본 프레임 픽셀 좌표 x1, y1, x2, y2
    frame_w: int = Field(gt=0)
    frame_h: int = Field(gt=0)
    quality_ok: bool = True

    @field_validator("detected_at")
    @classmethod
    def _needs_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("detected_at에 시간대가 없다")
        return v

    @model_validator(mode="after")
    def _bbox_inside_frame(self) -> "DetectionEvent":
        x1, y1, x2, y2 = self.bbox
        if not (0 <= x1 < x2 <= self.frame_w and 0 <= y1 < y2 <= self.frame_h):
            raise ValueError(f"bbox가 프레임 안의 올바른 박스가 아니다: {self.bbox}")
        return self
