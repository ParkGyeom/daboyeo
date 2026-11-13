"""관측 전송 계약 (가상 블랙박스 → 서버).

`POST /api/v1/observations` 의 multipart 요청에서 JSON 메타 부분이다.
캡처(jpg)와 클립(mp4)은 같은 요청의 파일 파트로 보낸다.
관측 1건 = 블랙박스가 연속 프레임 탐지를 합친 결함 1건 (edge/blackbox/track_merge.py).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .classes import DefectType
from .detection import BBox


class ObservationIn(BaseModel):
    # 재전송 시 중복 저장을 막는 키 (FR-2.4). 블랙박스가 만든다.
    client_observation_id: UUID = Field(default_factory=uuid4)
    trip_id: UUID
    vehicle_id: UUID

    defect_type: DefectType
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox | None = Field(default=None, description="최고 신뢰도 프레임의 박스 (차선 마모는 없음)")
    size_metric: float | None = Field(default=None, description="위험도용 크기 지표, 정의 TBD (#5)")
    is_suspected: bool = Field(default=False, description="시설물 '파손 의심' 여부")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    t_s: float = Field(ge=0, description="영상 시작 기준 초 (GPS CSV 의 t_s 와 같은 축)")
    observed_at: datetime = Field(description="trip.started_at + t_s")

    quality_score: float | None = None
