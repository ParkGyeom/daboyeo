"""server/schema.py 확인. 실행: .venv/bin/python -m pytest server"""

import json

import pytest
from pydantic import ValidationError

from server.schema import DetectionEvent

# docs/PRD.md 7장의 예시 그대로 (snapshot은 multipart 파일이라 JSON에는 없어도 된다)
PRD_EXAMPLE = """{
  "device_id": "bb-01",
  "detected_at": "2026-11-14T10:32:11+09:00",
  "lat": 37.5503, "lon": 127.0731,
  "type": "pothole", "confidence": 0.82,
  "bbox": [612, 401, 980, 488],
  "frame_w": 1920, "frame_h": 1080,
  "quality_ok": true,
  "snapshot": "<JPEG 파일 첨부 또는 경로>"
}"""


def good(**over):
    d = json.loads(PRD_EXAMPLE)
    d.update(over)
    return d


def test_prd_example_parses_and_roundtrips():
    e = DetectionEvent.model_validate_json(PRD_EXAMPLE)
    assert e.type == "pothole" and e.bbox == (612, 401, 980, 488)
    assert DetectionEvent.model_validate_json(e.model_dump_json()) == e


def test_unknown_fields_are_ignored():
    e = DetectionEvent.model_validate(good(new_field=1))
    assert not hasattr(e, "new_field")


@pytest.mark.parametrize("t", ["pothole", "crack", "lane_wear"])
def test_all_types_accepted(t):
    assert DetectionEvent.model_validate(good(type=t)).type == t


@pytest.mark.parametrize(
    "over",
    [
        {"type": "manhole"},
        {"confidence": 1.2},
        {"confidence": -0.1},
        {"lat": 95},
        {"detected_at": "2026-11-14T10:32:11"},  # 시간대 없음
        {"bbox": [980, 401, 612, 488]},  # x1 > x2
        {"bbox": [0, 0, 2000, 100]},  # 프레임 밖
        {"frame_w": 0},
        {"device_id": ""},
    ],
)
def test_invalid_rejected(over):
    with pytest.raises(ValidationError):
        DetectionEvent.model_validate(good(**over))
