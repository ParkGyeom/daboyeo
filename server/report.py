"""신고서 내용 (PRD F6, reports.payload). 제출은 모의 제출이다."""

from __future__ import annotations

from datetime import datetime

from server import rules


def build_payload(defect: dict, user_name: str, now: datetime) -> dict:
    """reports.payload(jsonb)에 그대로 저장한다: 사진·위치·유형·AI 신뢰도·관측 차량 수."""
    return {
        "reporter": user_name,
        "reported_at": now.astimezone(rules.KST).isoformat(timespec="seconds"),
        "defect_type": defect["defect_type"],
        "risk_level": defect["risk_level"],
        "location": {"lat": defect["lat"], "lon": defect["lon"],
                     "jurisdiction": rules.jurisdiction(defect["lat"], defect["lon"]),
                     "address": defect.get("address")},  # 없으면 None (server/geocode.py)
        "snapshot": defect["snapshot_path"],
        "confidence": round(float(defect["confidence"]), 3),
        "vehicle_count": defect["vehicle_count"],
        "observation_count": defect["observation_count"],
        "first_seen_at": defect["first_seen_at"].astimezone(rules.KST).isoformat(timespec="seconds"),
        "reward_eligible_road": defect["reward_eligible_road"],
    }
