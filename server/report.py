"""신고서 생성 (docs/DESIGN.md 6장). 제출은 모의 제출이다."""

from __future__ import annotations

from datetime import datetime

from server.rules import KST


def build_report_body(defect: dict, user_name: str, now: datetime) -> dict:
    """신고서에 들어갈 내용. DB의 jsonb 열에 그대로 저장한다."""
    return {
        "reporter": user_name,
        "reported_at": now.isoformat(timespec="seconds"),
        "defect_type": defect["type"],
        "severity": defect["severity"],
        "location": {"lat": defect["lat"], "lon": defect["lon"], "jurisdiction": defect["jurisdiction"]},
        "snapshot": defect["best_snapshot_path"],
        "confidence": round(float(defect["max_confidence"]), 3),
        "device_count": defect["device_count"],
        "first_seen": defect["first_seen"].astimezone(KST).isoformat(timespec="seconds"),
        "reward_eligible": defect["reward_eligible"],
        "submission": "제출됨(모의)",
    }
