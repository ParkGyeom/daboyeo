"""위험도, 노출 조건, 포상 대상, 신고 상태 전환 규칙 (docs/DESIGN.md 6장).

DB 없이 시험할 수 있게 순수 함수로만 만든다. 기준값은 초안이고 실험으로 조정한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from server import config

KST = timezone(timedelta(hours=9))

# 중복 정리 거리 기준(m)
DEDUP_DISTANCE_M = {
    "pothole": float(config.get("DEDUP_DISTANCE_M_POTHOLE_CRACK", "20")),
    "crack": float(config.get("DEDUP_DISTANCE_M_POTHOLE_CRACK", "20")),
    "lane_wear": float(config.get("DEDUP_DISTANCE_M_LANE_WEAR", "30")),
}

CONFIDENCE_VISIBLE = float(config.get("CONFIDENCE_VISIBLE", "0.6"))
MIN_DEVICES_VISIBLE = 2
REWARD_TARGET = 5

UNREPORTED, REPORTED, REPAIRING = "미신고", "신고됨", "보수 중"
STATUSES = (UNREPORTED, REPORTED, REPAIRING)
URGENT, ROUTINE = "빠른 보수", "정기 보수"

SEOUL = "서울특별시"
# DATA.md 3장의 "서울 대략 범위". 행정 경계가 아니라 사각형이라 경계 부근은 틀릴 수 있다.
_SEOUL_BOX = (37.42, 37.70, 126.76, 127.18)  # 위도 최소·최대, 경도 최소·최대
OUTSIDE_SEOUL = "서울 밖 (관할 미확정)"


def severity(defect_type: str) -> str:
    """포트홀은 빠른 보수, 균열·차선 마모는 정기 보수."""
    return URGENT if defect_type == "pothole" else ROUTINE


def in_seoul(lat: float, lon: float) -> bool:
    s, n, w, e = _SEOUL_BOX
    return s <= lat <= n and w <= lon <= e


def is_reward_eligible(lat: float, lon: float) -> bool:
    """위치가 서울 안이면 포상 대상. 서울시 관리 도로 목록을 구하면 그것으로 바꾼다."""
    return in_seoul(lat, lon)


def jurisdiction(lat: float, lon: float) -> str:
    """서울이면 서울특별시. 그 밖은 시연 구간의 시·군 표를 아직 만들지 못해 미확정으로 둔다."""
    return SEOUL if in_seoul(lat, lon) else OUTSIDE_SEOUL


def passes_confidence_or_devices(max_confidence: float, device_count: int) -> bool:
    """신뢰도가 기준 이상이거나 관측 기기가 2대 이상이면 시민에게 보여줄 만하다."""
    return max_confidence >= CONFIDENCE_VISIBLE or device_count >= MIN_DEVICES_VISIBLE


def is_visible_to_citizen(
    status: str, max_confidence: float, device_count: int, observed_by_user: bool, skipped_by_user: bool
) -> bool:
    """미신고이고, 그 시민의 기기가 관측했고, 넘기지 않았고, 신뢰도·기기 수 조건을 넘는 결함만 보여준다."""
    return (
        status == UNREPORTED
        and observed_by_user
        and not skipped_by_user
        and passes_confidence_or_devices(max_confidence, device_count)
    )


# 허용하는 상태 전환: 미신고 → 신고됨(신고서 생성 시) → 보수 중(관리자)
_TRANSITIONS = {(UNREPORTED, REPORTED), (REPORTED, REPAIRING)}


def can_transition(old: str, new: str) -> bool:
    return (old, new) in _TRANSITIONS


def quarter_start(now: datetime) -> datetime:
    """now가 속한 분기의 시작 시각 (한국 시간 기준)."""
    now = now.astimezone(KST)
    first_month = 3 * ((now.month - 1) // 3) + 1
    return now.replace(month=first_month, day=1, hour=0, minute=0, second=0, microsecond=0)


def reward_progress(valid_reports_this_quarter: int) -> dict:
    return {
        "count": valid_reports_this_quarter,
        "target": REWARD_TARGET,
        "remaining": max(0, REWARD_TARGET - valid_reports_this_quarter),
    }
