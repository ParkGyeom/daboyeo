"""위험도, 노출 조건, 포상 대상, 신고 상태 전환 규칙. DB 없이 시험할 수 있게 순수 함수로만 둔다.

값의 이름은 server/sql/001_init.sql 의 enum 과 daboyeo_contracts 를 그대로 쓴다.
화면에 보일 한국어 이름은 화면(ui/)이 붙인다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from daboyeo_contracts import RISK_BY_TYPE, DefectType

from server import config

KST = timezone(timedelta(hours=9))

# 중복 정리 거리 기준(m). 차선 마모는 구간이라 더 넓게 본다
_POINT_M = float(config.get("DEDUP_DISTANCE_M_POTHOLE_CRACK", "20"))
DEDUP_DISTANCE_M: dict[str, float] = {t.value: _POINT_M for t in DefectType}
DEDUP_DISTANCE_M[DefectType.LANE_WEAR.value] = float(config.get("DEDUP_DISTANCE_M_LANE_WEAR", "30"))

# 노출 기준 (PRD F7, 계획 3.1): 신뢰도 θ 이상이거나 관측 차량 k 대 이상. θ 는 박겸과 11/24 확정 예정
CONFIDENCE_VISIBLE = float(config.get("CONFIDENCE_VISIBLE", "0.6"))
MIN_VEHICLES_VISIBLE = int(config.get("MIN_VEHICLES_VISIBLE", "2"))
REWARD_TARGET = 5

# 001_init.sql 의 defect_status / finding_response
UNREPORTED, REPORTED, UNDER_REPAIR = "unreported", "reported", "under_repair"
STATUSES = (UNREPORTED, REPORTED, UNDER_REPAIR)
PENDING, RESP_REPORTED, SKIPPED, ALREADY_REPORTED = "pending", "reported", "skipped", "already_reported"

SEOUL = "서울특별시"
# 서울 대략 범위. 행정 경계가 아니라 사각형이라 경계 부근은 틀릴 수 있다
_SEOUL_BOX = (37.42, 37.70, 126.76, 127.18)  # 위도 최소·최대, 경도 최소·최대
OUTSIDE_SEOUL = "서울 밖 (관할 미확정)"


def risk_level(defect_type: str) -> str:
    """유형별 위험도 (contracts RISK_BY_TYPE, PRD F4). 크기 반영(#5)은 아직 하지 않는다."""
    return RISK_BY_TYPE[DefectType(defect_type)].value


def in_seoul(lat: float, lon: float) -> bool:
    s, n, w, e = _SEOUL_BOX
    return s <= lat <= n and w <= lon <= e


def is_reward_eligible(lat: float, lon: float) -> bool:
    """위치가 서울 안이면 포상 대상. 서울시 관리 도로 목록을 구하면 그것으로 바꾼다 (오픈 이슈 #7)."""
    return in_seoul(lat, lon)


def jurisdiction(lat: float, lon: float) -> str:
    return SEOUL if in_seoul(lat, lon) else OUTSIDE_SEOUL


def is_exposed(confidence: float, vehicle_count: int, is_suspected: bool = False) -> bool:
    """시민에게 보여줄 만한가: 신뢰도 θ 이상이거나 관측 차량 k 대 이상.
    시설물 '파손 의심'은 한 대의 판단으로는 보여주지 않는다 (차량 수 조건만)."""
    if is_suspected:
        return vehicle_count >= MIN_VEHICLES_VISIBLE
    return confidence >= CONFIDENCE_VISIBLE or vehicle_count >= MIN_VEHICLES_VISIBLE


# 허용하는 상태 전환: 미신고 → 신고됨(신고서 생성 시) → 보수 중(관리자)
_TRANSITIONS = {(UNREPORTED, REPORTED), (REPORTED, UNDER_REPAIR)}


def can_transition(old: str, new: str) -> bool:
    return (old, new) in _TRANSITIONS


def quarter_start(now: datetime) -> datetime:
    """now 가 속한 분기의 시작 시각 (한국 시간 기준)."""
    now = now.astimezone(KST)
    first_month = 3 * ((now.month - 1) // 3) + 1
    return now.replace(month=first_month, day=1, hour=0, minute=0, second=0, microsecond=0)


def quarter_label(now: datetime) -> str:
    """reports.quarter 값. 예: 2026-Q4 (한국 시간 기준)."""
    now = now.astimezone(KST)
    return f"{now.year}-Q{(now.month - 1) // 3 + 1}"


def reward_progress(valid_reports_this_quarter: int) -> dict:
    return {
        "count": valid_reports_this_quarter,
        "target": REWARD_TARGET,
        "remaining": max(0, REWARD_TARGET - valid_reports_this_quarter),
    }

