"""위험도, 노출 조건, 포상 대상, 신고 상태 전환 규칙. DB 없이 시험할 수 있게 순수 함수로만 둔다.

값의 이름은 server/sql/001_init.sql 의 enum 과 daboyeo_contracts 를 그대로 쓴다.
화면에 보일 한국어 이름은 화면(ui/)이 붙인다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from daboyeo_contracts import RISK_BY_TYPE, DefectType

from server import config

KST = timezone(timedelta(hours=9))

# 중복 정리 거리 기준(m), 오픈 이슈 #2 (11/17 확정 30 m): 재구성 GPS hold-out 오차가 달리는 구간에서
# gn_a 25.7 m · gn_b 7.6 m 로 7 m 이상이라 20 m 는 같은 결함이 나뉜다 (plan 11-3). 차선 마모는 구간이라 더 넓게 본다
_POINT_M = float(config.get("DEDUP_DISTANCE_M_POTHOLE_CRACK", "30"))
DEDUP_DISTANCE_M: dict[str, float] = {t.value: _POINT_M for t in DefectType}
DEDUP_DISTANCE_M[DefectType.LANE_WEAR.value] = float(config.get("DEDUP_DISTANCE_M_LANE_WEAR", "40"))

# 노출 기준 (PRD F7, 오픈 이슈 #3, 11/17 확정): 신뢰도 θ 이상이거나 관측 차량 k 대 이상.
# θ 는 유형별로 블랙박스와 같은 기준(시연 모델 카드의 class_conf_draft)을 쓴다. 그러면 시연에서 보이는 결함은
# 모두 '신뢰도로 통과한 것'이고, k 는 실서비스 설계(서로 다른 차량)로만 설명한다 (같은 영상 변형 기기는 독립이 아님).
# 카드에 없는 유형(차선 마모 등)은 CONFIDENCE_VISIBLE.
CONFIDENCE_VISIBLE = float(config.get("CONFIDENCE_VISIBLE", "0.6"))
CONFIDENCE_VISIBLE_JSON = config.get("CONFIDENCE_VISIBLE_JSON", "ai/models/detector_v3.json")
MIN_VEHICLES_VISIBLE = int(config.get("MIN_VEHICLES_VISIBLE", "2"))

# 탐지 클래스 이름 → 서버 결함 유형 (시설물은 '파손' 유형으로 들어온다)
_CLASS_TO_TYPE = {"pothole": "pothole", "crack": "crack", "guardrail": "guardrail_damage", "sign": "sign_damage"}


def load_type_conf(path: str | None = CONFIDENCE_VISIBLE_JSON) -> dict[str, float]:
    """모델 카드(class_conf_draft) 또는 기준 파일(class_conf) → {결함 유형: θ}. 서버는 daboyeo_ai 를 부르지 않고 JSON 만 읽는다.
    파일이 없으면 빈 dict (모든 유형이 CONFIDENCE_VISIBLE)."""
    if not path:
        return {}
    f = Path(path) if Path(path).is_absolute() else config.ROOT / path
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    conf = data.get("class_conf") or data.get("class_conf_draft") or {}
    return {_CLASS_TO_TYPE[k]: float(v) for k, v in conf.items() if k in _CLASS_TO_TYPE}


CONFIDENCE_VISIBLE_BY_TYPE: dict[str, float] = {t.value: CONFIDENCE_VISIBLE for t in DefectType}
CONFIDENCE_VISIBLE_BY_TYPE.update(load_type_conf())


def theta(defect_type: str) -> float:
    return CONFIDENCE_VISIBLE_BY_TYPE.get(defect_type, CONFIDENCE_VISIBLE)
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


def is_exposed(confidence: float, vehicle_count: int, is_suspected: bool = False, defect_type: str = "") -> bool:
    """시민에게 보여줄 만한가: 유형별 신뢰도 θ 이상이거나 관측 차량 k 대 이상.
    시설물 '파손 의심'은 한 대의 판단으로는 보여주지 않는다 (차량 수 조건만)."""
    if is_suspected:
        return vehicle_count >= MIN_VEHICLES_VISIBLE
    return confidence >= theta(defect_type) or vehicle_count >= MIN_VEHICLES_VISIBLE


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

