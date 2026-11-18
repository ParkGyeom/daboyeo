from datetime import datetime, timedelta, timezone

from server import rules as r

KST = timezone(timedelta(hours=9))


def test_risk_level_follows_contracts():
    assert r.risk_level("pothole") == r.risk_level("guardrail_damage") == "urgent"
    assert r.risk_level("crack") == r.risk_level("lane_wear") == r.risk_level("sign_damage") == "routine"


def test_exposure():
    theta, k = r.CONFIDENCE_VISIBLE, r.MIN_VEHICLES_VISIBLE
    assert r.is_exposed(theta, 1)                  # 신뢰도 θ 이상
    assert r.is_exposed(theta - 0.2, k)            # 신뢰도 낮아도 차량 k 대
    assert not r.is_exposed(theta - 0.2, k - 1)    # 둘 다 아님
    assert not r.is_exposed(0.99, k - 1, is_suspected=True)  # 시설물 파손 의심은 차량 수 조건만
    assert r.is_exposed(0.1, k, is_suspected=True)


def test_transitions():
    assert r.can_transition("unreported", "reported")
    assert r.can_transition("reported", "under_repair")
    assert not r.can_transition("unreported", "under_repair")
    assert not r.can_transition("reported", "unreported")
    assert not r.can_transition("under_repair", "reported")


def test_dedup_distance_per_type():
    assert r.DEDUP_DISTANCE_M["pothole"] == r.DEDUP_DISTANCE_M["crack"] == r.DEDUP_DISTANCE_M["sign_damage"]
    assert r.DEDUP_DISTANCE_M["lane_wear"] > r.DEDUP_DISTANCE_M["pothole"]


def test_seoul_reward_and_jurisdiction():
    assert r.is_reward_eligible(37.4979, 127.0276) and r.jurisdiction(37.4979, 127.0276) == "서울특별시"
    assert not r.is_reward_eligible(36.78, 127.17)
    assert r.jurisdiction(36.78, 127.17) == r.OUTSIDE_SEOUL


def test_quarter_label():
    now = datetime(2026, 11, 15, 1, 0, tzinfo=KST)
    assert r.quarter_label(now) == "2026-Q4"
    # UTC 로 9월 30일 16시 = 한국 10월 1일 1시 → 4분기
    assert r.quarter_label(datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)) == "2026-Q4"


def test_theta_is_per_type_from_model_card():
    """θ 는 블랙박스와 같은 유형별 기준 (v3 카드 class_conf_draft, 오픈 이슈 #3). 카드에 없는 차선 마모는 기본 θ."""
    assert r.theta("pothole") == 0.31 and r.theta("crack") == 0.42
    assert r.theta("lane_wear") == r.CONFIDENCE_VISIBLE
    assert r.is_exposed(0.45, 1, defect_type="crack")       # 블랙박스가 보낸 균열(0.42 이상)은 1대로도 보인다
    assert not r.is_exposed(0.40, 1, defect_type="crack")
    assert r.is_exposed(0.40, r.MIN_VEHICLES_VISIBLE, defect_type="crack")  # k 조건은 그대로 (실서비스 설계)


def test_load_type_conf_reads_card_and_maps_facilities(tmp_path):
    card = tmp_path / "card.json"
    card.write_text('{"class_conf_draft": {"pothole": 0.3, "crack": 0.4, "guardrail": 0.2, "sign": 0.25}}')
    assert r.load_type_conf(str(card)) == {"pothole": 0.3, "crack": 0.4, "guardrail_damage": 0.2, "sign_damage": 0.25}
    assert r.load_type_conf(str(tmp_path / "none.json")) == {}


def test_dedup_radius_is_30m():
    """오픈 이슈 #2 (11/17): 재구성 GPS 오차가 7 m 이상이라 30 m."""
    assert r.DEDUP_DISTANCE_M["pothole"] == r.DEDUP_DISTANCE_M["crack"] == 30.0
