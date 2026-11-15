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


def test_quarter_and_progress():
    now = datetime(2026, 11, 15, 1, 0, tzinfo=KST)
    assert r.quarter_label(now) == "2026-Q4"
    assert r.quarter_start(now) == datetime(2026, 10, 1, tzinfo=KST)
    # UTC 로 9월 30일 16시 = 한국 10월 1일 1시 → 4분기
    assert r.quarter_label(datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)) == "2026-Q4"
    assert r.reward_progress(3) == {"count": 3, "target": 5, "remaining": 2}
    assert r.reward_progress(7)["remaining"] == 0
