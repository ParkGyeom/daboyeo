from datetime import datetime, timedelta, timezone

from server import rules as r

KST = timezone(timedelta(hours=9))


def test_severity():
    assert r.severity("pothole") == "빠른 보수"
    assert r.severity("crack") == r.severity("lane_wear") == "정기 보수"


def test_visibility_conditions():
    v = r.is_visible_to_citizen
    assert v("미신고", 0.7, 1, True, False)          # 신뢰도 높음
    assert v("미신고", 0.4, 2, True, False)          # 신뢰도 낮아도 기기 2대
    assert not v("미신고", 0.4, 1, True, False)      # 신뢰도 낮고 기기 1대 -> 숨김
    assert not v("미신고", 0.9, 3, False, False)     # 그 시민의 기기가 보지 않음
    assert not v("미신고", 0.9, 3, True, True)       # 넘김
    assert not v("신고됨", 0.9, 3, True, False)
    assert not v("보수 중", 0.9, 3, True, False)
    assert v("미신고", 0.6, 1, True, False)          # 경계: 0.6은 통과


def test_transitions():
    assert r.can_transition("미신고", "신고됨")
    assert r.can_transition("신고됨", "보수 중")
    assert not r.can_transition("미신고", "보수 중")
    assert not r.can_transition("신고됨", "미신고")
    assert not r.can_transition("보수 중", "신고됨")
    assert not r.can_transition("신고됨", "신고됨")


def test_seoul_reward_and_jurisdiction():
    assert r.is_reward_eligible(37.5154, 126.9521)   # DATA.md 서울 우천 구간 시작점
    assert r.jurisdiction(37.5154, 126.9521) == "서울특별시"
    assert not r.is_reward_eligible(36.7850, 127.1736)  # 야간 구간
    assert r.jurisdiction(36.7850, 127.1736) == r.OUTSIDE_SEOUL


def test_quarter_start_and_progress():
    assert r.quarter_start(datetime(2026, 11, 14, 10, 0, tzinfo=KST)) == datetime(2026, 10, 1, tzinfo=KST)
    assert r.quarter_start(datetime(2026, 3, 31, 23, 59, tzinfo=KST)) == datetime(2026, 1, 1, tzinfo=KST)
    # UTC로 들어와도 한국 시간 기준: 2026-09-30 16:00Z = 10/1 01:00 KST
    assert r.quarter_start(datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)) == datetime(2026, 10, 1, tzinfo=KST)
    assert r.reward_progress(3) == {"count": 3, "target": 5, "remaining": 2}
    assert r.reward_progress(7)["remaining"] == 0
