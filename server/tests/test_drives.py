"""주행 종료 알림: 블랙박스가 주행을 끝내면 그 시민에게만 알림이 생기고, 확인하면 사라진다. DB가 필요하다."""

from server.tests.conftest import event, needs_db, north, post

pytestmark = needs_db

A, B = 1, 2  # 시연용 시민 (bb-01, bb-02)


def _end(client, device, started_at=None):
    body = {"device_id": device}
    if started_at:
        body["started_at"] = started_at
    r = client.post("/api/drives", json=body)
    assert r.status_code == 201
    return r.json()


def _notices(client, uid):
    r = client.get(f"/api/users/{uid}/notices")
    assert r.status_code == 200
    return r.json()


def test_no_notice_before_drive_ends(client):
    post(client, event())
    assert _notices(client, A) == []


def test_drive_end_creates_notice_for_that_citizen_only(client):
    post(client, event())
    post(client, event(lat=north(37.55, 100)))
    res = _end(client, "bb-01")
    assert res["defect_count"] == 2
    got = _notices(client, A)
    assert [n["id"] for n in got] == [res["drive_id"]]
    assert got[0]["defect_count"] == 2
    assert _notices(client, B) == []


def test_count_only_this_drive_and_visible(client):
    # 이전 주행에서 본 결함은 세지 않는다
    post(client, event(at="2026-11-14T09:00:00+09:00"))
    # 이번 주행: 보여줄 결함 1건 + 신뢰도가 낮아 시민에게 안 보이는 결함 1건
    post(client, event(lat=north(37.55, 200), at="2026-11-14T10:00:05+09:00"))
    post(client, event(lat=north(37.55, 400), confidence=0.3, at="2026-11-14T10:00:06+09:00"))
    assert _end(client, "bb-01", "2026-11-14T10:00:00+09:00")["defect_count"] == 1


def test_first_detection_at_start_second_is_counted(client):
    post(client, event(at="2026-11-14T10:00:00+09:00"))
    assert _end(client, "bb-01", "2026-11-14T10:00:00+09:00")["defect_count"] == 1


def test_reported_defect_not_counted(client):
    did = post(client, event()).json()["defect_id"]
    post(client, event(device="bb-02"))
    assert client.post(f"/api/defects/{did}/report", json={"user_id": B}).status_code == 200
    assert _end(client, "bb-01")["defect_count"] == 0


def test_seen_clears_notices(client):
    post(client, event())
    _end(client, "bb-01")
    _end(client, "bb-01")
    assert len(_notices(client, A)) == 2
    r = client.post(f"/api/users/{A}/notices/seen")
    assert r.json() == {"seen": 2}
    assert _notices(client, A) == []


def test_unknown_device_still_recorded_with_zero(client):
    assert _end(client, "bb-99")["defect_count"] == 0


def test_started_at_needs_timezone(client):
    r = client.post("/api/drives", json={"device_id": "bb-01", "started_at": "2026-11-14T10:00:00"})
    assert r.status_code == 422
