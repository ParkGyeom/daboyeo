"""관리자 지도의 나눠 보기: 지역(시·군·동)·차량 번호·기간 필터와 응답 필드 (GET /api/v1/defects)."""

from datetime import timedelta

from server import db
from server.tests.conftest import START, needs_db, north, observe, start


def _ids(client, **params) -> set[str]:
    r = client.get("/api/v1/defects", params=params)
    assert r.status_code == 200, r.text
    return {d["id"] for d in r.json()}


def _setup(client) -> dict[str, str]:
    """결함 셋: a(역삼동, bb-01), b(삼성동, bb-01·bb-02), c(동 없음, bb-02, 다음 날)."""
    t1, t2 = start(client, "bb-01"), start(client, "bb-02")
    lat = 37.4979
    a = observe(client, t1)["defect_id"]
    b = observe(client, t1, lat=north(lat, 300), t_s=5.0)["defect_id"]
    assert observe(client, t2, lat=north(lat, 300), t_s=6.0)["defect_id"] == b
    next_day = (START + timedelta(days=1)).isoformat()
    c = observe(client, t2, lat=north(lat, 600), observed_at=next_day)["defect_id"]
    with db.connect() as conn:
        for did, r3 in ((a, "역삼동"), (b, "삼성동"), (c, None)):
            conn.execute("UPDATE defects SET region1 = '서울특별시', region2 = '강남구', region3 = %s WHERE id = %s",
                         (r3, did))
    return {"a": a, "b": b, "c": c}


@needs_db
def test_response_has_region_plates_and_dates(client):
    d = _setup(client)
    rows = {r["id"]: r for r in client.get("/api/v1/defects").json()}
    assert rows[d["b"]]["plates"] == ["12가 3456", "34나 5678"]
    assert rows[d["a"]]["plates"] == ["12가 3456"]
    assert rows[d["a"]]["region3"] == "역삼동" and rows[d["c"]]["region3"] is None
    assert rows[d["a"]]["first_seen_at"] and rows[d["a"]]["last_seen_at"]


@needs_db
def test_region_filters(client):
    d = _setup(client)
    assert _ids(client, region1="서울특별시") == set(d.values())
    assert _ids(client, region1="부산광역시") == set()
    assert _ids(client, region2="강남구", region3="역삼동") == {d["a"]}
    assert _ids(client, region3="") == {d["c"]}  # 동 없음


@needs_db
def test_vehicle_filter_by_plate_label_or_id(client):
    d = _setup(client)
    assert _ids(client, vehicle="12가 3456") == {d["a"], d["b"]}
    assert _ids(client, vehicle="bb-02") == {d["b"], d["c"]}
    assert _ids(client, vehicle=str(db.demo_vehicle_id("bb-02"))) == {d["b"], d["c"]}
    assert _ids(client, vehicle="99하 9999") == set()


@needs_db
def test_date_filter_in_korean_days(client):
    d = _setup(client)
    today, tomorrow = START.date(), (START + timedelta(days=1)).date()
    assert _ids(client, date_from=today, date_to=today) == {d["a"], d["b"]}
    assert _ids(client, date_from=tomorrow) == {d["c"]}
    assert _ids(client, date_to=today) == {d["a"], d["b"]}
    # 차량과 기간을 같이 주면 그 차량의 그 기간 관측만 본다: bb-01 은 다음 날 관측이 없다
    assert _ids(client, vehicle="12가 3456", date_from=tomorrow) == set()
    assert _ids(client, vehicle="34나 5678", date_from=today, date_to=today) == {d["b"]}
    assert client.get("/api/v1/defects", params={"date_from": tomorrow, "date_to": today}).status_code == 422


@needs_db
def test_users_have_plates(client):
    rows = {u["vehicle_label"]: u["vehicle_plate"] for u in client.get("/api/v1/users").json()}
    assert rows == db.DEMO_PLATE
