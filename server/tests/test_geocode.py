"""결함 주소 (server/geocode.py): 응답 해석, 새 결함에 한 번 저장, 실패하면 나중에 다시, 지역(시·군·동)."""

from server import db, geocode
from server.tests.conftest import end, needs_db, observe, start, USER

ROAD = {
    "status": {"code": 0, "name": "ok"},
    "results": [
        {"name": "addr", "region": {"area1": {"name": "서울특별시", "alias": "서울"}, "area2": {"name": "강남구"},
                                    "area3": {"name": "역삼동"}, "area4": {"name": ""}},
         "land": {"type": "1", "number1": "825", "number2": "2"}},
        {"name": "roadaddr", "region": {"area1": {"name": "서울특별시", "alias": "서울"}, "area2": {"name": "강남구"},
                                        "area3": {"name": "역삼동"}},
         "land": {"name": "강남대로", "number1": "396", "number2": ""}},
    ],
}


def test_format_prefers_road_address():
    assert geocode.format_address(ROAD) == "서울 강남구 강남대로 396"


def test_format_falls_back_to_lot_address():
    body = {"results": [ROAD["results"][0]]}
    assert geocode.format_address(body) == "서울 강남구 역삼동 825-2"
    assert geocode.format_address({"results": []}) is None


def test_region_takes_dong_from_lot_address():
    # 도로명 주소에는 동이 없다. 동은 지번(addr) 결과의 area3
    assert geocode.parse_region(ROAD) == ("서울특별시", "강남구", "역삼동")
    assert geocode.parse(ROAD) == geocode.Place("서울 강남구 강남대로 396", "서울특별시", "강남구", "역삼동")
    road_only = {"results": [{**ROAD["results"][1], "region": {"area1": {"name": "서울특별시", "alias": "서울"},
                                                               "area2": {"name": "강남구"}, "area3": {"name": ""}}}]}
    assert geocode.parse(road_only) == geocode.Place("서울 강남구 강남대로 396", "서울특별시", "강남구", None)
    assert geocode.parse({"results": []}) == geocode.NOWHERE


PLACE = geocode.Place("서울 강남구 강남대로 396", "서울특별시", "강남구", "역삼동")


@needs_db
def test_new_defect_gets_address_once_and_report_has_it(client, monkeypatch):
    calls = []
    monkeypatch.setattr(geocode, "keys", lambda: ("id", "secret"))
    monkeypatch.setattr(geocode, "reverse", lambda lat, lon: calls.append((lat, lon)) or PLACE)
    trip = start(client)
    observe(client, trip)
    observe(client, trip, t_s=2.0)  # 같은 결함. 다시 부르지 않는다
    [d] = client.get("/api/v1/defects").json()
    assert d["address"] == "서울 강남구 강남대로 396"
    assert (d["region1"], d["region2"], d["region3"]) == ("서울특별시", "강남구", "역삼동")
    assert len(calls) == 1
    end(client, trip)
    r = client.post(f"/api/v1/defects/{d['id']}/report", json={"user_id": USER["bb-01"]})
    assert r.status_code == 200, r.text
    assert r.json()["payload"]["location"]["address"] == "서울 강남구 강남대로 396"


@needs_db
def test_failed_lookup_is_retried_later(client, monkeypatch):
    answer = {"v": None}
    monkeypatch.setattr(geocode, "keys", lambda: ("id", "secret"))
    monkeypatch.setattr(geocode, "reverse", lambda lat, lon: answer["v"])
    trip = start(client)
    observe(client, trip)
    [d] = client.get("/api/v1/defects").json()
    assert d["address"] is None  # 실패해도 결함은 그대로 생긴다
    answer["v"] = PLACE
    assert geocode.fill_missing() == 0  # RETRY_S 안에는 다시 부르지 않는다
    with db.connect() as conn:
        conn.execute("UPDATE defects SET address_tried_at = now() - interval '1 hour'")
    assert geocode.fill_missing() == 1
    assert client.get("/api/v1/defects").json()[0]["address"] == "서울 강남구 강남대로 396"


@needs_db
def test_no_result_place_is_not_asked_again(client, monkeypatch):
    calls = []
    monkeypatch.setattr(geocode, "keys", lambda: ("id", "secret"))
    monkeypatch.setattr(geocode, "reverse", lambda lat, lon: calls.append(1) or geocode.NOWHERE)  # 바다 위
    trip = start(client)
    observe(client, trip)
    with db.connect() as conn:
        conn.execute("UPDATE defects SET address_tried_at = now() - interval '1 hour'")
    [d] = client.get("/api/v1/defects").json()
    assert d["address"] == ""
    assert len(calls) == 1


@needs_db
def test_address_without_region_is_filled_again(client, monkeypatch):
    """004 전에 주소만 받아 둔 결함도 지역을 채운다. 다시 못 받으면 주소는 그대로 둔다."""
    answer = {"v": None}
    monkeypatch.setattr(geocode, "keys", lambda: ("id", "secret"))
    monkeypatch.setattr(geocode, "reverse", lambda lat, lon: answer["v"])
    trip = start(client)
    observe(client, trip)
    with db.connect() as conn:
        conn.execute("UPDATE defects SET address = '서울 강남구 강남대로 396', address_tried_at = now() - interval '1 hour'")
    assert geocode.fill_missing() == 0  # 인터넷 없음
    [d] = client.get("/api/v1/defects").json()
    assert d["address"] == "서울 강남구 강남대로 396" and d["region1"] is None
    with db.connect() as conn:
        conn.execute("UPDATE defects SET address_tried_at = now() - interval '1 hour'")
    answer["v"] = PLACE
    assert geocode.fill_missing() == 1
    [d] = client.get("/api/v1/defects").json()
    assert d["region3"] == "역삼동"
    assert geocode.fill_missing() == 0  # 다 채웠으면 다시 부르지 않는다


def test_address_without_province_gets_empty_region1():
    """시·도가 없는 응답이면 region1 을 "" 로 저장해 다시 묻지 않게 한다."""
    body = {"results": [{"name": "roadaddr", "region": {}, "land": {"name": "강남대로", "number1": "396"}}]}
    assert geocode.parse(body) == geocode.Place("강남대로 396", "", None, None)


@needs_db
def test_no_result_does_not_wipe_saved_address(client, monkeypatch):
    monkeypatch.setattr(geocode, "keys", lambda: ("id", "secret"))
    monkeypatch.setattr(geocode, "reverse", lambda lat, lon: geocode.NOWHERE)
    trip = start(client)
    observe(client, trip)
    with db.connect() as conn:
        conn.execute("UPDATE defects SET address = '서울 강남구 강남대로 396', region1 = NULL, "
                     "address_tried_at = now() - interval '1 hour'")
    geocode.fill_missing()
    [d] = client.get("/api/v1/defects").json()
    assert d["address"] == "서울 강남구 강남대로 396"
