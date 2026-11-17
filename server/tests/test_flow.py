"""주행 후 확인 목록, 노출 기준, 신고·중복 신고 거절, 넘기기, 상태 전환, 포상 (PRD F5~F8). DB가 필요하다."""

from concurrent.futures import ThreadPoolExecutor

from server.tests.conftest import USER, end, needs_db, north, observe, start

pytestmark = needs_db
A, B, C = USER["bb-01"], USER["bb-02"], USER["bb-03"]
LAT = 37.4979


def _findings(client, uid):
    r = client.get(f"/api/v1/users/{uid}/findings")
    assert r.status_code == 200, r.text
    return [d["id"] for d in r.json()]


def _notices(client, uid):
    return client.get(f"/api/v1/users/{uid}/notices").json()


def test_nothing_until_trip_ends(client):
    t = start(client)
    observe(client, t)
    assert _findings(client, A) == [] and _notices(client, A) == []
    assert end(client, t)["finding_count"] == 1
    assert len(_findings(client, A)) == 1
    (n,) = _notices(client, A)
    assert n["trip_id"] == t["trip_id"] and n["pending_count"] == 1 and n["vehicle_label"] == "bb-01"
    assert _findings(client, B) == []  # 다른 시민에게는 없다


def test_low_confidence_single_vehicle_hidden_until_second_vehicle(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    # θ 는 유형별 (포트홀 0.31, 오픈 이슈 #3). 그보다 낮은 신뢰도
    low = observe(client, ta, confidence=0.2)["defect_id"]
    high = observe(client, ta, lat=north(LAT, 100), confidence=0.8)["defect_id"]
    end(client, ta)
    assert _findings(client, A) == [high]
    # 다른 차량이 같은 결함을 보면 신뢰도가 낮아도 보인다 (A 의 끝난 주행도 다시 훑는다)
    observe(client, tb, confidence=0.2)
    end(client, tb)
    assert set(_findings(client, A)) == {low, high}
    assert _findings(client, B) == [low]
    assert _findings(client, C) == []


def test_report_then_second_report_rejected(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    did = observe(client, ta)["defect_id"]
    observe(client, tb)
    end(client, ta), end(client, tb)
    r = client.post(f"/api/v1/defects/{did}/report", json={"user_id": A})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["status"] == "reported"
    assert rep["payload"]["reporter"] == "시민 A" and rep["payload"]["vehicle_count"] == 2
    assert rep["payload"]["location"]["jurisdiction"] == "서울특별시"

    r2 = client.post(f"/api/v1/defects/{did}/report", json={"user_id": B})
    assert r2.status_code == 409 and r2.json()["detail"] == "이미 신고됨"
    # 신고된 결함은 다른 시민의 목록과 알림에서도 사라진다
    assert did not in _findings(client, B) and _notices(client, B) == []
    got = client.get(f"/api/v1/reports/{rep['report_id']}").json()
    assert got["defect_id"] == did and got["submission_status"] == "mock_submitted" and got["quarter"].startswith("20")
    assert client.get("/api/v1/defects", params={"status": "reported"}).json()[0]["id"] == did
    assert [x["id"] for x in client.get(f"/api/v1/users/{A}/reports").json()] == [rep["report_id"]]


def test_concurrent_reports_make_one(client):
    trips = [start(client, label) for label in ("bb-01", "bb-02", "bb-03")]
    did = [observe(client, t) for t in trips][0]["defect_id"]
    for t in trips:
        end(client, t)
    users = [A, B, C] * 7
    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(lambda u: client.post(f"/api/v1/defects/{did}/report", json={"user_id": u}).status_code,
                            users[:20]))
    assert codes.count(200) == 1 and codes.count(409) == 19


def test_cannot_report_defect_not_observed(client):
    t = start(client, "bb-01")
    did = observe(client, t)["defect_id"]
    assert client.post(f"/api/v1/defects/{did}/report", json={"user_id": C}).status_code == 403
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.post(f"/api/v1/defects/{missing}/report", json={"user_id": A}).status_code == 404


def test_skip_hides_only_for_that_user_and_is_not_asked_again(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    did = observe(client, ta)["defect_id"]
    observe(client, tb)
    end(client, ta), end(client, tb)
    assert client.post(f"/api/v1/defects/{did}/skip", json={"user_id": A}).status_code == 200
    assert did not in _findings(client, A)
    assert did in _findings(client, B)
    # 새 주행에서 또 봐도 다시 묻지 않는다
    t2 = start(client, "bb-01")
    observe(client, t2)
    end(client, t2)
    assert did not in _findings(client, A)
    # 목록에 없는 결함은 넘길 수 없다
    assert client.post(f"/api/v1/defects/{did}/skip", json={"user_id": C}).status_code == 404


def test_status_transitions(client):
    t = start(client)
    did = observe(client, t)["defect_id"]
    end(client, t)
    patch = lambda s: client.patch(f"/api/v1/defects/{did}/status", json={"status": s}).status_code  # noqa: E731
    assert patch("under_repair") == 409            # 미신고 → 보수 중 은 안 된다
    assert patch("nonsense") == 422
    client.post(f"/api/v1/defects/{did}/report", json={"user_id": A})
    assert patch("under_repair") == 200
    assert patch("reported") == 409


def test_reward_counts_only_eligible_reports(client):
    t = start(client)
    seoul = observe(client, t)["defect_id"]
    outside = observe(client, t, type="crack", lat=36.78, lon=127.17)["defect_id"]
    end(client, t)
    for did in (seoul, outside):
        assert client.post(f"/api/v1/defects/{did}/report", json={"user_id": A}).status_code == 200
    r = client.get(f"/api/v1/users/{A}/reward").json()
    assert (r["count"], r["remaining"]) == (1, 4)


def test_list_filters_and_jurisdiction(client):
    t = start(client)
    observe(client, t)
    observe(client, t, type="crack", lat=36.78, lon=127.17)
    rows = client.get("/api/v1/defects").json()
    assert {d["jurisdiction"] for d in rows} == {"서울특별시", "서울 밖 (관할 미확정)"}
    assert all(d["snapshot_path"] for d in rows)
    assert [d["defect_type"] for d in client.get("/api/v1/defects", params={"type": "crack"}).json()] == ["crack"]
    assert client.get("/api/v1/defects", params={"type": "x"}).status_code == 422
