"""노출 조건, 신고, 중복 신고 거절, 상태 전환, 포상 현황 (B3 완료 기준). DB가 필요하다."""

from server.tests.conftest import event, needs_db, north, post

pytestmark = needs_db

A, B, C = 1, 2, 3  # 시연용 시민 (bb-01, bb-02, bb-03)


def _pending(client, uid):
    r = client.get(f"/api/users/{uid}/pending")
    assert r.status_code == 200
    return [d["id"] for d in r.json()]


def test_low_confidence_single_device_hidden(client):
    low = post(client, event(confidence=0.4)).json()["defect_id"]
    high = post(client, event(lat=north(37.55, 100), confidence=0.8)).json()["defect_id"]
    assert _pending(client, A) == [high]
    # 다른 기기가 같은 결함을 보면 신뢰도가 낮아도 보인다
    post(client, event(confidence=0.4, device="bb-02"))
    assert set(_pending(client, A)) == {low, high}
    assert _pending(client, B) == [low]      # B의 기기는 low 만 봤다
    assert _pending(client, C) == []         # C의 기기는 아무것도 못 봤다


def test_report_then_second_report_rejected(client):
    did = post(client, event()).json()["defect_id"]
    post(client, event(device="bb-02"))
    r = client.post(f"/api/defects/{did}/report", json={"user_id": A})
    assert r.status_code == 200
    rep = r.json()
    assert rep["status"] == "신고됨" and rep["body"]["submission"] == "제출됨(모의)"
    assert rep["body"]["reporter"] == "시민 A" and rep["body"]["device_count"] == 2

    r2 = client.post(f"/api/defects/{did}/report", json={"user_id": B})
    assert r2.status_code == 409 and r2.json()["detail"] == "이미 신고됨"
    # 신고된 결함은 다른 시민의 목록에서 사라진다
    assert did not in _pending(client, B)
    # 신고서 조회
    got = client.get(f"/api/reports/{rep['report_id']}").json()
    assert got["defect_id"] == did and got["body"]["defect_type"] == "pothole"
    # 지도 대시보드에서 신고됨으로 보인다
    assert client.get("/api/defects", params={"status": "신고됨"}).json()[0]["id"] == did


def test_cannot_report_defect_not_observed(client):
    did = post(client, event(device="bb-01")).json()["defect_id"]
    assert client.post(f"/api/defects/{did}/report", json={"user_id": C}).status_code == 403
    assert client.post("/api/defects/9999/report", json={"user_id": A}).status_code == 404


def test_skip_hides_only_for_that_user(client):
    did = post(client, event()).json()["defect_id"]
    post(client, event(device="bb-02"))
    assert client.post(f"/api/defects/{did}/skip", json={"user_id": A}).status_code == 200
    assert did not in _pending(client, A)
    assert did in _pending(client, B)


def test_status_transitions(client):
    did = post(client, event()).json()["defect_id"]
    url = f"/api/defects/{did}/status"
    assert client.patch(url, json={"status": "보수 중"}).status_code == 409   # 미신고 -> 보수 중 불가
    client.post(f"/api/defects/{did}/report", json={"user_id": A})
    assert client.patch(url, json={"status": "보수 중"}).status_code == 200
    assert client.patch(url, json={"status": "미신고"}).status_code == 409
    assert client.patch(url, json={"status": "엉뚱함"}).status_code == 422


def test_reward_counts_only_eligible_reports(client):
    seoul = post(client, event(lat=37.5154, lon=126.9521)).json()["defect_id"]
    outside = post(client, event(lat=36.7850, lon=127.1736)).json()["defect_id"]
    client.post(f"/api/defects/{seoul}/report", json={"user_id": A})
    client.post(f"/api/defects/{outside}/report", json={"user_id": A})
    r = client.get(f"/api/users/{A}/reward").json()
    assert (r["count"], r["target"], r["remaining"]) == (1, 5, 4)


def test_stats(client):
    post(client, event(device="bb-01"))
    post(client, event(device="bb-02"))
    post(client, event(device="bb-02", lat=north(37.55, 200)))
    s = client.get("/api/stats").json()
    assert s["total_observations"] == 3
    assert [(d["device_id"], d["observations"]) for d in s["devices"]] == [("bb-01", 1), ("bb-02", 2)]
    assert s["total_bytes"] > 0
