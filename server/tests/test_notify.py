"""주행 종료 모의 문자와 가입 없는 링크 신고 (docs/notify-link-plan.md). DB가 필요하다."""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from server import db, notify
from server.tests.conftest import USER, end, needs_db, observe, start

pytestmark = needs_db
A, B = USER["bb-01"], USER["bb-02"]


def _inbox(client) -> list[dict]:
    r = client.get("/api/v1/notifications")
    assert r.status_code == 200, r.text
    return r.json()


def _token_of(msg: dict) -> str:
    return re.search(r"\?t=([\w-]+)", msg["body"]).group(1)


def _setup_one(client):
    """시민 A 의 주행 하나를 끝내고 (결함 번호, 토큰)을 돌려준다."""
    t = start(client, "bb-01")
    did = observe(client, t)["defect_id"]
    assert end(client, t)["notified"] is True
    (msg,) = _inbox(client)
    return did, _token_of(msg)


def test_trip_end_creates_one_mock_sms_with_link(client):
    t = start(client, "bb-01")
    observe(client, t)
    assert _inbox(client) == []  # 주행 중에는 보내지 않는다
    assert end(client, t)["notified"] is True
    (msg,) = _inbox(client)
    assert msg["user_name"] == "시민 A"
    assert "1건" in msg["body"] and msg["url"] in msg["body"] and "/?t=" in msg["url"]
    lines = msg["body"].splitlines()
    assert lines[0] == "[다보여] 도로 결함 확인 요청"
    assert lines[1] == "" and lines[2] == "시민 A님, 주행 중 블랙박스가 도로 결함 1건을 찾았어요."
    assert "· 포트홀 1건 — 위치 확인 중" in lines  # 테스트는 주소를 받지 않는다
    assert "■ 지금까지 내 신고: 아직 없음" in lines
    assert lines[lines.index("▶ 확인·신고하기") + 1] == msg["url"]


def test_sms_counts_reports_so_far(client):
    """두 번째 주행 문자에는 첫 주행에서 낸 신고가 누적으로 들어간다. 링크 화면에도 같은 수."""
    did, token = _setup_one(client)
    assert client.get(f"/api/v1/m/{token}/findings").json()["report_count"] == 0
    assert client.post(f"/api/v1/m/{token}/defects/{did}/report").status_code == 200
    assert client.get(f"/api/v1/m/{token}/findings").json()["report_count"] == 1  # 신고하면 바로 +1
    t = start(client, "bb-01")
    observe(client, t, lat=37.5100)
    end(client, t)
    newest = _inbox(client)[0]
    assert "■ 지금까지 내 신고: 1건" in newest["body"].splitlines()


KST_NOON = datetime(2026, 11, 18, 3, 54, tzinfo=timezone.utc)  # 한국 오후 12:54


def test_compose_long_body():
    items = ([{"defect_type": "crack", "address": "서울 강남구 봉은사로 102"}] * 3
             + [{"defect_type": "crack", "address": "서울 강남구 강남대로 396"}] * 2)
    body = notify.compose("정현우", KST_NOON, KST_NOON.replace(minute=58), items, 3,
                          "http://localhost:8501/?t=abc", 72)
    assert body.splitlines() == [
        "[다보여] 도로 결함 확인 요청",
        "",
        "정현우님, 주행 중 블랙박스가 도로 결함 5건을 찾았어요.",
        "",
        "■ 주행: 11월 18일 오후 12:54~12:58",
        "■ 찾은 결함 5건",
        "· 균열 3건 — 서울 강남구 봉은사로 102 부근",
        "· 균열 2건 — 서울 강남구 강남대로 396 부근",
        "■ 지금까지 내 신고: 3건",
        "",
        "사진을 보고 신고할지 직접 정해 주세요.",
        "신고하면 본인 명의로 서울시에 접수돼요.",
        "(시연에서는 모의 접수)",
        "",
        "▶ 확인·신고하기",
        "http://localhost:8501/?t=abc",
        "",
        "링크는 72시간 동안 열려요.",
        "신고하지 않으려면 이 문자를 무시하세요.",
    ]


def test_place_lines_group_and_cut():
    items = ([{"defect_type": "pothole", "address": None}]
             + [{"defect_type": "crack", "address": "A로 1"}] * 2
             + [{"defect_type": "sign_damage", "address": ""}]
             + [{"defect_type": "crack", "address": "B로 2"}] * 3
             + [{"defect_type": "pothole", "address": "A로 1"}] * 2)
    assert notify.place_lines(items) == [
        "· 균열 3건 — B로 2 부근",
        "· 균열 2건 — A로 1 부근",
        "· 포트홀 2건 — A로 1 부근",
        "· 외 2건",  # 위치 확인 중 포트홀 1 + 주소 없는 곳 표지판 1
    ]
    assert notify.place_lines(items[:1]) == ["· 포트홀 1건 — 위치 확인 중"]


def test_drive_span_crosses_midnight_in_kst():
    a = datetime(2026, 11, 18, 14, 50, tzinfo=timezone.utc)  # 한국 오후 11:50
    b = datetime(2026, 11, 18, 15, 10, tzinfo=timezone.utc)  # 한국 19일 오전 12:10
    assert notify.drive_span(a, b) == "11월 18일 오후 11:50~11월 19일 오전 12:10"
    noon = datetime(2026, 11, 18, 2, 50, tzinfo=timezone.utc)  # 한국 오전 11:50
    assert notify.drive_span(noon, KST_NOON) == "11월 18일 오전 11:50~오후 12:54"
    assert notify.drive_span(KST_NOON, KST_NOON.replace(second=40)) == "11월 18일 오후 12:54"


def test_no_findings_no_sms(client):
    t = start(client, "bb-01")
    r = end(client, t)
    assert r["finding_count"] == 0 and r["notified"] is False
    assert _inbox(client) == []


def test_ending_twice_sends_once(client):
    t = start(client, "bb-01")
    observe(client, t)
    end(client, t)
    assert end(client, t)["notified"] is False  # 이미 보냈다
    assert len(_inbox(client)) == 1


def test_phone_is_masked_in_inbox(client):
    t = start(client, "bb-01")
    observe(client, t)
    end(client, t)
    (msg,) = _inbox(client)
    assert msg["to_phone"] == "010-****-0001"
    with db.connect() as conn:  # DB 에는 원래 번호가 있다
        assert conn.execute("SELECT to_phone FROM notifications").fetchone()["to_phone"] == "010-0000-0001"


def test_no_phone_no_sms(client):
    with db.connect() as conn:
        conn.execute("UPDATE users SET phone = NULL WHERE id = %s", (A,))
    t = start(client, "bb-01")
    observe(client, t)
    assert end(client, t)["notified"] is False


def test_link_lists_only_that_users_findings(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    da = observe(client, ta, lat=37.4979)["defect_id"]
    observe(client, tb, lat=37.5100)
    end(client, ta), end(client, tb)
    msgs = {m["user_name"]: m for m in _inbox(client)}
    r = client.get(f"/api/v1/m/{_token_of(msgs['시민 A'])}/findings")
    assert r.status_code == 200
    body = r.json()
    assert body["user_name"] == "시민 A" and [d["id"] for d in body["findings"]] == [da]
    with db.connect() as conn:
        assert conn.execute("SELECT opened_at FROM notifications WHERE user_id = %s", (A,)).fetchone()["opened_at"]


def test_one_tap_report_through_link(client):
    did, token = _setup_one(client)
    r = client.post(f"/api/v1/m/{token}/defects/{did}/report")
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["status"] == "reported" and rep["payload"]["reporter"] == "시민 A"
    assert client.get("/api/v1/defects", params={"status": "reported"}).json()[0]["id"] == did
    got = client.get(f"/api/v1/m/{token}/reports/{rep['report_id']}")
    assert got.status_code == 200 and got.json()["submission_status"] == "mock_submitted"
    assert client.get(f"/api/v1/m/{token}/findings").json()["findings"] == []
    assert client.post(f"/api/v1/m/{token}/defects/{did}/report").status_code == 409


def test_report_marks_other_users_finding_already_reported(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    did = observe(client, ta)["defect_id"]
    observe(client, tb)
    end(client, ta), end(client, tb)
    msgs = {m["user_name"]: _token_of(m) for m in _inbox(client)}
    assert client.post(f"/api/v1/m/{msgs['시민 A']}/defects/{did}/report").status_code == 200
    r = client.post(f"/api/v1/m/{msgs['시민 B']}/defects/{did}/report")
    assert r.status_code == 409 and r.json()["detail"] == "이미 신고됨"
    assert client.get(f"/api/v1/m/{msgs['시민 B']}/findings").json()["findings"] == []


def test_skip_through_link(client):
    did, token = _setup_one(client)
    assert client.post(f"/api/v1/m/{token}/defects/{did}/skip").status_code == 200
    assert client.get(f"/api/v1/m/{token}/findings").json()["findings"] == []
    assert client.post(f"/api/v1/m/{token}/defects/{did}/report").status_code == 404  # 넘긴 항목은 신고 대상이 아니다


def test_unknown_and_expired_token(client):
    did, token = _setup_one(client)
    assert client.get("/api/v1/m/not-a-real-token/findings").status_code == 404
    assert client.post(f"/api/v1/m/not-a-real-token/defects/{did}/report").status_code == 404
    with db.connect() as conn:
        conn.execute("UPDATE notifications SET expires_at = now() - interval '1 minute'")
    assert client.get(f"/api/v1/m/{token}/findings").status_code == 410
    assert client.post(f"/api/v1/m/{token}/defects/{did}/report").status_code == 410


def test_link_cannot_report_defect_outside_its_list(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    observe(client, ta, lat=37.4979)
    other = observe(client, tb, lat=37.5100)["defect_id"]
    end(client, ta), end(client, tb)
    msgs = {m["user_name"]: _token_of(m) for m in _inbox(client)}
    r = client.post(f"/api/v1/m/{msgs['시민 A']}/defects/{other}/report")
    assert r.status_code == 404


def test_other_users_report_not_readable_with_my_token(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    observe(client, ta, lat=37.4979)
    db_b = observe(client, tb, lat=37.5100)["defect_id"]
    end(client, ta), end(client, tb)
    msgs = {m["user_name"]: _token_of(m) for m in _inbox(client)}
    rep = client.post(f"/api/v1/m/{msgs['시민 B']}/defects/{db_b}/report").json()
    assert client.get(f"/api/v1/m/{msgs['시민 A']}/reports/{rep['report_id']}").status_code == 404


def test_concurrent_link_reports_make_one(client):
    ta, tb = start(client, "bb-01"), start(client, "bb-02")
    did = observe(client, ta)["defect_id"]
    observe(client, tb)
    end(client, ta), end(client, tb)
    tokens = [_token_of(m) for m in _inbox(client)]
    assert len(tokens) == 2
    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(lambda tk: client.post(f"/api/v1/m/{tk}/defects/{did}/report").status_code, tokens * 5))
    assert codes.count(200) == 1 and codes.count(409) == 9


def test_notify_schema_is_idempotent(client):
    db.init_db()  # 이미 적용된 DB 에 002 를 다시 적용해도 오류가 없다
    db.init_db()
