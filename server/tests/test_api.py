"""관측 수신 (POST /api/v1/observations, ObservationIn) 과 주행 열기, 잘못된 요청 거절. DB가 필요하다."""

import json

from server import db
from server.tests.conftest import end, meta, needs_db, post, start

pytestmark = needs_db


def test_demo_users_and_vehicles_seeded(client):
    rows = client.get("/api/v1/users").json()
    assert [(r["name"], r["vehicle_label"]) for r in rows] == [("시민 A", "bb-01"), ("시민 B", "bb-02"), ("시민 C", "bb-03")]
    assert rows[0]["id"] == str(db.demo_user_id("bb-01"))


def test_observation_saved_with_photo_and_stats(client, tmp_path):
    t = start(client)
    m = meta(t, confidence=0.77, t_s=12.5)
    r = post(client, m, photo=b"\xff\xd8 jpeg-ish")
    assert r.status_code == 201, r.text
    with db.connect() as conn:
        row = conn.execute(
            "SELECT defect_type::text AS t, confidence, bbox, t_s, snapshot_path, snapshot_bytes,"
            " ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lon FROM observations WHERE id = %s",
            (r.json()["observation_id"],),
        ).fetchone()
    assert row["t"] == "pothole" and abs(row["confidence"] - 0.77) < 1e-6 and row["t_s"] == 12.5
    assert abs(row["lat"] - m["lat"]) < 1e-6 and row["bbox"] == m["bbox"]
    assert (tmp_path / row["snapshot_path"]).read_bytes() == b"\xff\xd8 jpeg-ish"
    s = client.get("/api/v1/stats").json()
    assert s["vehicles"] == [{"vehicle_label": "bb-01", "observations": 1, "bytes": row["snapshot_bytes"]}]


def test_observation_with_clip_counts_bytes(client, tmp_path):
    t = start(client)
    clip = b"\x00\x00\x00\x18ftypisom" + b"x" * 100
    r = post(client, meta(t), photo=b"\xff\xd8 jpeg-ish", clip=clip)
    assert r.status_code == 201, r.text
    with db.connect() as conn:
        row = conn.execute("SELECT clip_path, clip_bytes, snapshot_bytes FROM observations WHERE id = %s",
                           (r.json()["observation_id"],)).fetchone()
    assert row["clip_path"].endswith(".mp4") and row["clip_bytes"] == len(clip)
    assert (tmp_path / row["clip_path"]).read_bytes() == clip
    assert client.get("/api/v1/stats").json()["total_bytes"] == len(clip) + row["snapshot_bytes"]
    assert post(client, meta(t), clip=b"not an mp4 file").status_code == 422


def test_lane_wear_without_bbox_or_photo(client):
    t = start(client)
    assert post(client, meta(t, type="lane_wear"), photo=None).status_code == 201


def test_bad_observations_rejected(client, tmp_path):
    t = start(client)
    other = start(client, "bb-02")
    assert post(client, meta(t, confidence=1.5)).status_code == 422
    assert post(client, meta(t, type="manhole")).status_code == 422
    assert post(client, meta(t), photo=b"not a jpeg").status_code == 422
    assert post(client, meta(t, observed_at="2026-11-20T10:00:00")).status_code == 422          # 시간대 없음
    assert post(client, meta(t, vehicle_id=other["vehicle_id"])).status_code == 422             # 다른 차량
    assert post(client, meta(t, trip_id="00000000-0000-0000-0000-000000000000")).status_code == 404
    r = client.post("/api/v1/observations", data={"meta": "{not json"})
    assert r.status_code == 422
    assert list(tmp_path.iterdir()) == []  # 거절된 요청의 사진은 남지 않는다
    with db.connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM observations").fetchone()["n"] == 0


def test_trip_start_and_end(client):
    assert client.post("/api/v1/trips", json={}).status_code == 422
    assert client.post("/api/v1/trips", json={"vehicle_label": "bb-09"}).status_code == 404
    assert client.post("/api/v1/trips", json={"vehicle_label": "bb-01",
                                              "started_at": "2026-11-20T10:00:00"}).status_code == 422
    t = start(client)
    e1 = end(client, t)
    e2 = end(client, t)  # 두 번 닫아도 같은 시각
    assert e1["ended_at"] == e2["ended_at"]
    assert client.post("/api/v1/trips/00000000-0000-0000-0000-000000000000/end").status_code == 404
    assert client.get(f"/api/v1/trips/{t['trip_id']}/findings").json() == []


def test_meta_is_contracts_observation(client):
    """블랙박스가 보내는 것과 같은 모양: contracts.ObservationIn 의 JSON."""
    from daboyeo_contracts import ObservationIn

    t = start(client)
    obs = ObservationIn.model_validate(meta(t))
    r = client.post("/api/v1/observations", data={"meta": obs.model_dump_json()})
    assert r.status_code == 201
    assert json.loads(obs.model_dump_json())["defect_type"] == "pothole"
