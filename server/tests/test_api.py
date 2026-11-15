"""B1 완료 기준: POST /api/detections 로 보내면 observations 에 저장되고 사진 파일이 남는다. DB가 필요하다."""

import json

from server import db
from server.fake import make_events


from server.tests.conftest import needs_db, post as _post

pytestmark = needs_db


def test_demo_users_seeded(client):
    with db.connect() as conn:
        rows = conn.execute("SELECT device_id FROM users ORDER BY id").fetchall()
    assert [r["device_id"] for r in rows] == ["bb-01", "bb-02", "bb-03"]


def test_detection_saved_with_photo(client, tmp_path):
    import random
    (ev, photo), = make_events(37.55, 127.07, 1, 1, 0, 1, ["pothole"], random.Random(0))
    r = _post(client, ev, photo)
    assert r.status_code == 201
    oid = r.json()["observation_id"]
    with db.connect() as conn:
        row = conn.execute(
            "SELECT device_id, type, confidence, bbox, snapshot_path, bytes,"
            " ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lon FROM observations WHERE id=%s", (oid,)
        ).fetchone()
    assert row["type"] == "pothole" and row["device_id"] == "bb-01"
    assert abs(row["lat"] - ev["lat"]) < 1e-6 and abs(row["lon"] - ev["lon"]) < 1e-6
    assert row["bbox"] == ev["bbox"]
    assert (tmp_path / row["snapshot_path"]).read_bytes() == photo
    assert row["bytes"] > len(photo)


def test_bad_requests_rejected(client):
    import random
    (ev, photo), = make_events(37.55, 127.07, 1, 1, 0, 1, ["crack"], random.Random(0))
    assert _post(client, {**ev, "type": "manhole"}, photo).status_code == 422
    assert _post(client, ev, b"not a jpeg").status_code == 422
