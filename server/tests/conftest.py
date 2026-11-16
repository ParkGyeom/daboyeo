"""DB가 필요한 테스트의 공용 준비물.

테스트는 시연용 DB 가 아니라 따로 둔 테스트용 DB 를 쓴다(테스트마다 비우기 때문이다).
테스트용 DB 주소는 TEST_DATABASE_URL, 없으면 시연용 DB 이름 뒤에 _test 를 붙인 것. 없으면 만들고,
이전 형식 테이블이 남아 있으면 테스트용 DB 의 스키마를 지우고 001_init.sql 로 다시 만든다.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from psycopg import conninfo

from server import config, db
from server.fake import snapshot_jpeg

KST = timezone(timedelta(hours=9))
START = datetime(2026, 11, 20, 10, 0, tzinfo=KST)


def _test_database_url() -> str:
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    info = conninfo.conninfo_to_dict(config.DATABASE_URL)
    name = info.get("dbname", "daboyeo")
    # 이 파일은 pytest 가 한 번, 테스트 파일의 import 로 한 번 더 읽는다. 이미 바꿔 놓았으면 그대로 쓴다
    info["dbname"] = name if name.endswith("_test") else f"{name}_test"
    return conninfo.make_conninfo(**info)


def _prepare_test_db() -> bool:
    """테스트용 DB 를 준비하고 연결 주소를 그쪽으로 바꾼다. DB 서버가 꺼져 있으면 False."""
    test_url = _test_database_url()
    try:
        db.ensure_database(test_url)
        config.DATABASE_URL = test_url
        try:
            db.init_db()
        except db.OldSchemaError:
            with db.connect() as conn:  # 테스트용 DB 만 지운다
                conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
            db.init_db()
    except psycopg.OperationalError:
        return False
    return True


needs_db = pytest.mark.skipif(not _prepare_test_db(), reason="DB가 떠 있지 않다 (docker compose up -d db)")

USER = {label: str(db.demo_user_id(label)) for _, label in db.DEMO}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(config, "CLIP_DIR", tmp_path)
    from server import geocode

    monkeypatch.setattr(geocode, "keys", lambda: None)  # 테스트는 네이버를 부르지 않는다 (.env 에 키가 있어도)
    from server.main import app

    db.reset_db()
    with TestClient(app) as c:
        yield c


def start(client, label="bb-01") -> dict:
    r = client.post("/api/v1/trips", json={"vehicle_label": label, "started_at": START.isoformat()})
    assert r.status_code == 201, r.text
    return r.json()


def end(client, trip) -> dict:
    r = client.post(f"/api/v1/trips/{trip['trip_id']}/end")
    assert r.status_code == 200, r.text
    return r.json()


def meta(trip, lat=37.4979, lon=127.0276, type="pothole", confidence=0.8, t_s=1.0, **kw) -> dict:
    m = {"trip_id": trip["trip_id"], "vehicle_id": trip["vehicle_id"], "defect_type": type,
         "confidence": confidence, "bbox": None if type == "lane_wear" else [0.1, 0.5, 0.3, 0.7],
         "lat": lat, "lon": lon, "t_s": t_s, "observed_at": (START + timedelta(seconds=t_s)).isoformat(),
         "client_observation_id": str(uuid.uuid4())}
    m.update(kw)
    return m


def post(client, m: dict, photo: bytes | None = b"default", clip: bytes | None = None):
    import json

    if photo == b"default":
        photo = snapshot_jpeg(m["defect_type"], m["confidence"])
    files = {"snapshot": ("s.jpg", photo, "image/jpeg")} if photo is not None else {}
    if clip is not None:
        files["clip"] = ("c.mp4", clip, "video/mp4")
    return client.post("/api/v1/observations", data={"meta": json.dumps(m)}, files=files)


def observe(client, trip, **kw) -> dict:
    r = post(client, meta(trip, **kw))
    assert r.status_code == 201, r.text
    return r.json()


def north(lat, meters):
    return lat + meters / 111_320
