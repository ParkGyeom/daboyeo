"""DB가 필요한 테스트의 공용 준비물.

테스트는 시연용 DB 가 아니라 따로 둔 테스트용 DB 를 쓴다(테스트마다 비우기 때문이다).
테스트용 DB 주소는 TEST_DATABASE_URL, 없으면 시연용 DB 이름 뒤에 _test 를 붙인 것. 없으면 만든다.
"""

import json
import os
import random

import psycopg
import pytest
from psycopg import conninfo

from server import config, db
from server.fake import make_events, snapshot_jpeg


def _test_database_url() -> str:
    if url := os.environ.get("TEST_DATABASE_URL"):
        return url
    info = conninfo.conninfo_to_dict(config.DATABASE_URL)
    name = info.get("dbname", "daboyeo")
    # 이 파일은 pytest 가 한 번, 테스트 파일의 import 로 한 번 더 읽는다. 이미 바꿔 놓았으면 그대로 쓴다
    info["dbname"] = name if name.endswith("_test") else f"{name}_test"
    return conninfo.make_conninfo(**info)


def _prepare_test_db() -> bool:
    """테스트용 DB 가 없으면 만들고, 연결 주소를 그쪽으로 바꾼다. DB 서버가 꺼져 있으면 False."""
    test_url = _test_database_url()
    try:
        db.ensure_database(test_url)
    except psycopg.OperationalError:
        return False
    config.DATABASE_URL = test_url
    return True


def db_up() -> bool:
    if not _prepare_test_db():
        return False
    try:
        db.connect().close()
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not db_up(), reason="DB가 떠 있지 않다 (docker compose up -d db)")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path)
    from server.main import app

    db.init_db()
    db.reset_db()
    with TestClient(app) as c:
        yield c


def post(client, ev, photo=None):
    photo = photo or snapshot_jpeg(ev["type"], ev["confidence"])
    return client.post("/api/detections", data={"event": json.dumps(ev)},
                       files={"snapshot": ("s.jpg", photo, "image/jpeg")})


def event(lat=37.55, lon=127.07, type="pothole", device="bb-01", confidence=0.8, at="2026-11-14T10:00:00+09:00"):
    return {"device_id": device, "detected_at": at, "lat": lat, "lon": lon, "type": type,
            "confidence": confidence, "bbox": [10, 10, 100, 100], "frame_w": 1920, "frame_h": 1080,
            "quality_ok": True}


def north(lat, meters):
    return lat + meters / 111_320
