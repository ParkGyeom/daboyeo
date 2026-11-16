"""DB 연결과 테이블 (docs/DESIGN.md 6장). PostgreSQL/PostGIS."""

from __future__ import annotations

import psycopg
from psycopg import conninfo, sql
from psycopg.rows import dict_row

from server import config

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS users (
    id         serial PRIMARY KEY,
    name       text NOT NULL,
    device_id  text NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS defects (
    id                  serial PRIMARY KEY,
    type                text NOT NULL,
    location            geography(Point, 4326) NOT NULL,
    severity            text NOT NULL,
    status              text NOT NULL DEFAULT '미신고',
    first_seen          timestamptz NOT NULL,
    last_seen           timestamptz NOT NULL,
    observation_count   integer NOT NULL DEFAULT 1,
    device_count        integer NOT NULL DEFAULT 1,
    max_confidence      real NOT NULL,
    best_snapshot_path  text,
    reward_eligible     boolean NOT NULL DEFAULT false,
    jurisdiction        text
);

CREATE TABLE IF NOT EXISTS observations (
    id             serial PRIMARY KEY,
    device_id      text NOT NULL,
    detected_at    timestamptz NOT NULL,
    location       geography(Point, 4326) NOT NULL,
    type           text NOT NULL,
    confidence     real NOT NULL,
    bbox           integer[] NOT NULL,
    snapshot_path  text,
    defect_id      integer REFERENCES defects(id),
    bytes          integer NOT NULL,
    received_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS observations_defect_idx ON observations(defect_id);
CREATE INDEX IF NOT EXISTS defects_location_idx ON defects USING gist(location);

CREATE TABLE IF NOT EXISTS reports (
    id          serial PRIMARY KEY,
    defect_id   integer NOT NULL UNIQUE REFERENCES defects(id),
    user_id     integer NOT NULL REFERENCES users(id),
    created_at  timestamptz NOT NULL DEFAULT now(),
    body        jsonb NOT NULL
);

CREATE TABLE IF NOT EXISTS skips (
    user_id    integer NOT NULL REFERENCES users(id),
    defect_id  integer NOT NULL REFERENCES defects(id),
    PRIMARY KEY (user_id, defect_id)
);

-- 주행 종료. 가상 블랙박스가 구간 재생을 끝내면 한 줄 생긴다.
-- 시민 화면은 seen_at 이 비어 있는 줄을 "주행이 끝났어요" 알림으로 보여준다.
CREATE TABLE IF NOT EXISTS drives (
    id            serial PRIMARY KEY,
    device_id     text NOT NULL,
    started_at    timestamptz,
    ended_at      timestamptz NOT NULL DEFAULT now(),
    defect_count  integer NOT NULL,
    seen_at       timestamptz
);
CREATE INDEX IF NOT EXISTS drives_device_idx ON drives(device_id);
"""

# 시연용 시민 3명. 기기 번호는 가상 블랙박스의 --device-id 와 같다.
DEMO_USERS = [("시민 A", "bb-01"), ("시민 B", "bb-02"), ("시민 C", "bb-03")]


def connect() -> psycopg.Connection:
    return psycopg.connect(config.DATABASE_URL, row_factory=dict_row)


def ensure_database(url: str) -> None:
    """url 의 DB 가 없으면 같은 DB 서버에 만든다. DB 서버가 꺼져 있으면 psycopg.OperationalError."""
    info = conninfo.conninfo_to_dict(url)
    name = info.get("dbname", "postgres")
    with psycopg.connect(conninfo.make_conninfo(**{**info, "dbname": "postgres"}), autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def init_db() -> None:
    """DB·테이블이 없으면 만들고 시연용 시민을 넣는다. 여러 번 불러도 안전하다."""
    ensure_database(config.DATABASE_URL)
    with connect() as conn:
        conn.execute(SCHEMA)
        for name, device_id in DEMO_USERS:
            conn.execute(
                "INSERT INTO users (name, device_id) VALUES (%s, %s) ON CONFLICT (device_id) DO NOTHING",
                (name, device_id),
            )


def reset_db() -> None:
    """모든 데이터를 지우고 처음 상태로 되돌린다 (시민은 다시 넣는다)."""
    with connect() as conn:
        conn.execute("TRUNCATE drives, skips, reports, observations, defects, users RESTART IDENTITY CASCADE")
    init_db()


# 결함 한 건을 읽을 때 쓰는 열. 위치는 위도·경도로 풀어서 돌려준다.
DEFECT_COLS = """
    d.id, d.type, ST_Y(d.location::geometry) AS lat, ST_X(d.location::geometry) AS lon,
    d.severity, d.status, d.first_seen, d.last_seen, d.observation_count, d.device_count,
    d.max_confidence, d.best_snapshot_path, d.reward_eligible, d.jurisdiction
"""
