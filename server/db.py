"""DB 연결과 스키마. 스키마는 server/sql/001_init.sql 한 곳에만 있다 (PRD 9장).

docker compose 의 DB 는 처음 뜰 때 001_init.sql 을 스스로 실행한다. 테스트용 DB 처럼 테이블이 없는 DB 는
init_db() 가 같은 파일을 실행한다. 시연용 시민 3명과 차량(bb-01~03)은 고정 UUID 로 넣어 몇 번 불러도 같다.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import psycopg
from psycopg import conninfo, sql
from psycopg.rows import dict_row

from server import config

SCHEMA_SQL = Path(__file__).resolve().parent / "sql" / "001_init.sql"

# 시연용 시민과 차량. 차량 이름(label)은 가상 블랙박스의 --device-id 와 같다
DEMO = [("시민 A", "bb-01"), ("시민 B", "bb-02"), ("시민 C", "bb-03")]
_NS = uuid.UUID("6f1d3c52-6b8e-4a35-9d1e-2b8f6c0a9d41")


def demo_user_id(label: str) -> uuid.UUID:
    return uuid.uuid5(_NS, f"user/{label}")


def demo_vehicle_id(label: str) -> uuid.UUID:
    return uuid.uuid5(_NS, f"vehicle/{label}")


def connect() -> psycopg.Connection:
    return psycopg.connect(config.DATABASE_URL, row_factory=dict_row)


def ensure_database(url: str) -> None:
    """url 의 DB 가 없으면 같은 DB 서버에 만든다. DB 서버가 꺼져 있으면 psycopg.OperationalError."""
    info = conninfo.conninfo_to_dict(url)
    name = info.get("dbname", "postgres")
    with psycopg.connect(conninfo.make_conninfo(**{**info, "dbname": "postgres"}), autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


class OldSchemaError(RuntimeError):
    pass


def _has_schema(conn: psycopg.Connection) -> bool:
    """001_init.sql 테이블이 있으면 True. 이전 형식 테이블(defects 는 있고 trips 는 없음)이면 OldSchemaError."""
    r = conn.execute(
        "SELECT to_regclass('public.trips') IS NOT NULL AS new, to_regclass('public.defects') IS NOT NULL AS any"
    ).fetchone()
    if r["any"] and not r["new"]:
        raise OldSchemaError(
            "DB 에 이전 형식 테이블이 있다. 새 스키마(001_init.sql)로 DB 를 다시 만든다: "
            "docker compose down -v && docker compose up -d db (이전 컨테이너 daboyeo-db-1 은 docker rm -f 로 지운다)"
        )
    return r["new"]


def seed_demo(conn: psycopg.Connection) -> None:
    for name, label in DEMO:
        uid, vid = demo_user_id(label), demo_vehicle_id(label)
        conn.execute("INSERT INTO users (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING", (uid, name))
        conn.execute(
            "INSERT INTO vehicles (id, owner_user_id, label) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
            (vid, uid, label),
        )


def init_db() -> None:
    """DB 와 테이블이 없으면 만들고(001_init.sql) 시연용 시민·차량을 넣는다. 여러 번 불러도 안전하다."""
    ensure_database(config.DATABASE_URL)
    with connect() as conn:
        if not _has_schema(conn):
            conn.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
        seed_demo(conn)


def reset_db() -> None:
    """모든 데이터를 지우고 처음 상태로 (시민·차량은 다시 넣는다)."""
    with connect() as conn:
        conn.execute("TRUNCATE reports, findings, observations, defects, trips, vehicles, users CASCADE")
        seed_demo(conn)


# 결함 한 건을 읽을 때 쓰는 열. 위치는 위도·경도로 풀고, 대표 사진은 신뢰도가 가장 높은 관측의 사진이다
DEFECT_COLS = """
    d.id, d.defect_type, ST_Y(d.location::geometry) AS lat, ST_X(d.location::geometry) AS lon,
    d.risk_level, d.status, d.observation_count, d.vehicle_count, d.confidence, d.is_suspected,
    d.reward_eligible_road, d.first_seen_at, d.last_seen_at,
    (SELECT o.snapshot_path FROM observations o
      WHERE o.defect_id = d.id AND o.snapshot_path IS NOT NULL
      ORDER BY o.confidence DESC, o.observed_at LIMIT 1) AS snapshot_path
"""
