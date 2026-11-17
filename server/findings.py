"""주행 후 확인 목록 (findings, PRD F6). 주행이 끝나면 그 주행에서 관측한 결함 중 시민에게 보여줄 것을 목록에 넣는다.

- 보여줄 것: 아직 신고되지 않았고 노출 기준(rules.is_exposed)을 넘는 결함.
- 같은 시민에게 같은 결함은 한 번만 묻는다 (findings 의 UNIQUE(user_id, defect_id), 오픈 이슈 #8).
- 노출 기준은 다른 차량의 관측이 늘면 나중에 넘을 수 있다. 그래서 목록을 읽을 때마다 끝난 주행을 다시 훑는다.
"""

from __future__ import annotations

import json
from uuid import UUID

import psycopg

from server import db, rules

# rules.is_exposed 와 같은 조건 (SQL). θ 는 유형별 (rules.CONFIDENCE_VISIBLE_BY_TYPE, JSON 으로 넘긴다)
_EXPOSED = """
    ((d.is_suspected AND d.vehicle_count >= %(k)s)
     OR (NOT d.is_suspected AND (
         d.confidence >= COALESCE((%(theta_by_type)s::jsonb ->> d.defect_type::text)::real, %(theta)s)
         OR d.vehicle_count >= %(k)s)))
"""


def _params(**kw) -> dict:
    return {"k": rules.MIN_VEHICLES_VISIBLE, "theta": rules.CONFIDENCE_VISIBLE,
            "theta_by_type": json.dumps(rules.CONFIDENCE_VISIBLE_BY_TYPE), **kw}


def refresh_trip(conn: psycopg.Connection, trip_id: UUID) -> int:
    """끝난 주행 하나의 목록을 채운다. 새로 넣은 항목 수를 돌려준다. 끝나지 않은 주행은 0."""
    return conn.execute(
        f"""
        INSERT INTO findings (trip_id, user_id, defect_id)
        SELECT DISTINCT t.id, v.owner_user_id, d.id
        FROM trips t
        JOIN vehicles v ON v.id = t.vehicle_id
        JOIN observations o ON o.trip_id = t.id
        JOIN defects d ON d.id = o.defect_id
        WHERE t.id = %(trip)s AND t.ended_at IS NOT NULL AND d.status = 'unreported' AND {_EXPOSED}
        ON CONFLICT (user_id, defect_id) DO NOTHING
        """,
        _params(trip=trip_id),
    ).rowcount


def refresh_user(conn: psycopg.Connection, user_id: UUID) -> None:
    trips = conn.execute(
        """SELECT t.id FROM trips t JOIN vehicles v ON v.id = t.vehicle_id
           WHERE v.owner_user_id = %s AND t.ended_at IS NOT NULL""",
        (user_id,),
    ).fetchall()
    for t in trips:
        refresh_trip(conn, t["id"])


def pending(conn: psycopg.Connection, user_id: UUID, trip_id: UUID | None = None) -> list[dict]:
    """그 시민이 아직 답하지 않은 항목 (결함 정보 포함). 신뢰도 높은 순."""
    return conn.execute(
        f"""
        SELECT f.id AS finding_id, f.trip_id, {db.DEFECT_COLS}
        FROM findings f JOIN defects d ON d.id = f.defect_id
        WHERE f.user_id = %s AND f.response = 'pending' AND d.status = 'unreported'
          AND (%s::uuid IS NULL OR f.trip_id = %s)
        ORDER BY d.confidence DESC, d.first_seen_at
        """,
        (user_id, trip_id, trip_id),
    ).fetchall()


def notices(conn: psycopg.Connection, user_id: UUID) -> list[dict]:
    """끝난 주행 중 아직 답하지 않은 항목이 남은 것. 최근 것부터. 다 답하면 알림이 사라진다."""
    return conn.execute(
        """
        SELECT t.id AS trip_id, t.started_at, t.ended_at, v.label AS vehicle_label,
               count(*) FILTER (WHERE f.response = 'pending' AND d.status = 'unreported') AS pending_count,
               count(f.id) AS finding_count
        FROM trips t
        JOIN vehicles v ON v.id = t.vehicle_id
        LEFT JOIN findings f ON f.trip_id = t.id
        LEFT JOIN defects d ON d.id = f.defect_id
        WHERE v.owner_user_id = %s AND t.ended_at IS NOT NULL
        GROUP BY t.id, v.label
        HAVING count(*) FILTER (WHERE f.response = 'pending' AND d.status = 'unreported') > 0
        ORDER BY t.ended_at DESC
        """,
        (user_id,),
    ).fetchall()
