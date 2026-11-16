"""중복 정리 (docs/DESIGN.md 6장). 탐지 결과 한 건을 받아 결함에 붙이거나 새 결함을 만든다.

알고 있는 한계: 거리 기준 안에 있는 서로 다른 결함 둘은 하나로 묶인다.
같은 기기가 연속 프레임에서 같은 결함을 두 번 보는 경우도 이 규칙으로 하나가 된다.
"""

from __future__ import annotations

import psycopg

from server import rules
from server.schema import DetectionEvent

_POINT = "ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography"

# 결함 번호를 정하는 동안 다른 요청이 같은 위치에 결함을 또 만들지 못하게 막는 잠금 번호
_LOCK_KEY = 7301


def ingest(conn: psycopg.Connection, ev: DetectionEvent, snapshot_path: str, nbytes: int) -> tuple[int, int, bool]:
    """탐지 결과를 저장하고 (관측 번호, 결함 번호, 새 결함인지)를 돌려준다. 호출한 쪽이 커밋한다."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))

    near = conn.execute(
        f"""
        SELECT id FROM defects
        WHERE type = %s AND ST_DWithin(location, {_POINT}, %s)
        ORDER BY ST_Distance(location, {_POINT}) LIMIT 1
        """,
        (ev.type, ev.lon, ev.lat, rules.DEDUP_DISTANCE_M[ev.type], ev.lon, ev.lat),
    ).fetchone()

    is_new = near is None
    if is_new:
        defect_id = conn.execute(
            f"""
            INSERT INTO defects (type, location, severity, first_seen, last_seen, max_confidence,
                                 reward_eligible, jurisdiction)
            VALUES (%s, {_POINT}, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (
                ev.type, ev.lon, ev.lat, rules.severity(ev.type), ev.detected_at, ev.detected_at,
                ev.confidence, rules.is_reward_eligible(ev.lat, ev.lon), rules.jurisdiction(ev.lat, ev.lon),
            ),
        ).fetchone()["id"]
    else:
        defect_id = near["id"]

    obs_id = conn.execute(
        f"""
        INSERT INTO observations
          (device_id, detected_at, location, type, confidence, bbox, snapshot_path, defect_id, bytes)
        VALUES (%s, %s, {_POINT}, %s, %s, %s, %s, %s, %s) RETURNING id
        """,
        (ev.device_id, ev.detected_at, ev.lon, ev.lat, ev.type, ev.confidence, list(ev.bbox),
         snapshot_path, defect_id, nbytes),
    ).fetchone()["id"]

    # 관측 수, 기기 수, 시각, 위치(관측들의 평균), 최고 신뢰도와 대표 사진을 관측들로부터 다시 계산한다
    conn.execute(
        """
        UPDATE defects d SET
            observation_count  = s.n,
            device_count       = s.devices,
            first_seen         = s.first_seen,
            last_seen          = s.last_seen,
            max_confidence     = s.max_conf,
            location           = ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326)::geography,
            best_snapshot_path = (SELECT snapshot_path FROM observations
                                  WHERE defect_id = d.id ORDER BY confidence DESC, id LIMIT 1)
        FROM (
            SELECT count(*) AS n, count(DISTINCT device_id) AS devices,
                   min(detected_at) AS first_seen, max(detected_at) AS last_seen,
                   max(confidence) AS max_conf,
                   avg(ST_Y(location::geometry)) AS lat, avg(ST_X(location::geometry)) AS lon
            FROM observations WHERE defect_id = %s
        ) s
        WHERE d.id = %s
        """,
        (defect_id, defect_id),
    )
    return obs_id, defect_id, is_new
