"""관측 저장과 중복 정리 (계획 2.4 clustering). 관측 한 건을 같은 유형의 가까운 결함에 붙이거나 새 결함을 만든다.

- 재전송: client_observation_id 가 이미 있으면 새로 저장하지 않고 처음 저장한 결과를 돌려준다 (FR-2.4).
- 동시성: 결함 번호를 정하는 동안 다른 요청이 같은 위치에 결함을 또 만들지 못하게 트랜잭션 잠금을 건다.
- 알고 있는 한계: 거리 기준 안에 있는 서로 다른 결함 둘은 하나로 묶인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import psycopg
import psycopg.types.json
from daboyeo_contracts import ObservationIn

from server import rules

_POINT = "ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography"
_LOCK_KEY = 7301


@dataclass
class Ingested:
    observation_id: UUID
    defect_id: UUID
    is_new_defect: bool
    duplicate: bool  # 같은 client_observation_id 가 이미 있었다


def ingest(conn: psycopg.Connection, obs: ObservationIn, snapshot_path: str | None,
           snapshot_bytes: int | None) -> Ingested:
    """관측을 저장하고 결함에 묶는다. 호출한 쪽이 커밋한다."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))

    old = conn.execute(
        "SELECT id, defect_id FROM observations WHERE client_observation_id = %s", (obs.client_observation_id,)
    ).fetchone()
    if old is not None:
        return Ingested(old["id"], old["defect_id"], False, True)

    t = obs.defect_type.value
    near = conn.execute(
        f"""
        SELECT id FROM defects
        WHERE defect_type = %s AND ST_DWithin(location, {_POINT}, %s)
        ORDER BY ST_Distance(location, {_POINT}) LIMIT 1
        """,
        (t, obs.lon, obs.lat, rules.DEDUP_DISTANCE_M[t], obs.lon, obs.lat),
    ).fetchone()

    is_new = near is None
    if is_new:
        defect_id = conn.execute(
            f"""
            INSERT INTO defects (defect_type, location, risk_level, confidence, is_suspected,
                                 reward_eligible_road, first_seen_at, last_seen_at)
            VALUES (%s, {_POINT}, %s, %s, %s, %s, %s, %s) RETURNING id
            """,
            (t, obs.lon, obs.lat, rules.risk_level(t), obs.confidence, obs.is_suspected,
             rules.is_reward_eligible(obs.lat, obs.lon), obs.observed_at, obs.observed_at),
        ).fetchone()["id"]
    else:
        defect_id = near["id"]

    bbox = psycopg.types.json.Jsonb(list(obs.bbox)) if obs.bbox is not None else None
    obs_id = conn.execute(
        f"""
        INSERT INTO observations
          (client_observation_id, trip_id, vehicle_id, defect_type, confidence, bbox, size_metric, is_suspected,
           location, t_s, observed_at, quality_score, snapshot_path, snapshot_bytes, defect_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, {_POINT}, %s, %s, %s, %s, %s, %s) RETURNING id
        """,
        (obs.client_observation_id, obs.trip_id, obs.vehicle_id, t, obs.confidence, bbox, obs.size_metric,
         obs.is_suspected, obs.lon, obs.lat, obs.t_s, obs.observed_at, obs.quality_score, snapshot_path,
         snapshot_bytes, defect_id),
    ).fetchone()["id"]

    # 관측 수·차량 수·시각·대표 위치(관측 평균)·대표 신뢰도(최고)·파손 의심(하나라도)을 관측들로 다시 계산한다
    conn.execute(
        """
        UPDATE defects d SET
            observation_count = s.n,
            vehicle_count     = s.vehicles,
            first_seen_at     = s.first_seen,
            last_seen_at      = s.last_seen,
            confidence        = s.max_conf,
            is_suspected      = s.suspected,
            location          = ST_SetSRID(ST_MakePoint(s.lon, s.lat), 4326)::geography
        FROM (
            SELECT count(*) AS n, count(DISTINCT vehicle_id) AS vehicles,
                   min(observed_at) AS first_seen, max(observed_at) AS last_seen,
                   max(confidence) AS max_conf, bool_or(is_suspected) AS suspected,
                   avg(ST_Y(location::geometry)) AS lat, avg(ST_X(location::geometry)) AS lon
            FROM observations WHERE defect_id = %s
        ) s
        WHERE d.id = %s
        """,
        (defect_id, defect_id),
    )
    return Ingested(obs_id, defect_id, is_new, False)
