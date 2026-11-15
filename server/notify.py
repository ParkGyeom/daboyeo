"""주행 종료 문자 (모의 발송). docs/notify-link-plan.md

실제 문자·메일은 보내지 않는다. 주행이 끝나면 "보낼 문자"를 notifications 표에 저장하는 것이 발송이다.
문자에는 그 시민 전용 링크가 들어 있고, 링크의 토큰이 신원 확인을 대신한다 (가입·로그인 없음).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import psycopg

from server import config, findings


def link_url(token: str) -> str:
    return f"{config.PUBLIC_BASE_URL}/?t={token}"


def mask_phone(phone: str | None) -> str | None:
    """010-1234-5678 → 010-****-5678 (가운데 번호를 가린다)."""
    if not phone:
        return phone
    parts = phone.split("-")
    if len(parts) == 3:
        return f"{parts[0]}-{'*' * len(parts[1])}-{parts[2]}"
    return phone[:3] + "*" * max(0, len(phone) - 7) + phone[-4:]


def create_for_trip(conn: psycopg.Connection, trip_id: UUID) -> UUID | None:
    """끝난 주행의 차량 주인에게 모의 문자를 만든다. 문자를 만들었으면 그 번호, 아니면 None.

    만들지 않는 경우: 그 시민의 이 주행 확인 목록이 비어 있다 / 연락처가 없다 / 이 (주행, 시민)에게 이미 보냈다.
    호출하기 전에 findings.refresh_trip 으로 확인 목록을 채워 둔다.
    """
    row = conn.execute(
        """SELECT v.owner_user_id AS user_id, u.phone
           FROM trips t JOIN vehicles v ON v.id = t.vehicle_id JOIN users u ON u.id = v.owner_user_id
           WHERE t.id = %s AND t.ended_at IS NOT NULL""",
        (trip_id,),
    ).fetchone()
    if row is None or not row["phone"]:
        return None
    n = len(findings.pending(conn, row["user_id"], trip_id))
    if n == 0:
        return None
    token = secrets.token_urlsafe(24)
    body = f"[다보여] 주행 중 도로 결함 {n}건을 발견했어요. 확인하고 신고하기: {link_url(token)}"
    made = conn.execute(
        """INSERT INTO notifications (user_id, trip_id, to_phone, body, token, expires_at)
           VALUES (%s, %s, %s, %s, %s, %s)
           ON CONFLICT (trip_id, user_id) DO NOTHING RETURNING id""",
        (row["user_id"], trip_id, row["phone"], body, token,
         datetime.now(timezone.utc) + timedelta(hours=config.LINK_TTL_HOURS)),
    ).fetchone()
    return made["id"] if made else None


def recent(conn: psycopg.Connection, limit: int = 50) -> list[dict]:
    """최근 모의 문자. 받는 번호는 가운데를 가리고, 링크 주소를 함께 준다 (메시지함 화면용)."""
    rows = conn.execute(
        """SELECT n.id, u.name AS user_name, n.to_phone, n.body, n.token, n.created_at, n.expires_at, n.opened_at
           FROM notifications n JOIN users u ON u.id = n.user_id
           ORDER BY n.created_at DESC LIMIT %s""",
        (limit,),
    ).fetchall()
    return [{**r, "to_phone": mask_phone(r["to_phone"]), "url": link_url(r["token"])} for r in rows]
