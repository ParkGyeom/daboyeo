"""주행 종료 문자 (모의 발송). docs/notify-link-plan.md

실제 문자·메일은 보내지 않는다. 주행이 끝나면 "보낼 문자"를 notifications 표에 저장하는 것이 발송이다.
문자에는 그 시민 전용 링크가 들어 있고, 링크의 토큰이 신원 확인을 대신한다 (가입·로그인 없음).
한 줄짜리 문자는 스팸처럼 보여서, 언제·어디서 무엇을 찾았는지와 링크를 누르면 무엇이 되는지를 함께 적는다 (compose).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import psycopg

from server import config, findings, geocode, report, rules

# 화면(ui/api.py TYPE_KO)과 같은 한국어 이름. 문자는 서버가 만들어서 여기에도 둔다
TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모",
           "guardrail_damage": "가드레일", "sign_damage": "표지판"}
MAX_PLACE_LINES = 3  # 위치 줄이 이보다 많으면 위에서부터 이만큼만 쓰고 "외 N건"


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


def _clock(t: datetime) -> str:
    t = t.astimezone(rules.KST)
    return f"{'오전' if t.hour < 12 else '오후'} {(t.hour % 12) or 12}:{t.minute:02d}"


def _date(t: datetime) -> str:
    t = t.astimezone(rules.KST)
    return f"{t.month}월 {t.day}일"


def drive_span(started_at: datetime, ended_at: datetime) -> str:
    """'11월 18일 오후 12:54~12:58' (한국 시각). 오전·오후가 바뀌면 끝에도 오전·오후를, 날짜가 넘어가면 날짜도 붙인다."""
    a, b = _clock(started_at), _clock(ended_at)
    if _date(started_at) == _date(ended_at) and a == b:
        return f"{_date(started_at)} {a}"  # 1분 안에 끝난 주행 (빠른 재생 시연)
    if _date(started_at) != _date(ended_at):
        end = f"{_date(ended_at)} {b}"
    else:
        end = b.split(" ", 1)[1] if a[:2] == b[:2] else b
    return f"{_date(started_at)} {_clock(started_at)}~{end}"


def place_text(address: str | None) -> str:
    """주소 → '서울 강남구 봉은사로 102 부근'. 아직 못 받았으면(None) '위치 확인 중', 주소 없는 곳("")은 '주소 없는 곳'."""
    if address is None:
        return "위치 확인 중"
    return f"{address} 부근" if address else "주소 없는 곳"


def place_lines(items: list[dict], limit: int = MAX_PLACE_LINES) -> list[str]:
    """결함 목록 → '· 균열 3건 — 서울 강남구 봉은사로 102 부근' 줄들. 같은 (유형, 주소)끼리 묶고 많은 것부터.
    줄이 limit 보다 많으면 limit 줄 + '· 외 N건' (N 은 남은 결함 수)."""
    groups: dict[tuple[str, str], int] = {}
    for d in items:  # dict 는 넣은 순서를 지켜서, 수가 같으면 목록(신뢰도 높은 순)에서 먼저 나온 것이 앞
        key = (TYPE_KO.get(d["defect_type"], d["defect_type"]), place_text(d.get("address")))
        groups[key] = groups.get(key, 0) + 1
    ranked = sorted(groups.items(), key=lambda kv: -kv[1])
    lines = [f"· {t} {n}건 — {where}" for (t, where), n in ranked[:limit]]
    if rest := sum(n for _, n in ranked[limit:]):
        lines.append(f"· 외 {rest}건")
    return lines


def count_line(report_count: int) -> str:
    """신고 누적 한 줄 (사람 기준). 포상 기준·남은 건수는 넣지 않는다 (PRD F8)."""
    return f"■ 지금까지 내 신고: {report_count}건" if report_count else "■ 지금까지 내 신고: 아직 없음"


def compose(user_name: str, started_at: datetime, ended_at: datetime, items: list[dict], report_count: int,
            url: str, ttl_hours: int) -> str:
    """문자 본문. items 는 그 주행의 확인 목록 결함(defect_type, address).

    실제 안내 문자처럼 제목 / 인사 / ■ 항목 / 안내 / ▶ 링크 / 끝 안내를 빈 줄로 나눈다. 한 줄이 길면 휴대폰이 알아서 접는다.
    """
    return "\n".join([
        "[다보여] 도로 결함 확인 요청",
        "",
        f"{user_name}님, 주행 중 블랙박스가 도로 결함 {len(items)}건을 찾았어요.",
        "",
        f"■ 주행: {drive_span(started_at, ended_at)}",
        f"■ 찾은 결함 {len(items)}건",
        *place_lines(items),
        count_line(report_count),
        "",
        "사진을 보고 신고할지 직접 정해 주세요.",
        "신고하면 본인 명의로 서울시에 접수돼요.",
        "(시연에서는 모의 접수)",
        "",
        "▶ 확인·신고하기",
        url,
        "",
        f"링크는 {ttl_hours}시간 동안 열려요.",
        "신고하지 않으려면 이 문자를 무시하세요.",
    ])


def create_for_trip(conn: psycopg.Connection, trip_id: UUID) -> UUID | None:
    """끝난 주행의 차량 주인에게 모의 문자를 만든다. 문자를 만들었으면 그 번호, 아니면 None.

    만들지 않는 경우: 그 시민의 이 주행 확인 목록이 비어 있다 / 연락처가 없다 / 이 (주행, 시민)에게 이미 보냈다.
    호출하기 전에 findings.refresh_trip 으로 확인 목록을 채워 둔다.
    """
    row = conn.execute(
        """SELECT v.owner_user_id AS user_id, u.name, u.phone, t.started_at, t.ended_at
           FROM trips t JOIN vehicles v ON v.id = t.vehicle_id JOIN users u ON u.id = v.owner_user_id
           WHERE t.id = %s AND t.ended_at IS NOT NULL""",
        (trip_id,),
    ).fetchone()
    if row is None or not row["phone"]:
        return None
    pending = findings.pending(conn, row["user_id"], trip_id)
    if not pending:
        return None
    if conn.execute("SELECT 1 FROM notifications WHERE trip_id = %s AND user_id = %s",
                    (trip_id, row["user_id"])).fetchone():
        return None  # 이미 보냈다. 아래 주소 받기(인터넷)를 하지 않으려고 먼저 본다
    # 문자에 주소를 넣으려고 아직 못 받은 주소를 한 번 더 받는다 (보통은 관측이 들어올 때 이미 받아 둠)
    if any(d["address"] is None for d in pending) and geocode.keys() is not None:
        for d in pending:
            if d["address"] is None:
                geocode.fill_one(conn, d["id"])
        pending = findings.pending(conn, row["user_id"], trip_id)
    token = secrets.token_urlsafe(24)
    body = compose(row["name"], row["started_at"], row["ended_at"], pending,
                   report.count_for(conn, row["user_id"]), link_url(token), config.LINK_TTL_HOURS)
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
