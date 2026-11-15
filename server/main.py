"""FastAPI 서버. 실행: .venv/bin/python -m uvicorn server.main:app --port 8000

탐지 결과 수신(B1), 중복 정리·신고 상태 API(B3).
서버는 torch, ultralytics 를 불러오지 않는다.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import psycopg
import psycopg.types.json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from server import config, db, dedup, report, rules
from server.schema import DetectionEvent


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(title="다보여 서버", lifespan=lifespan)
config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=config.SNAPSHOT_DIR), name="snapshots")


@app.get("/health")
def health():
    with db.connect() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}


@app.post("/api/detections", status_code=201)
def post_detection(event: str = Form(...), snapshot: UploadFile = File(...)):
    """탐지 결과 한 건과 사진을 받아 저장한다. event 는 server/schema.py 의 JSON.

    async 로 두지 않는다. DB 작업이 동기라서 async 면 처리하는 동안 서버 전체가 멈춘다.
    일반 함수면 FastAPI 가 별도 스레드에서 돌린다.
    """
    try:
        ev = DetectionEvent.model_validate_json(event)
    except ValidationError as e:
        raise HTTPException(422, detail=e.errors(include_url=False, include_context=False))

    photo = snapshot.file.read()
    if len(photo) > config.MAX_SNAPSHOT_BYTES:
        raise HTTPException(413, "사진이 너무 크다")
    if not photo.startswith(b"\xff\xd8"):
        raise HTTPException(422, "snapshot 은 JPEG 파일이어야 한다")

    name = f"{uuid.uuid4().hex}.jpg"
    path = config.SNAPSHOT_DIR / name
    path.write_bytes(photo)
    received = len(event.encode("utf-8")) + len(photo)

    try:
        with db.connect() as conn:
            obs_id, defect_id, is_new = dedup.ingest(conn, ev, name, received)
    except Exception:
        path.unlink(missing_ok=True)  # 저장에 실패하면 사진도 남기지 않는다
        raise
    return {"observation_id": obs_id, "defect_id": defect_id, "is_new_defect": is_new}


class UserBody(BaseModel):
    user_id: int


class StatusBody(BaseModel):
    status: str


def _user(conn, user_id: int) -> dict:
    u = conn.execute("SELECT id, name, device_id FROM users WHERE id = %s", (user_id,)).fetchone()
    if u is None:
        raise HTTPException(404, "없는 시민")
    return u


def _defect(conn, defect_id: int) -> dict:
    d = conn.execute(f"SELECT {db.DEFECT_COLS} FROM defects d WHERE d.id = %s", (defect_id,)).fetchone()
    if d is None:
        raise HTTPException(404, "없는 결함")
    return d


def _visible_defects(conn, u: dict, since: datetime | None = None) -> list[dict]:
    """그 시민에게 보여줄 결함 (rules.is_visible_to_citizen 조건).
    since 를 주면 그 시각 이후에 그 시민의 기기가 관측한 결함만 (이번 주행에서 발견한 것)."""
    rows = conn.execute(
        f"""
        SELECT {db.DEFECT_COLS} FROM defects d
        WHERE d.status = %s
          AND EXISTS (SELECT 1 FROM observations o WHERE o.defect_id = d.id AND o.device_id = %s
                      AND (%s::timestamptz IS NULL OR o.detected_at >= %s))
          AND NOT EXISTS (SELECT 1 FROM skips k WHERE k.user_id = %s AND k.defect_id = d.id)
        ORDER BY d.max_confidence DESC, d.id
        """,
        (rules.UNREPORTED, u["device_id"], since, since, u["id"]),
    ).fetchall()
    return [
        d for d in rows
        if rules.is_visible_to_citizen(d["status"], d["max_confidence"], d["device_count"], True, False)
    ]


@app.get("/api/users/{user_id}/pending")
def pending(user_id: int):
    """그 시민에게 보여줄 결함 목록 (rules.is_visible_to_citizen 조건)."""
    with db.connect() as conn:
        return _visible_defects(conn, _user(conn, user_id))


class DriveEndBody(BaseModel):
    device_id: str
    started_at: datetime | None = None  # 주어지면 이 시각 이후 관측만 "이번 주행"으로 센다


@app.post("/api/drives", status_code=201)
def end_drive(body: DriveEndBody):
    """가상 블랙박스가 주행(구간 재생)을 끝냈다고 알린다.

    주행 중에는 운전자에게 아무것도 알리지 않고, 끝난 뒤에만 시민 화면에 알림을 띄운다.
    알림에 적을 결함 수는 지금 이 시각 그 시민에게 보여줄 결함 중 이번 주행에서 관측한 것.
    """
    if body.started_at is not None and body.started_at.tzinfo is None:
        raise HTTPException(422, "started_at 에 시간대가 없다")
    with db.connect() as conn:
        u = conn.execute("SELECT id, name, device_id FROM users WHERE device_id = %s", (body.device_id,)).fetchone()
        count = len(_visible_defects(conn, u, body.started_at)) if u else 0
        row = conn.execute(
            "INSERT INTO drives (device_id, started_at, defect_count) VALUES (%s, %s, %s) RETURNING id",
            (body.device_id, body.started_at, count),
        ).fetchone()
    return {"drive_id": row["id"], "defect_count": count}


@app.get("/api/users/{user_id}/notices")
def notices(user_id: int):
    """그 시민이 아직 확인하지 않은 주행 종료 알림. 최근 것부터."""
    with db.connect() as conn:
        u = _user(conn, user_id)
        return conn.execute(
            """
            SELECT id, device_id, started_at, ended_at, defect_count FROM drives
            WHERE device_id = %s AND seen_at IS NULL ORDER BY ended_at DESC, id DESC
            """,
            (u["device_id"],),
        ).fetchall()


@app.post("/api/users/{user_id}/notices/seen")
def notices_seen(user_id: int):
    """그 시민의 알림을 모두 확인한 것으로 표시한다."""
    with db.connect() as conn:
        u = _user(conn, user_id)
        n = conn.execute(
            "UPDATE drives SET seen_at = now() WHERE device_id = %s AND seen_at IS NULL", (u["device_id"],)
        ).rowcount
    return {"seen": n}


@app.post("/api/defects/{defect_id}/report")
def report_defect(defect_id: int, body: UserBody):
    """신고. 신고서를 만들고 상태를 신고됨으로 바꾼다. 이미 신고된 결함이면 409."""
    with db.connect() as conn:
        u = _user(conn, body.user_id)
        _defect(conn, defect_id)
        seen = conn.execute(
            "SELECT 1 FROM observations WHERE defect_id = %s AND device_id = %s LIMIT 1", (defect_id, u["device_id"])
        ).fetchone()
        if seen is None:
            raise HTTPException(403, "이 시민의 기기가 관측한 결함이 아니다")
        # 상태를 바꾸는 UPDATE 가 한 번에 한 요청만 통과시킨다
        won = conn.execute(
            "UPDATE defects SET status = %s WHERE id = %s AND status = %s RETURNING id",
            (rules.REPORTED, defect_id, rules.UNREPORTED),
        ).fetchone()
        if won is None:
            raise HTTPException(409, "이미 신고됨")
        now = datetime.now(rules.KST)
        body_json = report.build_report_body(_defect(conn, defect_id), u["name"], now)
        rep = conn.execute(
            "INSERT INTO reports (defect_id, user_id, created_at, body) VALUES (%s, %s, %s, %s::jsonb) RETURNING id",
            (defect_id, u["id"], now, psycopg.types.json.Jsonb(body_json)),
        ).fetchone()
    return {"report_id": rep["id"], "status": rules.REPORTED, "body": body_json}


@app.post("/api/defects/{defect_id}/skip")
def skip_defect(defect_id: int, body: UserBody):
    with db.connect() as conn:
        _user(conn, body.user_id)
        _defect(conn, defect_id)
        conn.execute(
            "INSERT INTO skips (user_id, defect_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (body.user_id, defect_id)
        )
    return {"ok": True}


@app.get("/api/reports/{report_id}")
def get_report(report_id: int):
    with db.connect() as conn:
        r = conn.execute(
            "SELECT r.id, r.defect_id, r.user_id, r.created_at, r.body FROM reports r WHERE r.id = %s", (report_id,)
        ).fetchone()
    if r is None:
        raise HTTPException(404, "없는 신고서")
    return r


@app.get("/api/users/{user_id}/reward")
def reward(user_id: int):
    """이번 분기(한국 시간)에 그 시민이 낸 포상 대상 신고 수와 5건까지 남은 수."""
    since = rules.quarter_start(datetime.now(rules.KST))
    with db.connect() as conn:
        _user(conn, user_id)
        n = conn.execute(
            """
            SELECT count(*) AS n FROM reports r JOIN defects d ON d.id = r.defect_id
            WHERE r.user_id = %s AND d.reward_eligible AND r.created_at >= %s
            """,
            (user_id, since),
        ).fetchone()["n"]
    return {**rules.reward_progress(n), "since": since.isoformat()}


@app.get("/api/defects")
def list_defects(type: str | None = None, status: str | None = None):
    if type is not None and type not in ("pothole", "crack", "lane_wear"):
        raise HTTPException(422, "type 이 올바르지 않다")
    if status is not None and status not in rules.STATUSES:
        raise HTTPException(422, "status 가 올바르지 않다")
    with db.connect() as conn:
        return conn.execute(
            f"""
            SELECT {db.DEFECT_COLS} FROM defects d
            WHERE (%s::text IS NULL OR d.type = %s) AND (%s::text IS NULL OR d.status = %s)
            ORDER BY d.id
            """,
            (type, type, status, status),
        ).fetchall()


@app.patch("/api/defects/{defect_id}/status")
def change_status(defect_id: int, body: StatusBody):
    """관리자가 상태를 바꾼다. 허용하는 전환은 rules.can_transition 만."""
    if body.status not in rules.STATUSES:
        raise HTTPException(422, "status 가 올바르지 않다")
    with db.connect() as conn:
        old = _defect(conn, defect_id)["status"]
        if not rules.can_transition(old, body.status):
            raise HTTPException(409, f"{old} 에서 {body.status} 로 바꿀 수 없다")
        # 읽은 뒤 다른 요청이 상태를 바꿨으면 고치지 않는다
        done = conn.execute(
            "UPDATE defects SET status = %s WHERE id = %s AND status = %s RETURNING id", (body.status, defect_id, old)
        ).fetchone()
        if done is None:
            raise HTTPException(409, "그 사이 상태가 바뀌었다. 다시 불러온다")
    return {"id": defect_id, "status": body.status}


@app.get("/api/stats")
def stats():
    """기기별 받은 건수와 용량 (통신량 실험용)."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT device_id, count(*) AS observations, sum(bytes)::bigint AS bytes FROM observations GROUP BY device_id ORDER BY device_id"
        ).fetchall()
    return {"devices": rows, "total_observations": sum(r["observations"] for r in rows),
            "total_bytes": sum(r["bytes"] for r in rows)}
