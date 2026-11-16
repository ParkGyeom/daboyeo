"""FastAPI 서버. 실행: uv run python -m uvicorn server.main:app --port 8000

스키마는 server/sql/001_init.sql, 블랙박스가 보내는 관측 형식은 contracts 의 ObservationIn (PRD 9장).
흐름: 블랙박스가 주행을 연다(POST /api/v1/trips) → 관측을 보낸다(POST /api/v1/observations)
     → 주행을 닫는다(POST /api/v1/trips/{id}/end) → 시민의 확인 목록(findings)이 생긴다 → 신고·넘기기.
서버는 torch, ultralytics 를 불러오지 않는다.

DB 를 쓰는 엔드포인트는 async 로 두지 않는다. DB 작업이 동기라 async 면 처리하는 동안 서버 전체가 멈춘다.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

import psycopg
import psycopg.types.json
from daboyeo_contracts import DefectType, ObservationIn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from server import config, db, dedup, findings, geocode, notify, report, rules


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(title="다보여 서버", lifespan=lifespan)
config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=config.SNAPSHOT_DIR), name="snapshots")
config.CLIP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory=config.CLIP_DIR), name="clips")

API = "/api/v1"


@app.get("/health")
def health():
    with db.connect() as conn:
        conn.execute("SELECT 1")
    return {"ok": True}


# ---------- 공통 조회 ----------

def _user(conn, user_id: UUID) -> dict:
    u = conn.execute("SELECT id, name FROM users WHERE id = %s", (user_id,)).fetchone()
    if u is None:
        raise HTTPException(404, "없는 시민")
    return u


def _defect(conn, defect_id: UUID) -> dict:
    d = conn.execute(f"SELECT {db.DEFECT_COLS} FROM defects d WHERE d.id = %s", (defect_id,)).fetchone()
    if d is None:
        raise HTTPException(404, "없는 결함")
    return d


def _view(d: dict) -> dict:
    """화면용으로 관할을 붙인다 (스키마에 없는 값이라 위치로 계산)."""
    return {**d, "jurisdiction": rules.jurisdiction(d["lat"], d["lon"])}


@app.get(f"{API}/users")
def list_users():
    """시민과 그 차량. 시연용 시민은 3명 (차량 bb-01~03)."""
    with db.connect() as conn:
        return conn.execute(
            """SELECT u.id, u.name, v.id AS vehicle_id, v.label AS vehicle_label
               FROM users u LEFT JOIN vehicles v ON v.owner_user_id = u.id ORDER BY v.label NULLS LAST, u.name"""
        ).fetchall()


# ---------- 주행 (블랙박스) ----------

class TripStart(BaseModel):
    vehicle_id: UUID | None = None
    vehicle_label: str | None = None  # 둘 중 하나. 가상 블랙박스는 --device-id(bb-01) 를 그대로 보낸다
    started_at: datetime | None = None


@app.post(f"{API}/trips", status_code=201)
def start_trip(body: TripStart):
    if (body.vehicle_id is None) == (body.vehicle_label is None):
        raise HTTPException(422, "vehicle_id 와 vehicle_label 중 하나만 준다")
    if body.started_at is not None and body.started_at.tzinfo is None:
        raise HTTPException(422, "started_at 에 시간대가 없다")
    with db.connect() as conn:
        v = conn.execute(
            "SELECT id, label FROM vehicles WHERE id = %s OR label = %s", (body.vehicle_id, body.vehicle_label)
        ).fetchone()
        if v is None:
            raise HTTPException(404, "없는 차량")
        t = conn.execute(
            "INSERT INTO trips (vehicle_id, started_at) VALUES (%s, coalesce(%s, now())) RETURNING id, started_at",
            (v["id"], body.started_at),
        ).fetchone()
    return {"trip_id": t["id"], "vehicle_id": v["id"], "vehicle_label": v["label"], "started_at": t["started_at"]}


@app.post(f"{API}/trips/{{trip_id}}/end")
def end_trip(trip_id: UUID):
    """주행을 닫고 그 주행의 확인 목록을 만든다. 주행 중에는 시민에게 알리지 않고 끝난 뒤에만 알린다.
    이미 닫힌 주행이면 목록만 다시 훑는다."""
    with db.connect() as conn:
        t = conn.execute(
            "UPDATE trips SET ended_at = coalesce(ended_at, now()) WHERE id = %s RETURNING id, ended_at", (trip_id,)
        ).fetchone()
        if t is None:
            raise HTTPException(404, "없는 주행")
        findings.refresh_trip(conn, trip_id)
        n = conn.execute("SELECT count(*) AS n FROM findings WHERE trip_id = %s", (trip_id,)).fetchone()["n"]
        # 확인할 결함이 있으면 그 차량 주인에게 모의 문자(링크 포함)를 만든다. 실제 발송은 하지 않는다
        notified = notify.create_for_trip(conn, trip_id) is not None
    return {"trip_id": trip_id, "ended_at": t["ended_at"], "finding_count": n, "notified": notified}


@app.get(f"{API}/trips/{{trip_id}}/findings")
def trip_findings(trip_id: UUID):
    with db.connect() as conn:
        t = conn.execute(
            "SELECT v.owner_user_id FROM trips t JOIN vehicles v ON v.id = t.vehicle_id WHERE t.id = %s", (trip_id,)
        ).fetchone()
        if t is None:
            raise HTTPException(404, "없는 주행")
        findings.refresh_trip(conn, trip_id)
        return [_view(d) for d in findings.pending(conn, t["owner_user_id"], trip_id)]


@app.post(f"{API}/observations", status_code=201)
def post_observation(bg: BackgroundTasks, meta: str = Form(...), snapshot: UploadFile | None = File(None),
                     clip: UploadFile | None = File(None)):
    """관측 한 건 (multipart: meta = ObservationIn JSON, snapshot = 결함 주변 JPEG, clip = 결함 구간 mp4 (선택)).
    같은 client_observation_id 를 다시 보내면 저장하지 않고 처음 결과를 duplicate=true 로 돌려준다."""
    try:
        obs = ObservationIn.model_validate_json(meta)
    except ValidationError as e:
        raise HTTPException(422, detail=e.errors(include_url=False, include_context=False))
    if obs.observed_at.tzinfo is None:
        raise HTTPException(422, "observed_at 에 시간대가 없다")

    photo = snapshot.file.read() if snapshot is not None else None
    if photo is not None:
        if len(photo) > config.MAX_SNAPSHOT_BYTES:
            raise HTTPException(413, "사진이 너무 크다")
        if not photo.startswith(b"\xff\xd8"):
            raise HTTPException(422, "snapshot 은 JPEG 파일이어야 한다")
    video = clip.file.read() if clip is not None else None
    if video is not None:
        if len(video) > config.MAX_CLIP_BYTES:
            raise HTTPException(413, "클립이 너무 크다")
        if video[4:8] != b"ftyp":
            raise HTTPException(422, "clip 은 mp4 파일이어야 한다")

    with db.connect() as conn:
        trip = conn.execute("SELECT vehicle_id FROM trips WHERE id = %s", (obs.trip_id,)).fetchone()
        if trip is None:
            raise HTTPException(404, "없는 주행 (먼저 POST /api/v1/trips)")
        if trip["vehicle_id"] != obs.vehicle_id:
            raise HTTPException(422, "vehicle_id 가 그 주행의 차량이 아니다")
        name = path = clip_name = clip_path = None
        if photo is not None:
            name = f"{uuid.uuid4().hex}.jpg"
            path = config.SNAPSHOT_DIR / name
            path.write_bytes(photo)
        if video is not None:
            clip_name = f"{uuid.uuid4().hex}.mp4"
            clip_path = config.CLIP_DIR / clip_name
            clip_path.write_bytes(video)
        files = [f for f in (path, clip_path) if f]
        try:
            r = dedup.ingest(conn, obs, name, len(photo) if photo is not None else None,
                             clip_name, len(video) if video is not None else None)
        except Exception:
            for f in files:
                f.unlink(missing_ok=True)  # 저장에 실패하면 사진·클립도 남기지 않는다
            raise
        if r.duplicate:
            for f in files:
                f.unlink(missing_ok=True)
    if r.is_new_defect:
        bg.add_task(geocode.fill_missing)  # 새 결함의 주소는 응답 뒤에 받는다. 블랙박스를 기다리게 하지 않는다
    return {"observation_id": r.observation_id, "defect_id": r.defect_id,
            "is_new_defect": r.is_new_defect, "duplicate": r.duplicate}


# ---------- 시민 ----------

class UserBody(BaseModel):
    user_id: UUID


@app.get(f"{API}/users/{{user_id}}/findings")
def user_findings(user_id: UUID):
    """그 시민이 아직 답하지 않은 확인 목록 (모든 끝난 주행)."""
    with db.connect() as conn:
        _user(conn, user_id)
        findings.refresh_user(conn, user_id)
        return [_view(d) for d in findings.pending(conn, user_id)]


@app.get(f"{API}/users/{{user_id}}/notices")
def user_notices(user_id: UUID):
    """'주행이 끝났어요' 알림: 끝난 주행 중 아직 답하지 않은 항목이 남은 것. 최근 것부터."""
    with db.connect() as conn:
        _user(conn, user_id)
        findings.refresh_user(conn, user_id)
        return findings.notices(conn, user_id)


def _do_report(conn, u: dict, defect_id: UUID) -> dict:
    """신고 한 건을 처리한다 (시민 선택 화면의 API 와 문자 링크의 API 가 같이 쓴다)."""
    _defect(conn, defect_id)
    seen = conn.execute(
        """SELECT 1 FROM observations o JOIN vehicles v ON v.id = o.vehicle_id
           WHERE o.defect_id = %s AND v.owner_user_id = %s LIMIT 1""",
        (defect_id, u["id"]),
    ).fetchone()
    if seen is None:
        raise HTTPException(403, "이 시민의 차량이 관측한 결함이 아니다")
    geocode.fill_one(conn, defect_id)  # 아직 주소가 없으면 신고서에 넣으려고 한 번 더 시도한다
    # 상태를 바꾸는 UPDATE 가 한 번에 한 요청만 통과시킨다
    won = conn.execute(
        "UPDATE defects SET status = 'reported' WHERE id = %s AND status = 'unreported' RETURNING id", (defect_id,)
    ).fetchone()
    if won is None:
        raise HTTPException(409, "이미 신고됨")
    d = _defect(conn, defect_id)
    now = datetime.now(rules.KST)
    payload = report.build_payload(d, u["name"], now)
    rep = conn.execute(
        """INSERT INTO reports (defect_id, user_id, quarter, payload, submission_status, reward_valid, created_at)
           VALUES (%s, %s, %s, %s, 'mock_submitted', %s, %s)
           ON CONFLICT (defect_id) DO NOTHING RETURNING id""",
        (defect_id, u["id"], rules.quarter_label(now), psycopg.types.json.Jsonb(payload),
         d["reward_eligible_road"], now),
    ).fetchone()
    if rep is None:
        raise HTTPException(409, "이미 신고됨")
    # 신고한 시민의 항목은 신고함, 같은 결함을 기다리던 다른 시민의 항목은 이미 신고됨
    conn.execute(
        """UPDATE findings SET response = CASE WHEN user_id = %s THEN 'reported' ELSE 'already_reported' END::finding_response,
                               responded_at = now()
           WHERE defect_id = %s AND response = 'pending'""",
        (u["id"], defect_id),
    )
    return {"report_id": rep["id"], "status": rules.REPORTED, "payload": payload}


def _do_skip(conn, user_id: UUID, defect_id: UUID) -> None:
    n = conn.execute(
        """UPDATE findings SET response = 'skipped', responded_at = now()
           WHERE user_id = %s AND defect_id = %s AND response = 'pending'""",
        (user_id, defect_id),
    ).rowcount
    if n == 0:
        raise HTTPException(404, "그 시민의 확인 목록에 없는 결함")


@app.post(f"{API}/defects/{{defect_id}}/report")
def report_defect(defect_id: UUID, body: UserBody):
    """신고. 결함당 신고서는 1건 (FR-5.5). 이미 신고됐으면 409."""
    with db.connect() as conn:
        return _do_report(conn, _user(conn, body.user_id), defect_id)


@app.post(f"{API}/defects/{{defect_id}}/skip")
def skip_defect(defect_id: UUID, body: UserBody):
    """넘기기. 그 시민의 목록에서만 빠지고 다시 묻지 않는다."""
    with db.connect() as conn:
        _user(conn, body.user_id)
        _defect(conn, defect_id)
        _do_skip(conn, body.user_id, defect_id)
    return {"ok": True}


# ---------- 문자 링크 (가입·로그인 없이 토큰으로) ----------

def _token(conn, token: str) -> dict:
    """링크의 토큰으로 그 시민과 주행을 찾는다. 없으면 404, 기한이 지났으면 410."""
    n = conn.execute(
        """SELECT n.id, n.user_id, n.trip_id, u.name, n.expires_at, n.expires_at <= now() AS expired
           FROM notifications n JOIN users u ON u.id = n.user_id WHERE n.token = %s""",
        (token,),
    ).fetchone()
    if n is None:
        raise HTTPException(404, "없는 링크")
    if n["expired"]:
        raise HTTPException(410, "링크의 유효 시간이 지났다")
    return n


def _link_finding(conn, n: dict, defect_id: UUID) -> str:
    """이 링크 주인의 이 주행 확인 항목의 응답 상태. 목록에 없는 결함이면 404."""
    f = conn.execute(
        "SELECT response FROM findings WHERE user_id = %s AND trip_id = %s AND defect_id = %s",
        (n["user_id"], n["trip_id"], defect_id),
    ).fetchone()
    if f is None:
        raise HTTPException(404, "이 링크의 확인 목록에 없는 결함")
    return f["response"]


@app.get(f"{API}/m/{{token}}/findings")
def link_findings(token: str):
    """문자 링크를 열었을 때 보여 줄 그 주행의 확인 목록 (아직 답하지 않은 것)."""
    with db.connect() as conn:
        n = _token(conn, token)
        conn.execute("UPDATE notifications SET opened_at = coalesce(opened_at, now()) WHERE id = %s", (n["id"],))
        items = [_view(d) for d in findings.pending(conn, n["user_id"], n["trip_id"])]
    return {"user_name": n["name"], "expires_at": n["expires_at"], "findings": items}


@app.post(f"{API}/m/{{token}}/defects/{{defect_id}}/report")
def link_report(token: str, defect_id: UUID):
    """링크에서 [신고] 한 번. 시민은 토큰으로 정한다. 이미 신고됐으면 409."""
    with db.connect() as conn:
        n = _token(conn, token)
        resp = _link_finding(conn, n, defect_id)
        if resp in ("reported", "already_reported"):
            raise HTTPException(409, "이미 신고됨")
        if resp != "pending":
            raise HTTPException(404, "이 링크의 확인 목록에 없는 결함")
        return _do_report(conn, {"id": n["user_id"], "name": n["name"]}, defect_id)


@app.post(f"{API}/m/{{token}}/defects/{{defect_id}}/skip")
def link_skip(token: str, defect_id: UUID):
    with db.connect() as conn:
        n = _token(conn, token)
        _link_finding(conn, n, defect_id)
        _do_skip(conn, n["user_id"], defect_id)
    return {"ok": True}


@app.get(f"{API}/m/{{token}}/reports/{{report_id}}")
def link_get_report(token: str, report_id: UUID):
    """링크 주인이 낸 신고서만 볼 수 있다."""
    with db.connect() as conn:
        n = _token(conn, token)
        r = conn.execute(
            """SELECT id, defect_id, quarter, payload, submission_status, reward_valid, created_at
               FROM reports WHERE id = %s AND user_id = %s""",
            (report_id, n["user_id"]),
        ).fetchone()
    if r is None:
        raise HTTPException(404, "없는 신고서")
    return r


@app.get(f"{API}/notifications")
def notifications():
    """모의 문자 목록 (시연용 메시지함 화면). 받는 번호는 가운데를 가린다. 실제 서비스라면 공개하면 안 된다 (링크 토큰이 들어 있다)."""
    with db.connect() as conn:
        return notify.recent(conn)


@app.get(f"{API}/reports/{{report_id}}")
def get_report(report_id: UUID):
    with db.connect() as conn:
        r = conn.execute(
            """SELECT id, defect_id, user_id, quarter, payload, submission_status, reward_valid, created_at
               FROM reports WHERE id = %s""",
            (report_id,),
        ).fetchone()
    if r is None:
        raise HTTPException(404, "없는 신고서")
    return r


@app.get(f"{API}/users/{{user_id}}/reports")
def user_reports(user_id: UUID):
    """그 시민의 신고서. 최근 것부터."""
    with db.connect() as conn:
        _user(conn, user_id)
        return conn.execute(
            "SELECT id, defect_id, quarter, payload, created_at FROM reports WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()


@app.get(f"{API}/users/{{user_id}}/reward")
def reward(user_id: UUID):
    """이번 분기(한국 시간)에 그 시민이 낸 포상 대상 신고 수와 5건까지 남은 수."""
    now = datetime.now(rules.KST)
    with db.connect() as conn:
        _user(conn, user_id)
        n = conn.execute(
            "SELECT count(*) AS n FROM reports WHERE user_id = %s AND reward_valid AND quarter = %s",
            (user_id, rules.quarter_label(now)),
        ).fetchone()["n"]
    return {**rules.reward_progress(n), "quarter": rules.quarter_label(now),
            "since": rules.quarter_start(now).isoformat()}


# ---------- 관리자 ----------

class StatusBody(BaseModel):
    status: str


@app.get(f"{API}/defects")
def list_defects(bg: BackgroundTasks, type: str | None = None, status: str | None = None):
    if type is not None and type not in {t.value for t in DefectType}:
        raise HTTPException(422, "type 이 올바르지 않다")
    if status is not None and status not in rules.STATUSES:
        raise HTTPException(422, "status 가 올바르지 않다")
    with db.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {db.DEFECT_COLS} FROM defects d
            WHERE (%s::text IS NULL OR d.defect_type::text = %s) AND (%s::text IS NULL OR d.status::text = %s)
            ORDER BY d.first_seen_at, d.id
            """,
            (type, type, status, status),
        ).fetchall()
    if any(d["address"] is None for d in rows):
        bg.add_task(geocode.fill_missing)  # 주소를 못 받은 결함은 관리자 지도가 새로 고칠 때 다시 시도한다
    return [_view(d) for d in rows]


@app.patch(f"{API}/defects/{{defect_id}}/status")
def change_status(defect_id: UUID, body: StatusBody):
    """관리자가 상태를 바꾼다. 허용하는 전환은 rules.can_transition 만 (신고됨 → 보수 중)."""
    if body.status not in rules.STATUSES:
        raise HTTPException(422, "status 가 올바르지 않다")
    with db.connect() as conn:
        old = _defect(conn, defect_id)["status"]
        if not rules.can_transition(old, body.status):
            raise HTTPException(409, f"{old} 에서 {body.status} 로 바꿀 수 없다")
        done = conn.execute(
            "UPDATE defects SET status = %s WHERE id = %s AND status = %s RETURNING id", (body.status, defect_id, old)
        ).fetchone()
        if done is None:
            raise HTTPException(409, "그 사이 상태가 바뀌었다. 다시 불러온다")
    return {"id": defect_id, "status": body.status}


@app.get(f"{API}/stats")
def stats():
    """차량별 받은 관측 수와 용량 (통신량 실험용)."""
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT v.label AS vehicle_label, count(*) AS observations,
                      coalesce(sum(coalesce(o.snapshot_bytes, 0) + coalesce(o.clip_bytes, 0)), 0)::bigint AS bytes
               FROM observations o JOIN vehicles v ON v.id = o.vehicle_id GROUP BY v.label ORDER BY v.label"""
        ).fetchall()
    return {"vehicles": rows, "total_observations": sum(r["observations"] for r in rows),
            "total_bytes": sum(r["bytes"] for r in rows)}
