"""화면이 쓰는 서버 API 호출. 화면은 DB에 직접 접근하지 않는다 (docs/DESIGN.md 1장)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import streamlit as st

SERVER = os.environ.get("SERVER_URL", "http://localhost:8000")
TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모"}
USERS = {1: "시민 A (bb-01)", 2: "시민 B (bb-02)", 3: "시민 C (bb-03)"}


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _call(method: str, path: str, **kw):
    try:
        r = httpx.request(method, SERVER + path, timeout=10, **kw)
    except httpx.HTTPError as e:
        raise ApiError(0, f"서버에 연결할 수 없다 ({SERVER}): {e}") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(r.status_code, str(detail))
    return r.json()


def pending(user_id: int) -> list[dict]:
    return _call("GET", f"/api/users/{user_id}/pending")


def clock(iso: str) -> str:
    """서버가 준 시각(ISO 8601)을 한국 시간 '오후 3:05' 꼴로."""
    t = datetime.fromisoformat(iso).astimezone(timezone(timedelta(hours=9)))
    return f"{'오전' if t.hour < 12 else '오후'} {(t.hour % 12) or 12}:{t.minute:02d}"


def notices(user_id: int) -> list[dict]:
    """아직 확인하지 않은 주행 종료 알림. 최근 것부터."""
    return _call("GET", f"/api/users/{user_id}/notices")


def notices_seen(user_id: int) -> dict:
    return _call("POST", f"/api/users/{user_id}/notices/seen")


def report(defect_id: int, user_id: int) -> dict:
    return _call("POST", f"/api/defects/{defect_id}/report", json={"user_id": user_id})


def skip(defect_id: int, user_id: int) -> dict:
    return _call("POST", f"/api/defects/{defect_id}/skip", json={"user_id": user_id})


def get_report(report_id: int) -> dict:
    return _call("GET", f"/api/reports/{report_id}")


def defects(type: str | None = None, status: str | None = None) -> list[dict]:
    params = {k: v for k, v in {"type": type, "status": status}.items() if v}
    return _call("GET", "/api/defects", params=params)


def set_status(defect_id: int, status: str) -> dict:
    return _call("PATCH", f"/api/defects/{defect_id}/status", json={"status": status})


def snapshot_url(name: str | None) -> str | None:
    return f"{SERVER}/snapshots/{name}" if name else None


@st.cache_data(max_entries=512, show_spinner=False)
def _fetch_snapshot(name: str) -> bytes:
    """사진 파일 이름은 매번 새로 만들어져 내용이 바뀌지 않으므로 이름으로 캐시한다. 실패는 예외라 캐시되지 않는다."""
    r = httpx.get(snapshot_url(name), timeout=10)
    r.raise_for_status()
    return r.content


def snapshot_bytes(name: str | None) -> bytes | None:
    """사진을 서버에서 받아 온다. 브라우저가 서버 주소에 직접 닿지 않아도 보이게 하기 위해서다."""
    if not name:
        return None
    try:
        return _fetch_snapshot(name)
    except httpx.HTTPError:
        return None


def reward(user_id: int) -> dict:
    return _call("GET", f"/api/users/{user_id}/reward")
