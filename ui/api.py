"""화면이 쓰는 서버 API(/api/v1) 호출. 화면은 DB에 직접 접근하지 않는다.

서버가 주는 값은 001_init.sql 의 영어 이름(unreported, urgent, …)이고, 한국어 표시는 여기서 붙인다.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import streamlit as st

SERVER = os.environ.get("SERVER_URL", "http://localhost:8000")
API = SERVER + "/api/v1"

TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모",
           "guardrail_damage": "가드레일", "sign_damage": "표지판"}
RISK_KO = {"urgent": "빠른 보수", "routine": "정기 보수"}
STATUS_KO = {"unreported": "미신고", "reported": "신고됨", "under_repair": "보수 중"}


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _call(method: str, path: str, **kw):
    try:
        r = httpx.request(method, API + path, timeout=10, **kw)
    except httpx.HTTPError as e:
        raise ApiError(0, f"서버에 연결할 수 없다 ({SERVER}): {e}") from e
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(r.status_code, str(detail))
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def users() -> dict[str, str]:
    """시민 id → '시민 A (bb-01)'."""
    return {u["id"]: f"{u['name']} ({u['vehicle_label']})" for u in _call("GET", "/users")}


def pick_user(key: str = "user_id") -> str:
    """시민 선택 상자. 서버에 닿지 않으면 화면을 멈춘다."""
    try:
        us = users()
    except ApiError as e:
        st.error(e.detail)
        st.stop()
    return st.selectbox("시민", list(us), format_func=us.get, key=key)


def clock(iso: str) -> str:
    """서버가 준 시각(ISO 8601)을 한국 시간 '오후 3:05' 꼴로."""
    t = datetime.fromisoformat(iso).astimezone(timezone(timedelta(hours=9)))
    return f"{'오전' if t.hour < 12 else '오후'} {(t.hour % 12) or 12}:{t.minute:02d}"


def findings(user_id: str) -> list[dict]:
    return _call("GET", f"/users/{user_id}/findings")


def notices(user_id: str) -> list[dict]:
    """끝난 주행 중 아직 답하지 않은 항목이 남은 것. 최근 것부터."""
    return _call("GET", f"/users/{user_id}/notices")


def report(defect_id: str, user_id: str) -> dict:
    return _call("POST", f"/defects/{defect_id}/report", json={"user_id": user_id})


def skip(defect_id: str, user_id: str) -> dict:
    return _call("POST", f"/defects/{defect_id}/skip", json={"user_id": user_id})


def get_report(report_id: str) -> dict:
    return _call("GET", f"/reports/{report_id}")


def user_reports(user_id: str) -> list[dict]:
    return _call("GET", f"/users/{user_id}/reports")


def defects(type: str | None = None, status: str | None = None) -> list[dict]:
    params = {k: v for k, v in {"type": type, "status": status}.items() if v}
    return _call("GET", "/defects", params=params)


def set_status(defect_id: str, status: str) -> dict:
    return _call("PATCH", f"/defects/{defect_id}/status", json={"status": status})


def reward(user_id: str) -> dict:
    return _call("GET", f"/users/{user_id}/reward")


def short(uuid_str: str) -> str:
    """화면에 보일 짧은 번호 (UUID 앞 8자리)."""
    return str(uuid_str)[:8]


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
