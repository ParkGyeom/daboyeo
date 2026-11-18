"""화면이 쓰는 서버 API(/api/v1) 호출. 화면은 DB에 직접 접근하지 않는다.

서버가 주는 값은 001_init.sql 의 영어 이름(unreported, pothole, …)이고, 한국어 표시는 여기서 붙인다.
risk_level 은 서버·신고서 데이터에는 남아 있지만 화면에는 쓰지 않는다 (판별은 우리 몫이 아니다).
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, timezone

import httpx
import streamlit as st

SERVER = os.environ.get("SERVER_URL", "http://localhost:8000")
API = SERVER + "/api/v1"

TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모",
           "guardrail_damage": "가드레일", "sign_damage": "표지판"}
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


KST = timezone(timedelta(hours=9))


@st.cache_data(ttl=30, show_spinner=False)
def _users() -> list[dict]:
    return _call("GET", "/users")


def plates() -> list[str]:
    """등록된 차량 번호 (관리자 지도의 차량 고르기)."""
    return sorted({u["vehicle_plate"] for u in _users() if u.get("vehicle_plate")})


def clock(iso: str) -> str:
    """서버가 준 시각(ISO 8601)을 한국 시간 '오후 3:05' 꼴로."""
    t = datetime.fromisoformat(iso).astimezone(KST)
    return f"{'오전' if t.hour < 12 else '오후'} {(t.hour % 12) or 12}:{t.minute:02d}"


def when(iso: str) -> str:
    """날짜까지: '11월 18일 오후 3:05'."""
    t = datetime.fromisoformat(iso).astimezone(KST)
    return f"{t.month}월 {t.day}일 {clock(iso)}"


def short_when(iso: str) -> str:
    """표에 쓰는 짧은 시각: '11/18 15:05'."""
    return f"{datetime.fromisoformat(iso).astimezone(KST):%m/%d %H:%M}"


def get_report(report_id: str) -> dict:
    return _call("GET", f"/reports/{report_id}")


def reports() -> list[dict]:
    """관리자 신고 내역: 모든 신고서 (누가·어느 결함·지금 상태). 최근 것부터."""
    return _call("GET", "/reports")


def reporters() -> list[dict]:
    """시민(사람)별 신고 누적 수 (이름·차량 번호). 많이 낸 순."""
    return _call("GET", "/reporters")


def defects(type: str | None = None, status: str | None = None, region1: str | None = None,
            region2: str | None = None, region3: str | None = None, vehicle: str | None = None,
            date_from: date | None = None, date_to: date | None = None) -> list[dict]:
    """관리자 지도의 결함 목록. None 인 조건은 걸지 않는다. region3="" 는 동이 없는 결함."""
    params = {"type": type, "status": status, "region1": region1, "region2": region2, "region3": region3,
              "vehicle": vehicle, "date_from": date_from and date_from.isoformat(),
              "date_to": date_to and date_to.isoformat()}
    return _call("GET", "/defects", params={k: v for k, v in params.items() if v is not None})


def set_status(defect_id: str, status: str) -> dict:
    return _call("PATCH", f"/defects/{defect_id}/status", json={"status": status})


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


# ---------- 문자 링크 (가입·로그인 없이 토큰으로) ----------

def link_findings(token: str) -> dict:
    """{'user_name', 'expires_at', 'findings': [...], 'report_count'}. 없는 링크는 ApiError(404), 만료는 ApiError(410)."""
    return _call("GET", f"/m/{token}/findings")


def link_report(token: str, defect_id: str) -> dict:
    return _call("POST", f"/m/{token}/defects/{defect_id}/report")


def link_skip(token: str, defect_id: str) -> dict:
    return _call("POST", f"/m/{token}/defects/{defect_id}/skip")


def link_get_report(token: str, report_id: str) -> dict:
    return _call("GET", f"/m/{token}/reports/{report_id}")


def inbox() -> list[dict]:
    """모의 문자 목록 (시연용 메시지함). 최근 것부터."""
    return _call("GET", "/notifications")


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()<>#+\-.!|~$])")


def _md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def sms_markdown(body: str, url: str) -> str:
    """모의 문자 본문을 줄을 지켜 보여 줄 마크다운으로 (메시지함). 본문 속 링크 주소는 누를 수 있는 링크로 바꾼다.

    본문 문구가 바뀌어도 깨지지 않게 문구로 자르지 않는다: 링크는 서버가 토큰으로 만든 url 그대로 찾는다.
    마크다운 기호(~ · [ 등)는 글자 그대로 보이게 막는다 ("12:54~12:58" 이 취소선이 되지 않게).
    """
    lines = []
    for line in body.splitlines():
        head, found, tail = line.partition(url)
        out = _md_escape(head)
        if found:
            out += f"[{_md_escape(url)}]({url})" + _md_escape(tail)
        lines.append(out)
    return "  \n".join(lines)
