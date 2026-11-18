"""네이버 지도로 그리는 관리자 지도.

네이버 클라우드 플랫폼 Maps(Dynamic Map)의 Client ID 가 필요하고, 지도를 그릴 때 인터넷이 필요하다.
  - .env 에 NAVER_MAP_KEY_ID=<Client ID> 를 적는다 (저장소에 올리지 않는다)
  - 콘솔의 Web 서비스 URL 에 http://localhost:8501 을 등록한다
키가 없으면 대시보드는 받아 둔 도로선 + 점 그림(Altair, 인터넷 없이)으로 그린다.

지도 페이지는 ui/static/naver_map.html 이고, Streamlit 정적 파일(/app/static/)로 iframe 에 띄운다.
components.html(srcdoc) 로 넣으면 주소가 없는 iframe 이라 네이버 인증이 실패한다(11/13 확인).
정적 파일 서빙은 .streamlit/config.toml 과 scripts/demo.sh 에서 켠다.

처음 화면은 대한민국 전체이고, 결함마다 깜빡이는 신호를 띄운다. 전국 축척에서 가까운 점은 숫자 묶음이 된다.
새로 생긴 결함(이 화면을 연 뒤 처음 본 것)은 FRESH_S 초 동안 크게 깜빡이고 '새 결함' 표시가 붙는다.

결함 목록은 iframe 주소의 # 뒤에 담는다. 목록이 바뀌면 # 만 바뀌어 지도는 다시 만들어지지 않고 점만 다시 찍힌다.
지도는 iframe 이라 점을 눌러도 Streamlit 쪽으로 값이 돌아오지 않는다.
그래서 점을 누르면 지도 위에 상세 말풍선(사진 포함)을 띄우고, 상태 변경은 아래 '결함 고르기'에서 한다.
핀 색은 결함 유형(TYPE_COLOR), 모양은 상태, 빨간 테두리는 빠른 보수다.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

import api

ROOT = Path(__file__).resolve().parents[1]
PAGE = "/app/static/naver_map.html"
MINI_PAGE = "/app/static/mini_map.html"
FRESH_S = 20  # 새 결함을 강조하는 시간(초). 대시보드는 5초마다 다시 그리므로 네 번 정도 보인다
SEV_COLOR = {"urgent": "#d62728", "routine": "#f2a900"}
# 핀 색 = 결함 유형. 블랙박스 화면의 탐지 상자 색(edge/blackbox/live.py COLORS, BGR)과 같다
TYPE_COLOR = {"pothole": "#e62828", "crack": "#ff9600", "lane_wear": "#ffdc00",
              "guardrail_damage": "#28a0e6", "sign_damage": "#50c850"}


def _ver(page: str) -> str:
    """지도 페이지 파일의 수정 시각. 주소에 붙여 브라우저가 고치기 전 페이지를 캐시에서 꺼내 쓰지 않게 한다
    (Streamlit 정적 파일은 Cache-Control 이 없어 브라우저가 알아서 캐시한다)."""
    try:
        return str(int((ROOT / "ui" / page.removeprefix("/app/")).stat().st_mtime))
    except OSError:
        return "0"


def key_id() -> str | None:
    """환경 변수가 우선이고, 없으면 프로젝트 폴더의 .env 에서 읽는다."""
    if v := os.environ.get("NAVER_MAP_KEY_ID", "").strip():
        return v
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "NAVER_MAP_KEY_ID" and v.strip():
                return v.strip()
    return None


def place(address: str | None) -> str | None:
    """저장해 둔 주소를 화면 글자로: "서울 강남구 강남대로 396 부근". 없으면 None (server/geocode.py)."""
    return f"{address} 부근" if address else None


def short_place(address: str | None) -> str:
    """제목에 쓰는 짧은 위치: "서울 강남구 봉은사로 102" → "봉은사로 102". 결함 번호(id) 대신 쓴다.

    도로명(…로·…길)부터 끝까지, 도로명이 없으면(지번 주소) 마지막 두 마디.
    주소를 아직 못 받았으면(None) "위치 확인 중", 네이버가 주소 없음으로 답한 곳("")은 "주소 없는 곳".
    """
    if address is None:
        return "위치 확인 중"
    words = address.split()
    if not words:
        return "주소 없는 곳"
    for i in range(len(words) - 1, -1, -1):
        if words[i].endswith(("로", "길")):
            return " ".join(words[i:])
    return " ".join(words[-2:])


def title(r: dict) -> str:
    """결함 한 건의 제목: "봉은사로 102 균열"."""
    return f"{short_place(r.get('address'))} {api.TYPE_KO.get(r['defect_type'], r['defect_type'])}"


def fresh_ids(rows: list[dict], state: dict | None = None, now: float | None = None) -> set[str]:
    """이번에 새로 강조할 결함 id. 화면을 처음 그릴 때 있던 결함은 새것이 아니다.

    state 는 세션마다 하나(기본 st.session_state). {"seen": 처음 본 시각(id별), "ready": 첫 그림 여부}
    """
    state = st.session_state.setdefault("naver_map_seen", {}) if state is None else state
    now = time.time() if now is None else now
    seen = state.setdefault("seen", {})
    first = not state.get("ready")
    for r in rows:
        seen.setdefault(r["id"], 0.0 if first else now)
    state["ready"] = True
    return {r["id"] for r in rows if now - seen[r["id"]] < FRESH_S}


def render(rows: list[dict], height: int = 560) -> None:
    """결함 목록(서버 /api/v1/defects 결과)을 네이버 지도에 깜빡이는 신호로 그린다."""
    fresh = fresh_ids(rows)
    data = [
        {
            "id": api.short(r["id"]), "title": title(r), "lat": r["lat"], "lon": r["lon"],
            "type": api.TYPE_KO.get(r["defect_type"], r["defect_type"]),
            "type_color": TYPE_COLOR.get(r["defect_type"], "#555"), "urgent": r["risk_level"] == "urgent",
            "severity": api.RISK_KO.get(r["risk_level"], r["risk_level"]),
            "status": api.STATUS_KO.get(r["status"], r["status"]),
            "color": SEV_COLOR.get(r["risk_level"], "#555"), "jurisdiction": r.get("jurisdiction"),
            "address": place(r.get("address")),
            "obs": r["observation_count"], "vehicles": r["vehicle_count"], "conf": round(float(r["confidence"]), 3),
            "photo": api.snapshot_url(r.get("snapshot_path")), "fresh": r["id"] in fresh,
        }
        for r in rows
    ]
    blob = base64.urlsafe_b64encode(json.dumps(data, ensure_ascii=False).encode("utf-8")).decode("ascii")
    components.iframe(f"{PAGE}?v={_ver(PAGE)}&key={quote(key_id() or '', safe='')}#{blob}", height=height)


def mini(lat: float, lon: float, risk_level: str | None = None, height: int = 170,
         defect_type: str | None = None) -> None:
    """결함 한 곳의 작은 지도(핀 하나). 시민 확인 화면·신고서에서 좌표 글자 대신 쓴다.

    핀 색은 유형(관리자 지도와 같은 TYPE_COLOR), 퍼지는 고리 색은 위험도.

    키가 없으면 받아 둔 도로선 위에 점(인터넷 없이), 도로도 없으면 좌표 글자.
    키가 있어도 인터넷이 없으면 지도 페이지가 좌표 글자로 대신 보인다.
    """
    if key := key_id():
        color = TYPE_COLOR.get(defect_type or "", SEV_COLOR.get(risk_level or "", "#d62728"))
        ring = SEV_COLOR.get(risk_level or "", color)
        components.iframe(
            f"{MINI_PAGE}?v={_ver(MINI_PAGE)}&key={quote(key, safe='')}&lat={lat:.7f}&lon={lon:.7f}"
            f"&color={quote(color, safe='')}&ring={quote(ring, safe='')}",
            height=height,
        )
        st.caption("지도 © NAVER Corp.")
        return
    _mini_roads(lat, lon, TYPE_COLOR.get(defect_type or "", SEV_COLOR.get(risk_level or "", "#d62728")), height)


def _mini_roads(lat: float, lon: float, color: str, height: int) -> None:
    """키가 없을 때의 작은 지도: 결함 주변 250m 의 도로선 + 점."""
    import math

    import altair as alt
    import pandas as pd

    import basemap

    m = 250
    dlat, dlon = m / 111_320, m / (111_320 * math.cos(math.radians(lat)))
    s, w, n, e = lat - dlat, lon - dlon, lat + dlat, lon + dlon
    roads = basemap.load_roads((s, w, n, e))
    if not roads:
        st.caption(f"위치 {lat:.5f}, {lon:.5f}")
        return
    xs = alt.Scale(domain=[w, e], zero=False, nice=False)
    ys = alt.Scale(domain=[s, n], zero=False, nice=False)
    rdf = pd.DataFrame([{"rid": r["id"], "i": i, "lat": la, "lon": lo}
                        for r in roads for i, (la, lo) in enumerate(r["coords"])])
    lines = (alt.Chart(rdf).mark_line(clip=True, color="#8a8f98", opacity=0.6, strokeWidth=2)
             .encode(x=alt.X("lon:Q", scale=xs, axis=None), y=alt.Y("lat:Q", scale=ys, axis=None),
                     detail="rid:N", order="i:Q"))
    pt = (alt.Chart(pd.DataFrame([{"lat": lat, "lon": lon}]))
          .mark_point(filled=True, size=220, color=color, stroke="white", strokeWidth=2, opacity=1)
          .encode(x=alt.X("lon:Q", scale=xs, axis=None), y=alt.Y("lat:Q", scale=ys, axis=None)))
    st.altair_chart(alt.layer(lines, pt).properties(height=height).configure_view(stroke=None), width="stretch")
    st.caption(basemap.ATTRIBUTION)
