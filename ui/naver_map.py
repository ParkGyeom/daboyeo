"""네이버 지도로 그리는 관리자 지도.

네이버 클라우드 플랫폼 Maps(Dynamic Map)의 Client ID 가 필요하고, 지도를 그릴 때 인터넷이 필요하다.
  - .env 에 NAVER_MAP_KEY_ID=<Client ID> 를 적는다 (저장소에 올리지 않는다)
  - 콘솔의 Web 서비스 URL 에 http://localhost:8501 을 등록한다
키가 없으면 대시보드는 받아 둔 도로선 + 점 그림(Altair, 인터넷 없이)으로 그린다.

지도 페이지는 ui/static/naver_map.html 이고, Streamlit 정적 파일(/app/static/)로 iframe 에 띄운다.
components.html(srcdoc) 로 넣으면 주소가 없는 iframe 이라 네이버 인증이 실패한다(11/13 확인).
정적 파일 서빙은 .streamlit/config.toml 과 scripts/demo.sh 에서 켠다.

결함 목록은 iframe 주소의 # 뒤에 담는다. 목록이 바뀌면 # 만 바뀌어 지도는 다시 만들어지지 않고 점만 다시 찍힌다.
지도는 iframe 이라 점을 눌러도 Streamlit 쪽으로 값이 돌아오지 않는다.
그래서 점을 누르면 지도 위에 상세 말풍선(사진 포함)을 띄우고, 상태 변경은 아래 '결함 번호'에서 한다.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.parse import quote

import streamlit.components.v1 as components

import api

ROOT = Path(__file__).resolve().parents[1]
PAGE = "/app/static/naver_map.html"
SEV_COLOR = {"urgent": "#d62728", "routine": "#f2a900"}


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


def render(rows: list[dict], height: int = 560) -> None:
    """결함 목록(서버 /api/v1/defects 결과)을 네이버 지도에 점으로 그린다."""
    data = [
        {
            "id": api.short(r["id"]), "lat": r["lat"], "lon": r["lon"],
            "type": api.TYPE_KO.get(r["defect_type"], r["defect_type"]),
            "severity": api.RISK_KO.get(r["risk_level"], r["risk_level"]),
            "status": api.STATUS_KO.get(r["status"], r["status"]),
            "color": SEV_COLOR.get(r["risk_level"], "#555"), "jurisdiction": r.get("jurisdiction"),
            "obs": r["observation_count"], "vehicles": r["vehicle_count"], "conf": round(float(r["confidence"]), 3),
            "photo": api.snapshot_url(r.get("snapshot_path")),
        }
        for r in rows
    ]
    blob = base64.urlsafe_b64encode(json.dumps(data, ensure_ascii=False).encode("utf-8")).decode("ascii")
    components.iframe(f"{PAGE}?key={quote(key_id() or '', safe='')}#{blob}", height=height)
