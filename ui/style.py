"""화면 공통 모양 (다보여 웹 화면). 색·글꼴 바탕은 .streamlit/config.toml 의 [theme], 나머지는 여기.

페이지는 맨 위에서 style.apply() 를 한 번 부르고, 제목은 st.title 대신 style.page_header() 를 쓴다.
상태·유형 표시는 badge()·type_chip() 이 돌려주는 HTML 조각을 st.html 안에 넣는다.
Streamlit 내부 요소(data-testid)를 고르는 CSS 는 이 파일에만 둔다 — 버전이 바뀌면 여기만 고친다.
"""

from __future__ import annotations

from html import escape

import streamlit as st

BRAND = "#2f4fb5"
BRAND_DARK = "#1e3a8a"
INK = "#1f2937"
MUTED = "#6b7280"
LINE = "#e3e8f2"

# 상태: (글자색, 바탕색). 키는 서버 값(001_init.sql)
STATUS_STYLE = {
    "unreported": ("#b42318", "#fdecea"),
    "reported": ("#1e3a8a", "#e8eefc"),
    "under_repair": ("#9a5b00", "#fff4e0"),
}
STATUS_KO = {"unreported": "미신고", "reported": "신고됨", "under_repair": "보수 중"}

# 유형 색은 지도 핀과 같게 (ui/naver_map.py 의 TYPE_COLOR)
TYPE_COLOR = {"pothole": "#e62828", "crack": "#ff9600", "lane_wear": "#e0b800",
              "guardrail_damage": "#28a0e6", "sign_damage": "#3fb43f"}
TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모",
           "guardrail_damage": "가드레일", "sign_damage": "표지판"}

_CSS = f"""
<style>
/* 머리말: 흰 바탕 + 아래 선. 상단 메뉴(st.navigation position=top)가 여기 들어간다 */
header[data-testid="stHeader"] {{
  background: #ffffffee; backdrop-filter: blur(6px);
  border-bottom: 1px solid {LINE};
}}
[data-testid="stDecoration"] {{ display: none; }}
/* 본문 */
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 4.5rem; padding-bottom: 3rem; }}
/* 테두리 있는 상자 = 흰 카드 */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
div[data-testid="stVerticalBlock"][class*="border"] {{
  background: #fff;
  box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 4px 16px rgba(16,24,40,.04);
}}
/* 숫자(st.metric)도 카드처럼 */
[data-testid="stMetric"] {{
  background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 18px;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; }}

/* 다보여 조각 (st.html 로 넣는 것) */
.dby-head {{ margin: 0 0 1.2rem; }}
.dby-head .kicker {{ color: {BRAND}; font-weight: 700; font-size: .82rem; letter-spacing: .04em; }}
.dby-head h1 {{ font-size: 1.85rem; font-weight: 800; color: {INK}; margin: .15rem 0 .25rem; padding: 0; line-height: 1.25; }}
.dby-head p {{ color: {MUTED}; margin: 0; font-size: .95rem; }}
.dby-badge {{
  display: inline-flex; align-items: center; gap: .35em;
  padding: .18em .65em; border-radius: 999px; font-size: .8rem; font-weight: 700; white-space: nowrap;
}}
.dby-badge::before {{ content: ""; width: .5em; height: .5em; border-radius: 50%; background: currentColor; }}
.dby-chip {{
  display: inline-flex; align-items: center; gap: .4em;
  padding: .18em .6em; border-radius: 6px; font-size: .8rem; font-weight: 700;
  background: #f3f4f6; color: {INK}; white-space: nowrap;
}}
.dby-chip i {{ width: .6em; height: .6em; border-radius: 2px; display: inline-block; }}
.dby-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: .2rem 0 1rem; }}
.dby-stat {{
  background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 18px;
  border-top: 4px solid var(--c, {BRAND});
}}
.dby-stat .l {{ color: {MUTED}; font-size: .82rem; font-weight: 600; }}
.dby-stat .v {{ color: {INK}; font-size: 1.9rem; font-weight: 800; line-height: 1.2; }}
.dby-stat .s {{ color: {MUTED}; font-size: .78rem; }}
</style>
"""

_MOBILE_CSS = """
<style>
.block-container, [data-testid="stMainBlockContainer"] { max-width: 440px; padding-top: 1rem; }
header[data-testid="stHeader"] { display: none; }
</style>
"""


def apply(mobile: bool = False) -> None:
    """공통 CSS 를 넣는다. mobile=True 는 시민 문자 링크 화면(휴대폰 폭, 머리말 없음)."""
    st.html(_CSS + (_MOBILE_CSS if mobile else ""))


def page_header(title: str, desc: str = "", kicker: str = "") -> None:
    """페이지 머리말: 작은 분류 글자 + 큰 제목 + 한 줄 설명. st.title 대신 쓴다."""
    k = f"<div class='kicker'>{escape(kicker)}</div>" if kicker else ""
    d = f"<p>{escape(desc)}</p>" if desc else ""
    st.html(f"<div class='dby-head'>{k}<h1>{escape(title)}</h1>{d}</div>")


def badge(status: str) -> str:
    """상태 배지 HTML. status 는 서버 값(unreported·reported·under_repair)."""
    fg, bg = STATUS_STYLE.get(status, (MUTED, "#f3f4f6"))
    return f"<span class='dby-badge' style='color:{fg};background:{bg}'>{escape(STATUS_KO.get(status, status))}</span>"


def type_chip(defect_type: str) -> str:
    """결함 유형 칩 HTML (지도 핀과 같은 색 네모 + 이름)."""
    c = TYPE_COLOR.get(defect_type, MUTED)
    return f"<span class='dby-chip'><i style='background:{c}'></i>{escape(TYPE_KO.get(defect_type, defect_type))}</span>"


def stat_cards(items: list[tuple[str, int | str, str, str]]) -> None:
    """숫자 카드 줄. items = [(이름, 값, 위 띠 색, 작은 설명), …]."""
    cells = "".join(
        f"<div class='dby-stat' style='--c:{c}'><div class='l'>{escape(label)}</div>"
        f"<div class='v'>{escape(str(v))}</div><div class='s'>{escape(sub)}</div></div>"
        for label, v, c, sub in items
    )
    st.html(f"<div class='dby-stats'>{cells}</div>")
