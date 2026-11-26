"""화면 공통 모양 (다보여 웹 화면). 색·글꼴 바탕은 .streamlit/config.toml 의 [theme], 나머지는 여기.

공공기관 업무 화면 느낌으로 맞춘다: 흰 바탕, 얇은 회색 선, 모서리 거의 없음(2~4px), 그림자·그러데이션·이모지 없음,
제목 아래 진한 선, 위치 표시(홈 › 관리자 › 화면), 표 머리는 회색 바탕 + 위쪽 진한 선, 상태는 테두리 배지.
글꼴은 Pretendard GOV (공공 누리집에서 많이 쓰는 글꼴, 인터넷이 없으면 Apple SD Gothic Neo).

페이지는 제목을 st.title 대신 page_header() 로, 흰 상자는 card() 로 만든다.
상태·유형 표시는 badge()·type_chip() 이 돌려주는 HTML 조각을 st.html 안에 넣는다.
Streamlit 내부 요소(data-testid)를 고르는 CSS 는 이 파일에만 둔다 — 버전이 바뀌면 여기만 고친다.
"""

from __future__ import annotations

from html import escape

import streamlit as st

# 색 (이름은 앞서 쓰던 것을 그대로 둔다 — 화면 파일이 style.BRAND 등을 쓴다)
BRAND = "#0b50b4"       # 주 색 (버튼·링크·선택)
BRAND_DARK = "#0a2f6b"  # 머리 띠·강조 제목
INK = "#1e2124"         # 본문 글자
MUTED = "#5d6670"       # 보조 글자
LINE = "#d5d9de"        # 선
SOFT = "#f3f5f7"        # 표 머리·보조 바탕

# 상태: (글자·테두리 색, 바탕색). 키는 서버 값(001_init.sql)
STATUS_STYLE = {
    "unreported": ("#c42b1c", "#fff5f4"),
    "reported": ("#0b50b4", "#f2f6fd"),
    "under_repair": ("#9a5a00", "#fff8ec"),
}
STATUS_KO = {"unreported": "미신고", "reported": "신고됨", "under_repair": "보수 중"}

# 유형 색은 지도 핀과 같게 (ui/naver_map.py 의 TYPE_COLOR)
TYPE_COLOR = {"pothole": "#e62828", "crack": "#ff9600", "lane_wear": "#e0b800",
              "guardrail_damage": "#28a0e6", "sign_damage": "#3fb43f"}
TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모",
           "guardrail_damage": "가드레일", "sign_damage": "표지판"}

FONT_CSS = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-gov-dynamic-subset.min.css"

_CSS = f"""
<style>
@import url("{FONT_CSS}");
/* 글꼴 이름은 config.toml [theme] font. 여기서는 내려받기만 한다 (넓게 덮어쓰면 아이콘 글꼴까지 바뀌어 글자로 보인다) */
/* 머리말: 흰 바탕 + 아래 진한 남색 선. 상단 메뉴(st.navigation position=top)가 여기 들어간다 */
header[data-testid="stHeader"] {{
  background: #fff; border-bottom: 2px solid {BRAND_DARK}; box-shadow: none;
}}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stTopNavLink"] {{ border-radius: 0 !important; }}
[data-testid="stTopNavLink"] p {{ font-weight: 600; color: {INK}; }}
[data-testid="stTopNavLink"][aria-current="page"],
[data-testid="stTopNavLink"]:has(+ *[aria-current="page"]) {{ background: transparent !important; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ box-shadow: inset 0 -3px 0 {BRAND}; }}
[data-testid="stTopNavLink"][aria-current="page"] p {{ color: {BRAND}; font-weight: 800; }}
/* 본문 */
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 4.6rem; padding-bottom: 2rem; }}
/* 흰 상자 = card() 로 만든 상자 (key 가 card- 로 시작 → class st-key-card-…) */
[class*="st-key-card-"] {{ background: #fff; border-radius: 4px !important; box-shadow: none; }}
/* 표 */
[data-testid="stDataFrame"] {{ border-top: 2px solid {INK}; }}
/* 숫자(st.metric) */
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 4px; padding: 12px 16px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; }}
/* 버튼·입력: 모서리 작게 */
button, [data-baseweb="select"] > div, [data-baseweb="input"], [data-testid="stExpander"] details {{ border-radius: 4px !important; }}
[data-testid="stExpander"] details {{ border-color: {LINE}; }}

/* 다보여 조각 (st.html 로 넣는 것) */
.dby-crumb {{ color: {MUTED}; font-size: .82rem; margin-bottom: .35rem; }}
.dby-crumb b {{ color: {INK}; font-weight: 600; }}
.dby-crumb span {{ margin: 0 .4em; color: #9aa1a9; }}
.dby-head {{ margin: 0 0 1.1rem; padding-bottom: .8rem; border-bottom: 2px solid {INK}; }}
.dby-head h1 {{ font-size: 1.7rem; font-weight: 800; color: {INK}; margin: 0 0 .2rem; padding: 0; line-height: 1.3; letter-spacing: -.02em; }}
.dby-head p {{ color: {MUTED}; margin: 0; font-size: .92rem; }}
.dby-badge {{
  display: inline-block; padding: .1em .5em; border: 1px solid currentColor; border-radius: 2px;
  font-size: .78rem; font-weight: 700; white-space: nowrap; line-height: 1.5;
}}
.dby-chip {{
  display: inline-flex; align-items: center; gap: .35em;
  padding: .1em .5em; border: 1px solid {LINE}; border-radius: 2px; font-size: .78rem; font-weight: 600;
  background: #fff; color: {INK}; white-space: nowrap; line-height: 1.5;
}}
.dby-chip i {{ width: .55em; height: .55em; display: inline-block; }}
.dby-stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  border: 1px solid {LINE}; border-top: 2px solid {INK}; margin: .2rem 0 1rem; background: #fff;
}}
.dby-stat {{ padding: 12px 18px; border-right: 1px solid {LINE}; }}
.dby-stat:last-child {{ border-right: 0; }}
.dby-stat .l {{ color: {MUTED}; font-size: .82rem; font-weight: 600; display: flex; align-items: center; gap: .4em; }}
.dby-stat .l i {{ width: .55em; height: .55em; display: inline-block; background: var(--c, {BRAND}); }}
.dby-stat .v {{ color: {INK}; font-size: 1.6rem; font-weight: 800; line-height: 1.3; }}
.dby-stat .v small {{ font-size: .85rem; font-weight: 600; color: {MUTED}; margin-left: .15em; }}
.dby-stat .s {{ color: {MUTED}; font-size: .76rem; }}
.dby-section {{ font-size: 1.05rem; font-weight: 800; color: {INK}; margin: .2rem 0 .5rem; }}
.dby-foot {{
  margin-top: 2.5rem; padding: 1rem 0 .4rem; border-top: 1px solid {LINE};
  color: {MUTED}; font-size: .78rem; line-height: 1.7;
}}
.dby-foot b {{ color: {INK}; }}
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
    """페이지 머리말: 위치 표시(홈 › 분류 › 제목) + 제목 + 한 줄 설명, 아래 진한 선. st.title 대신 쓴다.
    kicker 는 분류("관리자", "관리자 · 실시간")이고 위치 표시에 들어간다."""
    parts = ["홈", *[p.strip() for p in kicker.split("·") if p.strip()][:1]]
    crumb = "<span>›</span>".join(escape(p) for p in parts) + f"<span>›</span><b>{escape(title)}</b>"
    d = f"<p>{escape(desc)}</p>" if desc else ""
    st.html(f"<div class='dby-crumb'>{crumb}</div><div class='dby-head'><h1>{escape(title)}</h1>{d}</div>")


def section(title: str) -> None:
    """상자 안 소제목."""
    st.html(f"<div class='dby-section'>{escape(title)}</div>")


def card(key: str, **kw):
    """흰 상자 (얇은 선, 그림자 없음). st.container(border=True) 대신 쓴다. key 는 화면 안에서 겹치지 않게.
    Streamlit 은 테두리 상자를 CSS 로 구분할 표시가 없어 key 로 붙는 class(st-key-card-…)로 고른다."""
    return st.container(border=True, key=f"card-{key}", **kw)


def badge(status: str) -> str:
    """상태 배지 HTML (테두리 네모). status 는 서버 값(unreported·reported·under_repair)."""
    fg, bg = STATUS_STYLE.get(status, (MUTED, "#fff"))
    return f"<span class='dby-badge' style='color:{fg};background:{bg}'>{escape(STATUS_KO.get(status, status))}</span>"


def type_chip(defect_type: str) -> str:
    """결함 유형 표시 HTML (지도 핀과 같은 색 네모 + 이름)."""
    c = TYPE_COLOR.get(defect_type, MUTED)
    return f"<span class='dby-chip'><i style='background:{c}'></i>{escape(TYPE_KO.get(defect_type, defect_type))}</span>"


def stat_cards(items: list[tuple[str, int | str, str, str]]) -> None:
    """숫자 줄 (한 상자를 칸으로 나눈 현황표). items = [(이름, 값, 표시 색, 작은 설명), …]
    또는 단위를 붙여 (이름, 값, 색, 설명, "건")."""
    def cell(label, v, c, sub, unit=""):
        u = f"<small>{escape(unit)}</small>" if unit else ""
        return (f"<div class='dby-stat' style='--c:{c}'><div class='l'><i></i>{escape(label)}</div>"
                f"<div class='v'>{escape(str(v))}{u}</div><div class='s'>{escape(sub)}</div></div>")

    cells = "".join(cell(*it) for it in items)
    st.html(f"<div class='dby-stats'>{cells}</div>")


def footer() -> None:
    """맨 아래 안내. 실제 기관 시스템으로 오해하지 않게 '학술제 시연용 모의 시스템'임을 밝힌다."""
    st.html(
        "<div class='dby-foot'><b>다보여</b> 도로 결함 확인·신고 지원 시스템 · 학술제 시연용 모의 시스템<br>"
        "실제 기관 접수와 연결되어 있지 않으며, 문자·신고서 제출은 모두 모의입니다. "
        "지도 © NAVER Corp. · 도로 영상 AI 결함 탐지</div>"
    )
