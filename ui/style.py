"""화면 공통 모양 (다보여 웹 화면). 색·글꼴 바탕은 .streamlit/config.toml 의 [theme], 나머지는 여기.

항공사 예약 누리집 같은 차분한 서비스 화면을 참고한다 (모양만 참고, 로고·이름·사진은 쓰지 않는다):
- 색: 짙은 남색(제목·주 버튼·선택), 하늘색(강조 선·작은 표시), 옅은 하늘색·옅은 회색(바탕 띠·타일).
- 모양: 넉넉한 여백, 둥근 상자(10~16px), 알약 모양 버튼·배지, 그림자는 아주 옅게 또는 없음.
- 페이지 위는 옅은 하늘색 띠(위치 표시 + 큰 남색 제목), 맨 아래는 옅은 회청색 바닥글.
글꼴은 Pretendard (인터넷이 없으면 Apple SD Gothic Neo).

페이지는 제목을 st.title 대신 page_header() 로, 흰 상자는 card() 로, 소제목은 section() 으로 만든다.
상태·유형 표시는 badge()·type_chip() 이 돌려주는 HTML 조각을 st.html 안에 넣는다.
Streamlit 내부 요소(data-testid)를 고르는 CSS 는 이 파일에만 둔다 — 버전이 바뀌면 여기만 고친다.
"""

from __future__ import annotations

from html import escape

import streamlit as st

# 색 (이름은 앞서 쓰던 것을 그대로 둔다 — 화면 파일이 style.BRAND 등을 쓴다)
BRAND = "#051766"       # 짙은 남색: 제목·주 버튼·선택
BRAND_DARK = "#030f45"
SKY = "#57bbeb"         # 하늘색: 강조 선·작은 표시
SKY_SOFT = "#e6f3fb"    # 옅은 하늘색: 머리 띠·선택 바탕
INK = "#111111"         # 본문 글자
MUTED = "#5e5e5e"       # 보조 글자
LINE = "#e3e7ee"        # 선
SOFT = "#f6f7f9"        # 타일·표 머리 바탕

# 상태: (글자색, 바탕색). 키는 서버 값(001_init.sql)
STATUS_STYLE = {
    "unreported": ("#c42b1c", "#fdeeec"),
    "reported": ("#051766", "#e6f3fb"),
    "under_repair": ("#8a5300", "#fff3dc"),
}
STATUS_KO = {"unreported": "미신고", "reported": "신고됨", "under_repair": "보수 중"}

# 유형 색은 지도 핀과 같게 (ui/naver_map.py 의 TYPE_COLOR)
TYPE_COLOR = {"pothole": "#e62828", "crack": "#ff9600", "lane_wear": "#e0b800",
              "guardrail_damage": "#28a0e6", "sign_damage": "#3fb43f"}
TYPE_KO = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모",
           "guardrail_damage": "가드레일", "sign_damage": "표지판"}

FONT_CSS = "https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"

_CSS = f"""
<style>
@import url("{FONT_CSS}");
/* 글꼴 이름은 config.toml [theme] font. 여기서는 내려받기만 한다 (넓게 덮어쓰면 아이콘 글꼴까지 바뀌어 글자로 보인다) */

/* 머리말: 흰 바탕, 아래 옅은 선. 메뉴는 남색 글자, 고른 메뉴는 남색 밑줄 */
header[data-testid="stHeader"] {{ background: #fff; border-bottom: 1px solid {LINE}; box-shadow: none; height: 4rem; }}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stTopNavLink"] {{ border-radius: 0 !important; background: transparent !important; padding: 0 .9rem !important; }}
[data-testid="stTopNavLink"] p {{ font-size: 1rem; font-weight: 600; color: {BRAND}; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ box-shadow: inset 0 -2px 0 {BRAND}; }}
[data-testid="stTopNavLink"][aria-current="page"] p {{ font-weight: 800; }}
/* 본문 */
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 5rem; padding-bottom: 1rem; }}
/* 흰 상자 = card() (key 가 card- 로 시작 → class st-key-card-…) */
[class*="st-key-card-"] {{
  background: #fff; border: 1px solid {LINE} !important; border-radius: 14px !important;
  box-shadow: 0 2px 10px rgba(5,23,102,.04);
}}
/* 표 */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
/* 숫자(st.metric) */
[data-testid="stMetric"] {{ background: {SOFT}; border-radius: 12px; padding: 14px 18px; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-weight: 600; }}
[data-testid="stMetricValue"] {{ color: {BRAND}; }}
/* 버튼: 알약 모양. 주 버튼은 남색 */
button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background: {BRAND} !important; border-color: {BRAND} !important; }}
button[kind="secondary"], [data-testid="stBaseButton-secondary"] {{ color: {BRAND}; border-color: #c9d1de; }}
[data-testid="stExpander"] details {{ border-color: {LINE}; border-radius: 12px; }}

/* 다보여 조각 (st.html 로 넣는 것) */
.dby-hero {{
  background: linear-gradient(180deg, {SKY_SOFT} 0%, #f4f9fd 100%);
  border-radius: 16px; padding: 22px 28px 24px; margin: 0 0 1.4rem;
}}
.dby-crumb {{ color: {MUTED}; font-size: .8rem; margin-bottom: .5rem; }}
.dby-crumb span {{ margin: 0 .45em; color: #9aa3b2; }}
.dby-crumb b {{ color: {BRAND}; font-weight: 600; }}
.dby-hero h1 {{ font-size: 1.9rem; font-weight: 700; color: {BRAND}; margin: 0 0 .3rem; padding: 0; line-height: 1.3; letter-spacing: -.02em; }}
.dby-hero p {{ color: #3f4a5c; margin: 0; font-size: .95rem; }}
.dby-badge {{
  display: inline-block; padding: .2em .75em; border-radius: 999px;
  font-size: .78rem; font-weight: 700; white-space: nowrap; line-height: 1.45;
}}
.dby-chip {{
  display: inline-flex; align-items: center; gap: .4em;
  padding: .2em .75em; border-radius: 999px; font-size: .78rem; font-weight: 600;
  background: {SOFT}; color: {INK}; white-space: nowrap; line-height: 1.45;
}}
.dby-chip i {{ width: .5em; height: .5em; border-radius: 50%; display: inline-block; }}
.dby-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: .2rem 0 1.2rem; }}
.dby-stat {{ background: {SOFT}; border-radius: 12px; padding: 16px 20px; }}
.dby-stat .l {{ color: {MUTED}; font-size: .84rem; font-weight: 600; display: flex; align-items: center; gap: .45em; }}
.dby-stat .l i {{ width: .5em; height: .5em; border-radius: 50%; display: inline-block; background: var(--c, {SKY}); }}
.dby-stat .v {{ color: {BRAND}; font-size: 1.75rem; font-weight: 700; line-height: 1.35; }}
.dby-stat .v small {{ font-size: .9rem; font-weight: 600; margin-left: .15em; }}
.dby-stat .s {{ color: {MUTED}; font-size: .76rem; }}
.dby-section {{ display: flex; align-items: baseline; justify-content: space-between; margin: .2rem 0 .7rem; }}
.dby-section h3 {{ font-size: 1.2rem; font-weight: 700; color: {BRAND}; margin: 0; padding: 0; }}
.dby-section span {{ color: {MUTED}; font-size: .82rem; }}
.dby-foot {{
  margin-top: 3rem; padding: 22px 28px; border-radius: 16px; background: #eef2f7;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 18px;
  color: #4b5563; font-size: .8rem; line-height: 1.75;
}}
.dby-foot b {{ display: block; color: {BRAND}; font-size: .86rem; margin-bottom: .2rem; }}
</style>
"""

_MOBILE_CSS = """
<style>
.block-container, [data-testid="stMainBlockContainer"] { max-width: 440px; padding-top: 1rem; }
header[data-testid="stHeader"] { display: none; }
.dby-hero { padding: 18px 20px; }
.dby-hero h1 { font-size: 1.45rem; }
</style>
"""


def apply(mobile: bool = False) -> None:
    """공통 CSS 를 넣는다. mobile=True 는 시민 문자 링크 화면(휴대폰 폭, 머리말 없음)."""
    st.html(_CSS + (_MOBILE_CSS if mobile else ""))


def page_header(title: str, desc: str = "", kicker: str = "") -> None:
    """페이지 머리 띠: 옅은 하늘색 둥근 띠 안에 위치 표시(홈 › 분류 › 제목) + 큰 남색 제목 + 한 줄 설명.
    st.title 대신 쓴다. kicker 는 분류("관리자", "관리자 · 실시간")이고 위치 표시에 들어간다."""
    parts = ["홈", *[p.strip() for p in kicker.split("·") if p.strip()][:1]]
    crumb = "<span>›</span>".join(escape(p) for p in parts) + f"<span>›</span><b>{escape(title)}</b>"
    d = f"<p>{escape(desc)}</p>" if desc else ""
    st.html(f"<div class='dby-hero'><div class='dby-crumb'>{crumb}</div><h1>{escape(title)}</h1>{d}</div>")


def section(title: str, note: str = "") -> None:
    """소제목 (남색 굵은 글자, 오른쪽에 작은 설명)."""
    n = f"<span>{escape(note)}</span>" if note else ""
    st.html(f"<div class='dby-section'><h3>{escape(title)}</h3>{n}</div>")


def card(key: str, **kw):
    """흰 둥근 상자. st.container(border=True) 대신 쓴다. key 는 화면 안에서 겹치지 않게.
    Streamlit 은 테두리 상자를 CSS 로 구분할 표시가 없어 key 로 붙는 class(st-key-card-…)로 고른다."""
    return st.container(border=True, key=f"card-{key}", **kw)


def badge(status: str) -> str:
    """상태 배지 HTML (알약 모양). status 는 서버 값(unreported·reported·under_repair)."""
    fg, bg = STATUS_STYLE.get(status, (MUTED, SOFT))
    return f"<span class='dby-badge' style='color:{fg};background:{bg}'>{escape(STATUS_KO.get(status, status))}</span>"


def type_chip(defect_type: str) -> str:
    """결함 유형 표시 HTML (지도 핀과 같은 색 점 + 이름, 알약 모양)."""
    c = TYPE_COLOR.get(defect_type, MUTED)
    return f"<span class='dby-chip'><i style='background:{c}'></i>{escape(TYPE_KO.get(defect_type, defect_type))}</span>"


def stat_cards(items: list[tuple]) -> None:
    """숫자 타일 줄 (옅은 회색 둥근 타일). items = [(이름, 값, 점 색, 작은 설명), …]
    또는 단위를 붙여 (이름, 값, 색, 설명, "건")."""
    def cell(label, v, c, sub, unit=""):
        u = f"<small>{escape(unit)}</small>" if unit else ""
        return (f"<div class='dby-stat' style='--c:{c}'><div class='l'><i></i>{escape(label)}</div>"
                f"<div class='v'>{escape(str(v))}{u}</div><div class='s'>{escape(sub)}</div></div>")

    st.html(f"<div class='dby-stats'>{''.join(cell(*it) for it in items)}</div>")


def footer() -> None:
    """맨 아래 바닥글 (옅은 회청색 띠, 칸 나눔). 실제 기관 시스템으로 오해하지 않게 '모의'임을 밝힌다."""
    st.html(
        "<div class='dby-foot'>"
        "<div><b>다보여</b>블랙박스 영상으로 도로 결함을 찾고<br>여러 차량의 관측을 묶어 시민 신고를 돕습니다.</div>"
        "<div><b>시연 안내</b>학술제 시연용 모의 시스템입니다.<br>문자·신고서 제출은 모두 모의이며 실제 기관과 연결되지 않습니다.</div>"
        "<div><b>출처</b>지도 © NAVER Corp.<br>결함 탐지: 다보여 AI 모델 (공개 도로 영상·AI Hub 데이터 학습)</div>"
        "</div>"
    )
