"""문자 링크 화면 (시민, 휴대폰 폭). 가입·로그인 없이 링크의 토큰으로 그 시민의 확인 목록을 연다.

[신고하기] 한 번으로 신고서가 만들어지고, 같은 화면에 신고서 요약이 나온다.
모양은 항공사 모바일 앱처럼: 흰 위쪽 바, 하늘색 둥근 띠(이름 + 신고 건수), 결함은 탑승권 같은 둥근 카드,
알약 버튼. 폭·머리말 숨김은 app.py 의 style.apply(mobile=True).
"""

import base64
from html import escape
from pathlib import Path

import streamlit as st

import api
import naver_map
import style

# 이 화면에만 쓰는 모양. 문자 링크로 들어오면 이 페이지 하나만 뜨므로 버튼·사진 CSS 가 다른 화면에 번지지 않는다
# (style.py 로 옮길 것: 휴대폰 위쪽 바, 라벨·값 2열 칸(lc-grid), 탑승권 절취선(lc-perf) — 다른 휴대폰 화면이 생기면 공통으로)
_CSS = f"""
<style>
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 0; padding-left: 16px; padding-right: 16px; }}
/* 위쪽 바: 흰 바탕 + 로고. 스크롤해도 위에 붙어 있다 (st.html 은 자기 상자 안에 들어가므로 그 상자를 붙인다) */
[data-testid="stElementContainer"]:has(.lc-bar) {{ position: sticky; top: 0; z-index: 20; }}
.lc-bar {{
  margin: 0 -16px .1rem; padding: 10px 16px; background: #fff; border-bottom: 1px solid {style.LINE};
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}}
.lc-bar img {{ height: 32px; width: auto; display: block; }}
.lc-bar .who {{ font-size: .8rem; color: {style.MUTED}; font-weight: 600; white-space: nowrap; }}
/* 하늘색 띠: 이름 + 신고 건수(마일리지 표시처럼 큰 남색 숫자) */
.lc-hero {{ background: linear-gradient(180deg, {style.SKY_SOFT} 0%, #f4f9fd 100%); border-radius: 16px; padding: 18px 20px 16px; }}
.lc-hero h2 {{ font-size: 1.35rem; font-weight: 700; color: {style.BRAND}; margin: 0; padding: 0; line-height: 1.35; letter-spacing: -.02em; }}
.lc-hero p {{ color: #3f4a5c; margin: .25rem 0 0; font-size: .86rem; line-height: 1.55; }}
.lc-miles {{ display: flex; align-items: flex-end; justify-content: space-between; margin-top: 14px; padding-top: 12px;
  border-top: 1px solid #cfe3f2; }}
.lc-miles .l {{ font-size: .8rem; color: {style.MUTED}; font-weight: 600; line-height: 1.5; }}
.lc-miles .l b {{ display: block; color: {style.BRAND}; font-size: .9rem; }}
.lc-miles .v {{ color: {style.BRAND}; font-size: 2.2rem; font-weight: 700; line-height: 1; letter-spacing: -.02em; }}
.lc-miles .v small {{ font-size: 1rem; font-weight: 600; margin-left: 2px; }}
/* 안내 상자 (완료·알림·링크 오류·빈 목록) */
.lc-box {{ border-radius: 14px; padding: 14px 16px; font-size: .9rem; line-height: 1.6; color: {style.INK}; background: {style.SOFT}; }}
.lc-box.warn {{ background: #fff3dc; color: #8a5300; }}
.lc-box.done {{ background: {style.SKY_SOFT}; padding: 16px 18px; }}
.lc-box.done .t {{ color: {style.BRAND}; font-weight: 700; font-size: 1rem; display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }}
.lc-box.done .t i {{ width: 22px; height: 22px; border-radius: 50%; background: {style.BRAND}; color: #fff; font-style: normal;
  font-size: .75rem; display: grid; place-items: center; flex: none; }}
.lc-box.err {{ background: linear-gradient(180deg, {style.SKY_SOFT} 0%, #f4f9fd 100%); border-radius: 16px; padding: 24px 20px; }}
.lc-box.err .t {{ font-size: 1.2rem; font-weight: 700; color: {style.BRAND}; margin-bottom: .35rem; }}
.lc-box.err p {{ margin: 0; color: #3f4a5c; }}
/* 라벨(회색 작은 글자) 위, 값(남색) 아래. 2열, .wide 는 한 줄 전체 */
.lc-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px 14px; }}
.lc-grid .wide {{ grid-column: 1 / -1; }}
.lc-grid .l {{ font-size: .74rem; color: {style.MUTED}; font-weight: 600; margin-bottom: 2px; }}
.lc-grid .v {{ font-size: .98rem; color: {style.BRAND}; font-weight: 700; line-height: 1.4; word-break: keep-all; }}
.lc-grid .v.none {{ color: {style.MUTED}; font-weight: 600; }}
.lc-grid .v i {{ width: .5em; height: .5em; border-radius: 50%; display: inline-block; margin-right: .35em; vertical-align: .12em; }}
/* 탑승권 머리: 결함 번호 + 유형 */
.lc-pass {{ display: flex; align-items: center; justify-content: space-between; }}
.lc-pass .no {{ font-size: .76rem; font-weight: 700; color: {style.SKY}; letter-spacing: .06em; }}
/* 탑승권 절취선: 점선 + 양 끝 반원 홈 (카드 안쪽 여백만큼 밖으로 뺀다) */
.lc-perf {{ position: relative; height: 0; border-top: 1.5px dashed #d3dae6; margin: 4px -17px; }}
.lc-perf::before, .lc-perf::after {{ content: ""; position: absolute; top: -10px; width: 18px; height: 18px; border-radius: 50%;
  background: #fff; border: 1px solid {style.LINE}; }}
.lc-perf::before {{ left: -10px; clip-path: inset(0 0 0 50%); }}
.lc-perf::after {{ right: -10px; clip-path: inset(0 50% 0 0); }}
[data-testid="stImage"] img {{ border-radius: 12px; max-height: 260px; object-fit: cover; }}
/* 좁은 화면에서도 [신고하기]·[넘기기] 를 한 줄에 (Streamlit 은 640px 아래에서 칸을 세로로 쌓는다) */
[data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; gap: 10px; }}
[data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{ min-width: 0; width: auto; flex: 1 1 0; }}
/* 버튼: [신고하기] 남색 알약(꽉 참, 테마·style.py), [넘기기] 흰 알약 테두리 */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {{ min-height: 50px; border-radius: 999px; }}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-secondary"] p {{ font-size: 1rem; font-weight: 700; }}
[data-testid="stBaseButton-secondary"] {{ background: #fff; border: 1px solid #c9d1de; color: {style.BRAND}; }}
[data-testid="stBaseButton-secondary"]:hover {{ background: {style.SKY_SOFT}; border-color: {style.SKY}; color: {style.BRAND}; }}
</style>
"""

# 로고 (ui/static/logo.svg). st.html 이 <svg> 를 지워서 그림 주소(data:)로 넣는다
_LOGO_SVG = (Path(__file__).resolve().parent.parent / "static" / "logo.svg").read_bytes()
_LOGO = f"<img alt='다보여' src='data:image/svg+xml;base64,{base64.b64encode(_LOGO_SVG).decode()}'>"


def _top_bar(right: str = "") -> None:
    """위쪽 바: 로고 + 오른쪽 작은 글자."""
    st.html(f"<div class='lc-bar'>{_LOGO}<span class='who'>{escape(right)}</span></div>")


def _grid(cells: list[tuple[str, str, bool]]) -> str:
    """라벨·값 2열 칸 HTML. cells = [(라벨, 값 HTML(escape 한 것), 한 줄 전체?)]."""
    return "<div class='lc-grid'>" + "".join(
        f"<div class='{'wide' if wide else ''}'><div class='l'>{escape(k)}</div><div class='v'>{v}</div></div>"
        for k, v, wide in cells
    ) + "</div>"


def _type_html(defect_type: str) -> str:
    c = style.TYPE_COLOR.get(defect_type, style.MUTED)
    return f"<i style='background:{c}'></i>{escape(style.TYPE_KO.get(defect_type, defect_type))}"


def _error_box(title: str, sub: str) -> None:
    st.html(f"<div class='lc-box err'><div class='t'>{escape(title)}</div><p>{escape(sub)}</p></div>")


st.html(_CSS)
token = st.query_params.get("t", "")

try:
    data = api.link_findings(token)
except api.ApiError as e:
    _top_bar("주행 결과 확인")
    if e.status == 410:
        _error_box("링크 유효 시간이 지났습니다", "다음 주행 후 새로 받으신 문자의 링크로 확인해 주세요.")
    elif e.status == 404:
        _error_box("올바르지 않은 링크입니다", "문자에 있는 링크를 다시 눌러 주세요.")
    else:
        _error_box("일시적으로 확인할 수 없습니다", f"잠시 후 다시 열어 주세요. ({e.detail})")
    style.footer()
    st.stop()

_top_bar("주행 결과 확인")

# 신고 누적 (사람 기준). 포상 기준·남은 건수는 서울시 공지마다 달라 안내하지 않는다 (PRD F8)
st.html(
    f"<div class='lc-hero'><h2>{escape(data['user_name'])}님의 주행 결과</h2>"
    "<p>블랙박스가 주행 중 찾은 도로 결함입니다. 사진과 위치를 확인하신 후 신고 여부를 선택해 주세요.</p>"
    "<div class='lc-miles'><div class='l'><b>다보여 신고</b>지금까지 신고하신 건수</div>"
    f"<div class='v'>{data['report_count']}<small>건</small></div></div></div>"
)

if msg := st.session_state.pop("link_flash", None):
    st.html(f"<div class='lc-box {'' if msg[0] == 'ok' else 'warn'}'>{escape(msg[1])}</div>")

# 방금 낸 신고서 요약 (원터치 신고의 결과). 새로 고침하면 이 화면에서는 사라지지만 신고서는 남아 있다
for rid in st.session_state.get("link_reports", []):
    try:
        r = api.link_get_report(token, rid)
    except api.ApiError:
        continue
    b = r["payload"]
    where = (naver_map.place(b["location"].get("address"))
             or f"위치 {b['location']['lat']:.5f}, {b['location']['lon']:.5f}")
    st.html(
        "<div class='lc-box done'><div class='t'><i>✓</i>신고가 접수되었습니다 (모의 제출)</div>"
        + _grid([
            ("유형", _type_html(b["defect_type"]), False),
            ("관할", escape(b["location"]["jurisdiction"]), False),
            ("위치", escape(where), True),
            ("신고자", escape(b["reporter"]), False),
            ("접수 시각", escape(api.when(b["reported_at"])), False),
        ])
        + "</div>"
    )

items = data["findings"]
style.section("확인할 결함", f"총 {len(items)}건")
if not items:
    st.html("<div class='lc-box'>확인할 결함이 없습니다. 모두 처리하셨거나 다른 분이 이미 신고하였습니다.</div>")

for i, d in enumerate(items, 1):
    with style.card(f"defect-{d['id']}"):
        photo = api.snapshot_bytes(d["snapshot_path"])
        if photo:
            st.image(photo, width="stretch")
        addr = naver_map.place(d.get("address"))
        st.html(f"<div class='lc-pass'><span class='no'>결함 {i:02d} / {len(items):02d}</span>"
                f"{style.badge('unreported')}</div>")
        st.html(_grid([
            ("유형", _type_html(d["defect_type"]), False),
            ("관할", escape(d["jurisdiction"]), False),
            ("위치", escape(addr) if addr else "<span class='none'>주소 확인 중</span>", True),
            ("AI 신뢰도", f"{d['confidence'] * 100:.0f}%", False),
            ("관측 차량", f"{d['vehicle_count']}대", False),
        ]))
        st.html("<div class='lc-perf'></div>")
        naver_map.mini(d["lat"], d["lon"], d["defect_type"])
        c1, c2 = st.columns(2)
        if c1.button("신고하기", key=f"report-{d['id']}", type="primary", width="stretch"):
            try:
                rep = api.link_report(token, d["id"])
                st.session_state.setdefault("link_reports", []).append(rep["report_id"])
            except api.ApiError as e:
                st.session_state["link_flash"] = ("warn", e.detail)
            st.rerun()
        if c2.button("넘기기", key=f"skip-{d['id']}", width="stretch"):
            try:
                api.link_skip(token, d["id"])
                st.session_state["link_flash"] = ("ok", "넘기기 처리되었습니다. 이 결함은 목록에서 빠집니다.")
            except api.ApiError as e:
                st.session_state["link_flash"] = ("warn", e.detail)
            st.rerun()

style.footer()
