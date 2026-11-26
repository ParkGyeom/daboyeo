"""신고 내역 (관리자). 시민이 문자 링크로 낸 신고서 목록과 시민별 신고 실적, 고른 신고서의 내용.

시민은 문자 링크 화면만 쓰므로 신고서는 관리자 쪽에서 본다. 제출은 모의 제출이다.
신고 누적은 사람 기준(reports.user_id)이다. 한 사람이 차를 여러 대 가질 수 있어서 차량으로 세지 않는다.
포상 기준(분기 몇 건)은 서울시 공지마다 달라 화면에 안내하지 않는다 (PRD F8).
5초마다 다시 불러 시연 중 새 신고가 바로 보이게 한다.
모양: 예약 누리집처럼 얇은 선 목록·알약 배지, 신고서는 확인증(e-티켓)처럼 남색 머리 띠 + '라벨 위 / 값 아래' 칸.
"""

from html import escape

import streamlit as st

import api
import naver_map
import style

st.set_page_config(layout="wide")
style.page_header("신고 내역", "시민이 문자 링크로 제출한 도로 결함 신고서를 확인합니다. 제출은 모의 제출입니다.",
                  kicker="관리자")

# 이 페이지 조각 모양 (style.py 로 옮길 것: .dby-list·.dby-tbl 은 다른 화면 목록에도 쓸 수 있다)
st.html(f"""<style>
.dby-list {{ list-style: none; margin: 0 !important; padding: 0 !important; }}
.dby-list li {{ margin: 0 !important; display: grid; grid-template-columns: 28px 1fr auto; gap: 12px; align-items: center;
  padding: 12px 2px; border-bottom: 1px solid {style.LINE}; }}
.dby-list li:last-child {{ border-bottom: 0; }}
.dby-list .no {{ width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
  font-size: .78rem; font-weight: 700; background: {style.SOFT}; color: {style.MUTED}; }}
.dby-list .top .no {{ background: {style.SKY_SOFT}; color: {style.BRAND}; }}
.dby-list .nm {{ font-weight: 700; color: {style.BRAND}; }}
.dby-list .pl {{ color: {style.MUTED}; font-size: .8rem; }}
.dby-list .cnt {{ font-weight: 700; color: {style.BRAND}; font-size: 1.1rem; }}
.dby-list .cnt small {{ color: {style.MUTED}; font-weight: 600; font-size: .78rem; margin-left: 2px; }}
.dby-list .zero .nm, .dby-list .zero .cnt {{ color: #9aa3b2; }}
.dby-tbl-wrap {{ max-height: 380px; overflow-y: auto; }}
.dby-tbl {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .88rem; }}
.dby-tbl th {{ position: sticky; top: 0; background: {style.SOFT}; color: {style.MUTED}; font-weight: 600;
  padding: 10px 12px; text-align: left; white-space: nowrap; }}
.dby-tbl th:first-child {{ border-radius: 10px 0 0 10px; }}
.dby-tbl th:last-child {{ border-radius: 0 10px 10px 0; }}
.dby-tbl td {{ border-bottom: 1px solid {style.LINE}; padding: 12px 12px; color: {style.INK}; white-space: nowrap; }}
.dby-tbl td.who {{ font-weight: 700; color: {style.BRAND}; }}
.dby-tbl td.muted {{ color: {style.MUTED}; }}
.st-key-dby-ticket {{ border: 1px solid {style.LINE}; border-radius: 16px; overflow: hidden; gap: 0;
  box-shadow: 0 2px 12px rgba(5,23,102,.05); background: #fff; }}
.dby-tk-head {{ background: {style.BRAND}; color: #fff; padding: 20px 28px; display: flex; justify-content: space-between;
  align-items: center; gap: 16px; }}
.dby-tk-head .k {{ color: {style.SKY}; font-size: .8rem; font-weight: 600; }}
.dby-tk-head h2 {{ color: #fff; font-size: 1.45rem; font-weight: 700; margin: .1rem 0 0; padding: 0; }}
.dby-tk-head .no {{ text-align: right; font-size: .8rem; color: #c8d3ef; }}
.dby-tk-head .no b {{ display: block; color: #fff; font-size: 1.25rem; font-weight: 700; letter-spacing: .04em; }}
.dby-tk-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px 24px; padding: 22px 28px 20px; }}
.dby-tk-grid .w2 {{ grid-column: span 2; }}
.dby-tk-grid .l {{ color: {style.MUTED}; font-size: .78rem; font-weight: 600; margin-bottom: .25rem; }}
.dby-tk-grid .v {{ color: {style.INK}; font-size: 1rem; font-weight: 600; }}
.dby-tk-cut {{ position: relative; border-top: 2px dashed {style.LINE}; margin: 0 28px; }}
.dby-tk-cut::before, .dby-tk-cut::after {{ content: ""; position: absolute; top: -12px; width: 22px; height: 22px;
  border-radius: 50%; background: #fff; border: 1px solid {style.LINE}; }}
.dby-tk-cut::before {{ left: -40px; }}
.dby-tk-cut::after {{ right: -40px; }}
.st-key-dby-ticket-att {{ padding: 18px 28px 6px; }}
.dby-tk-att {{ color: {style.MUTED}; font-size: .78rem; font-weight: 600; margin-bottom: .1rem; }}
.dby-tk-foot {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin: 0 28px;
  padding: 14px 0 20px; border-top: 1px solid {style.LINE}; color: {style.MUTED}; font-size: .86rem; }}
.dby-tk-foot b {{ color: {style.BRAND}; font-size: 1rem; margin: 0 .35em; }}
</style>""")


def _people_list(people: list[dict]) -> None:
    """시민별 신고 실적 (얇은 선 목록). 많이 낸 순 (서버 정렬 그대로), 0건은 흐리게."""
    rows = []
    for i, p in enumerate(people, 1):
        n = p["report_count"]
        cls = "top" if n and i <= 3 else ("zero" if not n else "")
        rows.append(
            f"<li class='{cls}'><span class='no'>{i}</span>"
            f"<div><div class='nm'>{escape(p['user_name'])}</div>"
            f"<div class='pl'>{escape(', '.join(p['plates'] or []) or '차량 없음')}</div></div>"
            f"<div class='cnt'>{n}<small>건</small></div></li>"
        )
    st.html(f"<ul class='dby-list'>{''.join(rows)}</ul>")


def _reports_table(reps: list[dict]) -> None:
    """신고 접수 현황 표 (회색 둥근 머리 + 얇은 선 줄). 번호는 접수 순번이라 최근 것이 크다."""
    n = len(reps)
    rows = "".join(
        f"<tr><td class='muted'>{n - i}</td><td>{escape(api.short_when(str(r['created_at'])))}</td>"
        f"<td class='who'>{escape(r['user_name'])}</td>"
        f"<td class='muted'>{escape(', '.join(r['plates'] or []) or '-')}</td>"
        f"<td>{escape(naver_map.short_place(r.get('address')))}</td>"
        f"<td>{style.type_chip(r['defect_type'])}</td><td>{style.badge(r['status'])}</td></tr>"
        for i, r in enumerate(reps)
    )
    st.html("<div class='dby-tbl-wrap'><table class='dby-tbl'><thead><tr><th>번호</th><th>신고 일시</th>"
            "<th>신고인</th><th>차량 번호</th><th>위치</th><th>결함 유형</th><th>처리 상태</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>")


@st.fragment(run_every=5)
def history() -> None:
    try:
        reps = api.reports()
        people = api.reporters()
    except api.ApiError as e:
        st.error(e.detail)
        return

    style.stat_cards([
        ("총 신고", len(reps), style.BRAND, "시민이 제출한 신고서", "건"),
        ("신고 시민", sum(1 for p in people if p["report_count"]), style.SKY, f"전체 시민 {len(people)}명 중", "명"),
        ("신고 결함", len({r["defect_id"] for r in reps}), style.SKY, "같은 결함은 1건으로 계산", "건"),
        ("제출 완료", sum(1 for r in reps if r["submission_status"] == "mock_submitted"), style.SKY, "모의 제출", "건"),
    ])

    left, right = st.columns([1, 2.4], gap="large")
    with left, style.card("rep-people"):
        style.section("시민별 신고 실적")
        _people_list(people)
    with right, style.card("rep-list"):
        style.section("신고 접수 현황", f"최근 순 · 총 {len(reps)}건")
        if not reps:
            st.info("접수된 신고가 없습니다. 시민이 문자 링크에서 [신고]를 누르면 이 목록에 표시됩니다.")
            return
        _reports_table(reps)

    ids = [r["id"] for r in reps]
    n = len(reps)
    labels = {r["id"]: f"{n - i}. {api.short_when(str(r['created_at']))} · {r['user_name']} · "
                       f"{naver_map.short_place(r.get('address'))} {api.TYPE_KO.get(r['defect_type'], '')}"
              for i, r in enumerate(reps)}
    cur = st.session_state.get("rep_pick")
    st.html("<div style='height:1rem'></div>")
    style.section("신고서 상세", "목록에서 고른 신고서를 확인증 형태로 보여 줍니다")
    rid = st.selectbox("조회할 신고서", ids, index=ids.index(cur) if cur in ids else 0, format_func=labels.get)
    st.session_state["rep_pick"] = rid
    _ticket(rid)


def _ticket(rid: str) -> None:
    """신고서 내용 (reports.payload)을 확인증처럼. 신고서에 적힌 그대로 — 신고자·위치는 신고한 때의 값."""
    try:
        r = api.get_report(rid)
    except api.ApiError as e:
        st.error(e.detail)
        return
    b = r["payload"]
    submitted = r["submission_status"] == "mock_submitted"
    state = (f"<span class='dby-badge' style='color:{style.BRAND};background:{style.SKY_SOFT}'>제출 완료 (모의)</span>"
             if submitted else f"<span class='dby-badge' style='color:{style.MUTED};background:{style.SOFT}'>작성</span>")
    cells = [
        ("신고인", escape(b["reporter"]), ""),
        ("신고 일시", escape(api.when(b["reported_at"])), ""),
        ("결함 유형", style.type_chip(b["defect_type"]), ""),
        ("처리 상태", state, ""),
        ("발생 위치", escape(naver_map.place(b["location"].get("address")) or "-"), "w2"),
        ("관할 기관", escape(b["location"]["jurisdiction"]), ""),
        ("최초 관측", escape(api.when(b["first_seen_at"])), ""),
        ("AI 판독 신뢰도", f"{b['confidence']:.2f}", ""),
        ("관측 차량 수", f"{b['vehicle_count']}대", ""),
        ("신고 분기", escape(str(r["quarter"])), ""),
    ]
    grid = "".join(f"<div class='{c}'><div class='l'>{k}</div><div class='v'>{v}</div></div>" for k, v, c in cells)
    with st.container(key="dby-ticket"):
        st.html(
            "<div class='dby-tk-head'><div><div class='k'>다보여 도로 결함 확인·신고 (모의)</div>"
            "<h2>도로 결함 신고서</h2></div>"
            f"<div class='no'>접수 번호<b>R-{escape(api.short(str(r['id'])).upper())}</b></div></div>"
            f"<div class='dby-tk-grid'>{grid}</div><div class='dby-tk-cut'></div>"
        )
        # 사진 | 지도. 사진이 거의 정사각형이라 옆으로 둔다
        with st.container(key="dby-ticket-att"):
            photo_col, map_col = st.columns(2, gap="medium")
            with photo_col:
                st.html("<div class='dby-tk-att'>첨부 1 · 현장 사진</div>")
                photo = api.snapshot_bytes(b["snapshot"])
                if photo:
                    st.image(photo, width="stretch")
                else:
                    st.caption("사진 없음")
            with map_col:
                st.html("<div class='dby-tk-att'>첨부 2 · 발생 위치</div>")
                naver_map.mini(b["location"]["lat"], b["location"]["lon"], b["defect_type"], height=330)
        st.html(
            "<div class='dby-tk-foot'><span>위와 같이 도로 결함을 신고합니다. · "
            f"{escape(api.when(b['reported_at']))}</span>"
            f"<span>신고인<b>{escape(b['reporter'])}</b>"
            f"<span class='dby-badge' style='color:#c42b1c;background:#fdeeec'>모의</span></span></div>"
        )


history()
