"""신고 내역 (관리자). 시민이 문자 링크로 낸 신고서 목록과 시민별 신고 누적 수, 고른 신고서의 내용.

시민은 문자 링크 화면만 쓰므로 신고서는 관리자 쪽에서 본다. 제출은 모의 제출이다.
신고 누적은 사람 기준(reports.user_id)이다. 한 사람이 차를 여러 대 가질 수 있어서 차량으로 세지 않는다.
포상 기준(분기 몇 건)은 서울시 공지마다 달라 화면에 안내하지 않는다 (PRD F8).
5초마다 다시 불러 시연 중 새 신고가 바로 보이게 한다.
"""

import pandas as pd
import streamlit as st

import api
import naver_map

st.set_page_config(layout="wide")
st.title("신고 내역")


def _who(name: str, plates: list[str] | None) -> str:
    """'시민 A (12가 3456)'. 차량이 없으면 이름만."""
    return f"{name} ({', '.join(plates)})" if plates else name


@st.fragment(run_every=5)
def history() -> None:
    try:
        reps = api.reports()
        people = api.reporters()
    except api.ApiError as e:
        st.error(e.detail)
        return

    left, right = st.columns([2, 3])
    with left:
        st.subheader("시민별 신고 누적")
        st.dataframe(
            pd.DataFrame({"시민": [p["user_name"] for p in people],
                          "차량 번호": [", ".join(p["plates"] or []) or "-" for p in people],
                          "신고 누적": [f"{p['report_count']}건" for p in people]}),
            hide_index=True, width="stretch",
        )
    with right:
        st.subheader(f"신고 {len(reps)}건")
        if not reps:
            st.info("아직 신고가 없습니다. 시민이 문자 링크에서 [신고]를 누르면 여기에 나옵니다.")
            return
        st.dataframe(
            pd.DataFrame({"신고 일시": [api.short_when(str(r["created_at"])) for r in reps],
                          "신고자": [_who(r["user_name"], r["plates"]) for r in reps],
                          "위치": [naver_map.short_place(r.get("address")) for r in reps],
                          "유형": [api.TYPE_KO.get(r["defect_type"], r["defect_type"]) for r in reps],
                          "상태": [api.STATUS_KO.get(r["status"], r["status"]) for r in reps]}),
            hide_index=True, width="stretch",
        )

    ids = [r["id"] for r in reps]
    labels = {r["id"]: f"{api.short_when(str(r['created_at']))} · {r['user_name']} · "
                       f"{naver_map.short_place(r.get('address'))} {api.TYPE_KO.get(r['defect_type'], '')}"
              for r in reps}
    cur = st.session_state.get("rep_pick")
    rid = st.selectbox("신고서 보기", ids, index=ids.index(cur) if cur in ids else 0, format_func=labels.get)
    st.session_state["rep_pick"] = rid
    _paper(rid)


def _paper(rid: str) -> None:
    """신고서 내용 (reports.payload). 신고서에 적힌 그대로 보여 준다 — 신고자·위치는 신고한 때의 값."""
    try:
        r = api.get_report(rid)
    except api.ApiError as e:
        st.error(e.detail)
        return
    b = r["payload"]
    with st.container(border=True):
        st.success("✅ 제출됨 (모의)" if r["submission_status"] == "mock_submitted" else "신고서 작성됨")
        info, photo_col = st.columns([3, 2])
        rows = {
            "신고자": b["reporter"],
            "신고 일시": api.when(b["reported_at"]),
            "결함 유형": api.TYPE_KO.get(b["defect_type"], b["defect_type"]),
            "주소": naver_map.place(b["location"].get("address")) or "-",
            "관할": b["location"]["jurisdiction"],
            "AI 신뢰도": f"{b['confidence']:.2f}",
            "관측 차량 수": f"{b['vehicle_count']}대",
            "처음 관측": api.when(b["first_seen_at"]),
        }
        info.table(pd.DataFrame({"항목": list(rows), "내용": list(rows.values())}).set_index("항목"))
        with photo_col:
            photo = api.snapshot_bytes(b["snapshot"])
            if photo:
                st.image(photo, width="stretch")
            naver_map.mini(b["location"]["lat"], b["location"]["lon"], b["defect_type"], height=200)


history()
