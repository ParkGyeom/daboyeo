"""신고서 (시민). 그 시민의 신고서를 고르면 내용을 보여준다. 제출은 모의 제출이다."""

import streamlit as st

import api
import naver_map

st.markdown("<style>.block-container{max-width:430px;padding-top:2rem}</style>", unsafe_allow_html=True)
st.title("신고서")

user_id = api.pick_user()
try:
    reps = api.user_reports(user_id)
except api.ApiError as e:
    st.error(e.detail)
    st.stop()
if not reps:
    st.info("아직 신고서가 없습니다.")
    st.stop()

ids = [r["id"] for r in reps]
last = st.session_state.get("last_report_id")
labels = {r["id"]: f"{api.clock(r['created_at'])} · {api.TYPE_KO.get(r['payload']['defect_type'], '')} "
                    f"({api.short(r['id'])})" for r in reps}
rid = st.selectbox("신고서", ids, index=ids.index(last) if last in ids else 0, format_func=labels.get)
try:
    r = api.get_report(rid)
except api.ApiError as e:
    st.error(e.detail)
    st.stop()

b = r["payload"]
st.success("✅ 제출됨 (모의)" if r["submission_status"] == "mock_submitted" else "신고서 작성됨")
photo = api.snapshot_bytes(b["snapshot"])
if photo:
    st.image(photo, width="stretch")
rows = {
    "신고자": b["reporter"],
    "신고 일시": b["reported_at"].replace("T", " "),
    "결함 유형": api.TYPE_KO.get(b["defect_type"], b["defect_type"]),
    "위험도": api.RISK_KO.get(b["risk_level"], b["risk_level"]),
    "관할": b["location"]["jurisdiction"],
    "AI 신뢰도": f"{b['confidence']:.2f}",
    "관측 차량 수": f"{b['vehicle_count']}대",
    "처음 관측": b["first_seen_at"].replace("T", " "),
    "포상 대상": "예" if r["reward_valid"] else "아니오",
    "분기": r["quarter"],
}
st.markdown("**위치**")
naver_map.mini(b["location"]["lat"], b["location"]["lon"], b["risk_level"], height=200)
st.table({"항목": list(rows), "내용": list(rows.values())})
if st.button("확인·신고로 돌아가기"):
    st.switch_page("views/confirm.py")
