"""신고서 (시민). 방금 만든 신고서를 보여준다. 제출은 모의 제출이다."""

import streamlit as st

import api

st.markdown("<style>.block-container{max-width:430px;padding-top:2rem}</style>", unsafe_allow_html=True)
st.title("신고서")

report_id = st.number_input("신고서 번호", min_value=1, step=1, value=st.session_state.get("last_report_id", 1))
try:
    r = api.get_report(int(report_id))
except api.ApiError as e:
    st.info("아직 신고서가 없습니다." if e.status == 404 else e.detail)
    st.stop()

b = r["body"]
st.success(f"✅ {b['submission']}")
photo = api.snapshot_bytes(b["snapshot"])
if photo:
    st.image(photo, width="stretch")
rows = {
    "신고자": b["reporter"],
    "신고 일시": b["reported_at"].replace("T", " "),
    "결함 유형": api.TYPE_KO.get(b["defect_type"], b["defect_type"]),
    "위험도": b["severity"],
    "위치": f"{b['location']['lat']:.6f}, {b['location']['lon']:.6f}",
    "관할": b["location"]["jurisdiction"],
    "AI 신뢰도": f"{b['confidence']:.2f}",
    "관측 기기 수": f"{b['device_count']}대",
    "처음 관측": b["first_seen"].replace("T", " "),
    "포상 대상": "예" if b["reward_eligible"] else "아니오",
}
st.table({"항목": list(rows), "내용": list(rows.values())})
if st.button("확인·신고로 돌아가기"):
    st.switch_page("views/confirm.py")
