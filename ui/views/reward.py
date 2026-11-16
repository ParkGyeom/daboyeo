"""포상 현황 (시민). 이번 분기에 낸 포상 대상 신고 수와 5건까지 남은 수 (docs/DESIGN.md 7장)."""

from datetime import datetime

import streamlit as st

import api

st.markdown("<style>.block-container{max-width:430px;padding-top:2rem}</style>", unsafe_allow_html=True)
st.title("포상 현황")

user_id = st.selectbox("시민", list(api.USERS), format_func=api.USERS.get, key="user_id")
try:
    r = api.reward(user_id)
except api.ApiError as e:
    st.error(e.detail)
    st.stop()

since = datetime.fromisoformat(r["since"])
quarter = (since.month - 1) // 3 + 1
st.caption(f"{since.year}년 {quarter}분기 ({since:%m월 %d일}부터)")

c1, c2 = st.columns(2)
c1.metric("유효 신고", f"{r['count']}건")
c2.metric("포상까지", "달성" if r["remaining"] == 0 else f"{r['remaining']}건 남음")
st.progress(min(1.0, r["count"] / r["target"]), text=f"{r['count']} / {r['target']}건")

if r["remaining"] == 0:
    st.success("이번 분기 포상 기준을 채웠습니다.")
else:
    st.info(f"서울시 관리 도로의 결함을 {r['remaining']}건 더 신고하면 포상 대상이 됩니다.")
st.caption("포상 대상: 서울 범위 안에서 발견해 신고한 결함 (서울시 관리 도로 목록을 구하면 그것으로 바꾼다)")
