"""확인·신고 (시민, 휴대폰 폭). 시민을 고르면 그 시민의 기기가 발견한 결함이 나온다."""

import streamlit as st

import api

# 휴대폰 폭처럼 보이게 본문 폭을 줄인다
st.markdown("<style>.block-container{max-width:430px;padding-top:2rem}</style>", unsafe_allow_html=True)

st.title("다보여")
user_id = st.selectbox("시민", list(api.USERS), format_func=api.USERS.get, key="user_id")

if msg := st.session_state.pop("flash", None):
    (st.success if msg[0] == "ok" else st.warning)(msg[1])

try:
    items = api.pending(user_id)
    notices = api.notices(user_id)
except api.ApiError as e:
    st.error(e.detail)
    st.stop()

# 화면 전체를 다시 그릴 때 본 알림 목록. 아래 감시 부분이 이것과 비교한다
st.session_state["notice_ids"] = (user_id, tuple(n["id"] for n in notices))


@st.fragment(run_every="3s")
def _watch_notices(uid: int) -> None:
    """몇 초마다 서버를 확인하다가 새 주행 종료 알림이 오면 화면 전체를 다시 그린다."""
    try:
        ids = (uid, tuple(n["id"] for n in api.notices(uid)))
    except api.ApiError:
        return
    if ids != st.session_state.get("notice_ids"):
        st.rerun(scope="app")


_watch_notices(user_id)

# 주행 종료 알림. 주행 중에는 알리지 않고, 블랙박스가 주행을 끝냈다고 알린 뒤에만 뜬다
if notices:
    latest = notices[0]
    n = latest["defect_count"]
    with st.container(border=True):
        st.markdown("#### 🚗 주행이 끝났어요")
        if n:
            st.markdown(f"이번 주행에서 도로 결함 **{n}건**을 발견했어요.  \n확인하고 버튼 한 번으로 신고할 수 있어요.")
        else:
            st.markdown("이번 주행에서 새로 발견한 도로 결함은 없어요.")
        st.caption(f"주행 종료 {api.clock(latest['ended_at'])}")
        if st.button("확인하기", type="primary", width="stretch", key="notice-ok"):
            try:
                api.notices_seen(user_id)
            except api.ApiError as e:
                st.session_state["flash"] = ("warn", e.detail)
            st.rerun()

st.subheader(f"발견한 결함 {len(items)}건")
if not items:
    st.caption("주행 중 발견한 결함이 여기에 나옵니다.")

for d in items:
    with st.container(border=True):
        photo = api.snapshot_bytes(d["best_snapshot_path"])
        if photo:
            st.image(photo, width="stretch")
        badge = "🔴" if d["severity"] == "빠른 보수" else "🟡"
        st.markdown(f"**{api.TYPE_KO.get(d['type'], d['type'])}** · {badge} {d['severity']}")
        st.caption(
            f"위치 {d['lat']:.5f}, {d['lon']:.5f} · {d['jurisdiction']}  \n"
            f"AI 신뢰도 {d['max_confidence']:.2f} · 관측 기기 {d['device_count']}대 · "
            f"포상 {'대상' if d['reward_eligible'] else '대상 아님'}"
        )
        c1, c2 = st.columns(2)
        if c1.button("신고", key=f"report-{d['id']}", type="primary", width="stretch"):
            try:
                rep = api.report(d["id"], user_id)
                st.session_state["last_report_id"] = rep["report_id"]
                st.switch_page("views/report.py")
            except api.ApiError as e:
                st.session_state["flash"] = ("warn", e.detail)
                st.rerun()
        if c2.button("넘기기", key=f"skip-{d['id']}", width="stretch"):
            try:
                api.skip(d["id"], user_id)
                st.session_state["flash"] = ("ok", "넘겼습니다.")
            except api.ApiError as e:
                st.session_state["flash"] = ("warn", e.detail)
            st.rerun()
