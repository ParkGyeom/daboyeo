"""확인·신고 (시민, 휴대폰 폭). 주행이 끝나면 그 시민의 차량이 발견한 결함 목록(findings)이 나온다."""

import streamlit as st

import api
import naver_map

# 휴대폰 폭처럼 보이게 본문 폭을 줄인다
st.markdown("<style>.block-container{max-width:430px;padding-top:2rem}</style>", unsafe_allow_html=True)

st.title("다보여")
user_id = api.pick_user()

if msg := st.session_state.pop("flash", None):
    (st.success if msg[0] == "ok" else st.warning)(msg[1])

try:
    items = api.findings(user_id)
    notices = api.notices(user_id)
except api.ApiError as e:
    st.error(e.detail)
    st.stop()

# 화면 전체를 다시 그릴 때 본 알림 목록. 아래 감시 부분이 이것과 비교한다
st.session_state["notice_ids"] = (user_id, tuple((n["trip_id"], n["pending_count"]) for n in notices))
seen = st.session_state.setdefault("seen_trips", set())


@st.fragment(run_every="3s")
def _watch_notices(uid: str) -> None:
    """몇 초마다 서버를 확인하다가 주행 종료 알림이 바뀌면 화면 전체를 다시 그린다."""
    try:
        ids = (uid, tuple((n["trip_id"], n["pending_count"]) for n in api.notices(uid)))
    except api.ApiError:
        return
    if ids != st.session_state.get("notice_ids"):
        st.rerun(scope="app")


_watch_notices(user_id)

# 주행 종료 알림. 주행 중에는 알리지 않고, 블랙박스가 주행을 닫은 뒤에만 뜬다. 목록에 다 답하면 사라진다
fresh = [n for n in notices if n["trip_id"] not in seen]
if fresh:
    latest = fresh[0]
    with st.container(border=True):
        st.markdown("#### 🚗 주행이 끝났어요")
        st.markdown(f"이번 주행에서 도로 결함 **{latest['pending_count']}건**을 발견했어요.  \n"
                    "확인하고 버튼 한 번으로 신고할 수 있어요.")
        st.caption(f"주행 종료 {api.clock(latest['ended_at'])}")
        if st.button("확인하기", type="primary", width="stretch", key="notice-ok"):
            seen.update(n["trip_id"] for n in fresh)
            st.rerun()

st.subheader(f"발견한 결함 {len(items)}건")
if not items:
    st.caption("주행이 끝나면 발견한 결함이 여기에 나옵니다.")

for d in items:
    with st.container(border=True):
        photo = api.snapshot_bytes(d["snapshot_path"])
        if photo:
            st.image(photo, width="stretch")
        badge = "🔴" if d["risk_level"] == "urgent" else "🟡"
        st.markdown(f"**{api.TYPE_KO.get(d['defect_type'], d['defect_type'])}** · {badge} {api.RISK_KO[d['risk_level']]}")
        if addr := naver_map.place(d.get("address")):
            st.markdown(f"📍 {addr}")
        naver_map.mini(d["lat"], d["lon"], d["risk_level"], defect_type=d["defect_type"])
        st.caption(
            f"관할 {d['jurisdiction']}  \n"
            f"AI 신뢰도 {d['confidence']:.2f} · 관측 차량 {d['vehicle_count']}대 · "
            f"포상 {'대상' if d['reward_eligible_road'] else '대상 아님'}"
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
