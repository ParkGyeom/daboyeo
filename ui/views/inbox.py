"""메시지함 (시연용, 모의). 서버가 보낸 것으로 친 문자를 시민별로 보여 준다. 실제 문자는 보내지 않는다."""

import streamlit as st

import api

st.title("메시지함 (모의)")
st.caption("실제 문자는 보내지 않습니다. 서버가 보낼 문자를 저장하면 여기에 문자처럼 보입니다. 링크를 누르면 시민의 확인 화면이 새 탭으로 열립니다.")


@st.fragment(run_every="3s")
def _inbox() -> None:
    try:
        msgs = api.inbox()
    except api.ApiError as e:
        st.error(e.detail)
        return
    if not msgs:
        st.info("아직 문자가 없습니다. 가상 블랙박스가 주행을 끝내고 확인할 결함이 있으면 여기에 도착합니다.")
        return
    by_user: dict[str, list[dict]] = {}
    for m in msgs:
        by_user.setdefault(m["user_name"], []).append(m)
    cols = st.columns(len(by_user))
    for col, (name, items) in zip(cols, sorted(by_user.items())):
        with col, st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"받는 번호 {items[0]['to_phone']}")
            for m in items:
                with st.container(border=True):
                    st.caption(f"{api.clock(m['created_at'])}" + (" · 열어 봄" if m["opened_at"] else ""))
                    st.markdown(api.sms_markdown(m["body"], m["url"]))
                    st.link_button("링크 열기", m["url"], width="stretch")


_inbox()
