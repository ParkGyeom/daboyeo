"""다보여 화면. 실행: .venv/bin/streamlit run ui/app.py

서버(.venv/bin/python -m uvicorn server.main:app --port 8000)가 먼저 떠 있어야 한다.
서버 주소를 바꾸려면 SERVER_URL 환경 변수를 준다.
공통 모양은 ui/style.py 와 .streamlit/config.toml 의 [theme].
"""

from pathlib import Path

import streamlit as st

import style

st.set_page_config(page_title="다보여", page_icon="🛣️", layout="centered")

# 문자 링크(/?t=토큰)로 들어오면 가입·로그인 없이 그 시민의 확인 화면 하나만 보여 준다 (메뉴 없음)
if st.query_params.get("t"):
    style.apply(mobile=True)
    st.navigation([st.Page("views/link_confirm.py", title="확인·신고")], position="hidden").run()
    st.stop()

style.apply()
st.logo(str(Path(__file__).parent / "static" / "logo.svg"), size="large")

# 시민이 받는 것은 문자 하나와 그 링크뿐이라 시민 메뉴는 없다. 시연에서는 메시지함(모의)에서 링크를 누른다
# 메뉴는 위쪽 가로 줄 (웹 서비스처럼). 묶음을 두면 펼침 메뉴가 되어 한 번 더 눌러야 해서 한 줄로 둔다
pages = st.navigation(
    [
        st.Page("views/dashboard.py", title="지도 대시보드"),
        st.Page("views/reports.py", title="신고 내역"),
        st.Page("views/blackbox.py", title="블랙박스 화면"),
        st.Page("views/inbox.py", title="메시지함 (모의)", default=True),
    ],
    position="top",
)
pages.run()
style.footer()
