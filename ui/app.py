"""다보여 화면. 실행: .venv/bin/streamlit run ui/app.py

서버(.venv/bin/python -m uvicorn server.main:app --port 8000)가 먼저 떠 있어야 한다.
서버 주소를 바꾸려면 SERVER_URL 환경 변수를 준다.
"""

import streamlit as st

st.set_page_config(page_title="다보여", page_icon="🛣️", layout="centered")

# 문자 링크(/?t=토큰)로 들어오면 가입·로그인 없이 그 시민의 확인 화면 하나만 보여 준다 (메뉴 없음)
if st.query_params.get("t"):
    st.navigation([st.Page("views/link_confirm.py", title="확인·신고", icon="📱")], position="hidden").run()
    st.stop()

pages = st.navigation(
    {
        "시민": [
            st.Page("views/confirm.py", title="확인·신고", icon="📱", default=True),
            st.Page("views/report.py", title="신고서", icon="📄"),
            st.Page("views/reward.py", title="포상 현황", icon="🏅"),
        ],
        "시연": [
            st.Page("views/inbox.py", title="메시지함 (모의)", icon="💬"),
        ],
        "관리자": [
            st.Page("views/dashboard.py", title="지도 대시보드", icon="🗺️"),
            st.Page("views/blackbox.py", title="블랙박스 화면", icon="🎥"),
        ],
    }
)
pages.run()
