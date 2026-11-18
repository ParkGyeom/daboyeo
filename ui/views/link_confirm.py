"""문자 링크 화면 (시민, 휴대폰 폭). 가입·로그인 없이 링크의 토큰으로 그 시민의 확인 목록을 연다.

[신고] 한 번으로 신고서가 만들어지고, 같은 화면에 신고서 요약이 나온다.
"""

import streamlit as st

import api
import naver_map

st.markdown("<style>.block-container{max-width:430px;padding-top:2rem}</style>", unsafe_allow_html=True)

token = st.query_params.get("t", "")

st.title("다보여")

try:
    data = api.link_findings(token)
except api.ApiError as e:
    if e.status == 410:
        st.warning("이 링크는 유효 시간이 지났어요. 다음 주행 뒤에 새 문자가 오면 그 링크로 확인해 주세요.")
    elif e.status == 404:
        st.error("올바르지 않은 링크예요. 문자에 있는 링크를 다시 눌러 주세요.")
    else:
        st.error(e.detail)
    st.stop()

if msg := st.session_state.pop("link_flash", None):
    (st.success if msg[0] == "ok" else st.warning)(msg[1])

st.caption(f"{data['user_name']}님의 주행 결과")

# 방금 낸 신고서 요약 (원터치 신고의 결과). 새로 고침하면 이 화면에서는 사라지지만 신고서는 남아 있다
for rid in st.session_state.get("link_reports", []):
    try:
        r = api.link_get_report(token, rid)
    except api.ApiError:
        continue
    b = r["payload"]
    with st.container(border=True):
        st.success("✅ 신고가 접수됐어요 (제출됨, 모의)")
        where = (naver_map.place(b["location"].get("address"))
                 or f"위치 {b['location']['lat']:.5f}, {b['location']['lon']:.5f}")
        st.caption(
            f"{api.TYPE_KO.get(b['defect_type'], b['defect_type'])} · {where} · {b['location']['jurisdiction']}  \n"
            f"신고자 {b['reporter']} · {api.when(b['reported_at'])}  \n"
            f"포상 {'대상' if r['reward_valid'] else '대상 아님'}"
        )

items = data["findings"]
st.subheader(f"확인할 결함 {len(items)}건")
if not items:
    st.caption("확인할 결함이 없어요. 모두 처리했거나 다른 분이 이미 신고했어요.")

for d in items:
    with st.container(border=True):
        photo = api.snapshot_bytes(d["snapshot_path"])
        if photo:
            st.image(photo, width="stretch")
        st.markdown(f"**{api.TYPE_KO.get(d['defect_type'], d['defect_type'])}**")
        if addr := naver_map.place(d.get("address")):
            st.markdown(f"📍 {addr}")
        naver_map.mini(d["lat"], d["lon"], d["defect_type"])
        st.caption(
            f"관할 {d['jurisdiction']}  \n"
            f"AI 신뢰도 {d['confidence']:.2f} · 관측 차량 {d['vehicle_count']}대 · "
            f"포상 {'대상' if d['reward_eligible_road'] else '대상 아님'}"
        )
        c1, c2 = st.columns(2)
        if c1.button("신고", key=f"report-{d['id']}", type="primary", width="stretch"):
            try:
                rep = api.link_report(token, d["id"])
                st.session_state.setdefault("link_reports", []).append(rep["report_id"])
            except api.ApiError as e:
                st.session_state["link_flash"] = ("warn", e.detail)
            st.rerun()
        if c2.button("넘기기", key=f"skip-{d['id']}", width="stretch"):
            try:
                api.link_skip(token, d["id"])
                st.session_state["link_flash"] = ("ok", "넘겼습니다.")
            except api.ApiError as e:
                st.session_state["link_flash"] = ("warn", e.detail)
            st.rerun()
