"""메시지함 (시연용, 모의). 서버가 보낸 것으로 친 문자를 휴대폰 화면처럼 보여 준다. 실제 문자는 보내지 않는다.

위에서 받는 시민을 알약으로 고르면 가운데 휴대폰 1대에 그 시민의 문자와 '링크 열기'가 나온다.
처음에는 가장 최근 문자를 받은 시민. 고른 시민은 세션에 남아 3초 갱신에도 유지된다.
오른쪽은 전체 문자 목록 (시민·시각·열어 봄, 얇은 선 목록).
"""

from html import escape

import streamlit as st

import api
import style

st.set_page_config(layout="wide")
style.page_header("메시지함", "시민에게 보내는 확인 요청 문자를 시연용으로 보여 줍니다. 실제 문자는 보내지 않습니다. "
                  "[링크 열기]를 누르면 그 시민의 확인·신고 화면이 새 창으로 열립니다.", kicker="시연 · 모의 문자")

# 휴대폰 틀·말풍선·문자 목록. 키를 준 st.container 의 .st-key-<키> 를 고른다 (style.py 로 옮길 것)
st.html(f"""<style>
.st-key-dby-phone {{
  border: 9px solid #0b1230; border-radius: 34px; background: #fff; padding: 0 14px 18px;
  max-width: 390px; margin: .4rem auto 0; min-height: 600px; gap: .55rem;
  box-shadow: 0 8px 28px rgba(5,23,102,.10);
}}
.dby-phone-top {{ margin: 0 -14px; padding: 10px 18px 14px; background: {style.SKY_SOFT}; border-radius: 25px 25px 0 0; }}
.dby-phone-top .bar {{ display: flex; justify-content: center; }}
.dby-phone-top .bar i {{ width: 90px; height: 20px; border-radius: 12px; background: #0b1230; display: block; }}
.dby-phone-top .who {{ text-align: center; margin-top: 10px; }}
.dby-phone-top .nm {{ font-weight: 700; color: {style.BRAND}; font-size: 1.05rem; }}
.dby-phone-top .ph {{ color: {style.MUTED}; font-size: .8rem; }}
.dby-day {{ text-align: center; color: {style.MUTED}; font-size: .74rem; margin: .5rem 0 0; }}
.dby-from {{ color: {style.MUTED}; font-size: .76rem; font-weight: 600; margin: 0 0 -6px 4px; }}
[class*="st-key-dby-sms-"] {{
  background: {style.SOFT}; border-radius: 18px 18px 18px 6px; padding: 13px 15px; max-width: 95%;
  font-size: .86rem; line-height: 1.55; gap: .2rem;
}}
[class*="st-key-dby-sms-"] p {{ font-size: .86rem; margin-bottom: .35rem; word-break: break-all; }}
.dby-meta {{ color: {style.MUTED}; font-size: .74rem; margin: -2px 0 0 6px; }}
.dby-msgs {{ list-style: none; margin: 0 !important; padding: 0 !important; }}
.dby-msgs li {{ margin: 0 !important; display: grid; grid-template-columns: 1fr auto; gap: 4px 10px; align-items: center;
  padding: 13px 10px; border-bottom: 1px solid {style.LINE}; }}
.dby-msgs li:last-child {{ border-bottom: 0; }}
.dby-msgs li.on {{ background: {style.SKY_SOFT}; border-radius: 10px; border-bottom-color: transparent; }}
.dby-msgs .nm {{ font-weight: 700; color: {style.BRAND}; }}
.dby-msgs .sub {{ color: {style.MUTED}; font-size: .8rem; grid-column: 1 / -1; }}
</style>""")


def _seen(m: dict) -> str:
    if m["opened_at"]:
        return f"<span class='dby-badge' style='color:{style.BRAND};background:{style.SKY_SOFT}'>열어 봄</span>"
    return "<span class='dby-badge' style='color:#c42b1c;background:#fdeeec'>안 읽음</span>"


def _phone(name: str, items: list[dict]) -> None:
    """고른 시민의 휴대폰 1대. 문자는 최근 것부터 (api.inbox 순서 그대로)."""
    with st.container(key="dby-phone"):
        st.html(f"<div class='dby-phone-top'><div class='bar'><i></i></div><div class='who'>"
                f"<div class='nm'>{escape(name)}</div><div class='ph'>{escape(items[0]['to_phone'])}</div></div></div>")
        st.html("<div class='dby-day'>문자 메시지</div>")
        for m in items:
            st.html("<div class='dby-from'>다보여</div>")
            with st.container(key=f"dby-sms-{m['id']}"):
                st.markdown(api.sms_markdown(m["body"], m["url"]))
            st.html(f"<div class='dby-meta'>{escape(api.clock(m['created_at']))} · {_seen(m)}</div>")
            st.link_button("링크 열기", m["url"], type="primary", width="stretch")


def _all_list(msgs: list[dict], pick: str) -> None:
    """전체 문자 목록 (최근 것부터). 고른 시민의 문자는 하늘색 바탕."""
    rows = "".join(
        f"<li class='{'on' if m['user_name'] == pick else ''}'><span class='nm'>{escape(m['user_name'])}</span>"
        f"{_seen(m)}<span class='sub'>{escape(api.when(m['created_at']))} · {escape(m['to_phone'])}</span></li>"
        for m in msgs
    )
    st.html(f"<ul class='dby-msgs'>{rows}</ul>")


@st.fragment(run_every="3s")
def _inbox() -> None:
    try:
        msgs = api.inbox()
    except api.ApiError as e:
        st.error(e.detail)
        return
    if not msgs:
        st.info("받은 문자가 없습니다. 가상 블랙박스가 주행을 끝내고 확인할 결함이 있으면 여기에 도착합니다.")
        return
    by_user: dict[str, list[dict]] = {}
    for m in msgs:
        by_user.setdefault(m["user_name"], []).append(m)
    names = sorted(by_user)

    style.stat_cards([
        ("보낸 문자", len(msgs), style.BRAND, "모의 발송 · 실제로 보내지 않음", "건"),
        ("받은 시민", len(by_user), style.SKY, "주행을 끝낸 차량의 주인", "명"),
        ("열어 본 문자", sum(1 for m in msgs if m["opened_at"]), style.SKY, "링크를 누른 문자", "건"),
    ])

    # 고른 시민: 처음엔 가장 최근 문자를 받은 시민. 알약을 다시 눌러 비워도 마지막에 고른 시민을 그대로 보여 준다
    if st.session_state.get("inbox_pick") not in names:
        last = st.session_state.get("inbox_last")
        st.session_state["inbox_pick"] = last if last in names else msgs[0]["user_name"]
    phone_col, list_col = st.columns([3, 2], gap="large")
    with phone_col:
        st.pills("받는 시민", names, key="inbox_pick",
                 format_func=lambda n: n + (" ●" if any(not m["opened_at"] for m in by_user[n]) else ""))
        pick = st.session_state.get("inbox_pick") or st.session_state.get("inbox_last") or msgs[0]["user_name"]
        st.session_state["inbox_last"] = pick
        _phone(pick, by_user[pick])
    with list_col, style.card("inbox-list"):
        style.section("전체 문자", f"최근 순 · {len(msgs)}건")
        _all_list(msgs, pick)


_inbox()
