"""블랙박스 화면 (관리자·시연용). 가상 블랙박스가 지금 보고 있는 프레임과 탐지 박스.

가상 블랙박스(edge/blackbox/live.py)가 server_data/live/ 에 쓰는 파일을 0.5초마다 다시 읽는다.
기기 여러 대를 나란히 보여준다.
"""

import json
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edge.blackbox.live import COLORS, LABELS, LIVE_DIR  # noqa: E402

STALE_S = 5  # 이 시간 동안 새 프레임이 없으면 멈춘 것으로 본다

st.markdown("<style>.block-container{max-width:1400px}</style>", unsafe_allow_html=True)
st.title("블랙박스 화면")
st.caption("가상 블랙박스가 분석 중인 프레임. 굵은 박스는 서버로 보내는 결함, 얇은 박스는 위치만 찾는 시설물(가드레일·표지판)")


def legend() -> str:
    def hexcolor(bgr):
        b, g, r = bgr
        return f"#{r:02x}{g:02x}{b:02x}"

    return " ".join(
        f"<span style='display:inline-block;width:.8em;height:.8em;background:{hexcolor(COLORS[k])};"
        f"margin:0 .3em 0 .8em;border-radius:2px'></span>{v}" for k, v in LABELS.items()
    )


st.markdown(legend(), unsafe_allow_html=True)


def read_devices() -> list[tuple[dict, bytes | None]]:
    out = []
    for js in sorted(LIVE_DIR.glob("*.json")):
        try:
            status = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # 쓰는 중
        jpg = js.with_suffix(".jpg")
        try:
            img = jpg.read_bytes()
        except OSError:
            img = None
        out.append((status, img))
    return out


@st.fragment(run_every=0.5)
def view() -> None:
    devices = read_devices()
    if not devices:
        st.info("돌고 있는 블랙박스가 없습니다. 예: "
                "`uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --model ai/models/detector_v3.onnx`")
        return
    cols = st.columns(min(3, len(devices)))
    now = time.time()
    for col, (s, img) in zip(cols * (len(devices) // len(cols) + 1), devices):
        with col:
            if s.get("ended"):
                state = "⏹ 주행 끝"
            elif now - s.get("updated_at", 0) > STALE_S:
                state = "⏸ 멈춤"
            else:
                state = "🔴 주행 중"
            st.markdown(f"#### {s['device_id']}  \n{state}")
            if img:
                st.image(img, width="stretch")
            t = s.get("t")
            boxes = s.get("boxes") or []
            found = ", ".join(f"{LABELS.get(b['type'], b['type'])} {b['confidence']:.2f}" for b in boxes) or "없음"
            st.caption(f"{s.get('source', '')}" + (f" · {t:.1f}초" if isinstance(t, (int, float)) else ""))
            st.markdown(f"지금 보이는 탐지: **{found}**")
            # sent 는 관측 수다. 블랙박스가 연속 프레임의 같은 결함을 1건으로 합쳐(edge/blackbox/track_merge.py)
            # 결함이 화면에서 사라진 뒤에 보내므로 탐지 횟수(found)보다 작고 조금 늦게 오른다
            st.metric("서버로 보낸 관측", f"{s.get('sent', 0)}건",
                      help=f"프레임마다 찾은 탐지 {s.get('found', 0)}회를 결함마다 1건으로 합쳐 보낸 수")
            if s.get("note"):
                st.warning(s["note"])


view()
