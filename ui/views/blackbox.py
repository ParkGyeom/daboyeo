"""블랙박스 화면 (관리자·시연용). 가상 블랙박스가 지금 보고 있는 프레임과 탐지 박스.

가상 블랙박스(edge/blackbox/live.py)가 server_data/live/ 에 쓰는 파일을 0.5초마다 다시 읽는다.
기기 여러 대를 흰 둥근 카드로 나란히 보여준다. '화면 선택'에서 한 대를 고르면
그 차량만 화면 폭 가득 보여준다 (녹화에서 관리자 지도와 나란히 둘 때). 주소에 ?bb=bb-01 을 붙이면
처음부터 그 차량만 보인다.

0.5초마다 다시 그리므로 카드의 HTML 은 작게 둔다. 사진은 st.image 로 넣고(HTML 에 싣지 않는다),
카드 머리(차량 번호·상태)와 아래 정보 칸만 st.html 조각이다. 모양 CSS 는 fragment 밖에서 한 번 넣는다.
"""

import json
import sys
import time
from html import escape
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from edge.blackbox.live import COLORS, LABELS, LIVE_DIR, is_current  # noqa: E402
import api  # noqa: E402
import style  # noqa: E402

STALE_S = 5  # 이 시간 동안 새 프레임이 없으면 멈춘 것으로 본다
ALL = "전체"
STATE_TEXT = {"live": "수신 중", "stale": "정지", "ended": "주행 종료"}

# 차량 카드는 st.container(key="bbcard-…") 이고 Streamlit 이 붙이는 st-key-bbcard-… 클래스로 고른다.
# '화면 선택'은 key="bb_focus" 라 st-key-bb_focus 로 고른다. 이 화면에서만 쓰는 모양이라 style.py 가 아니라 여기 둔다
_CSS = f"""
<style>
.block-container, [data-testid="stMainBlockContainer"] {{ max-width: 1400px; }}
.bb-legend {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: -.4rem 0 1rem; }}
.bb-legend .note {{ color: {style.MUTED}; font-size: .8rem; margin-left: 8px; }}

/* 화면 선택: 알약 버튼 줄. 고른 것은 남색 바탕 흰 글자 (Streamlit 이 aria-checked 를 붙인다) */
.st-key-bb_focus button {{ border-radius: 999px !important; border: 1px solid #c9d1de !important;
                           padding: 0 14px !important; color: {style.BRAND}; background: #fff; }}
.st-key-bb_focus button[aria-checked="true"] {{ background: {style.BRAND} !important; border-color: {style.BRAND} !important; }}
.st-key-bb_focus button[aria-checked="true"] * {{ color: #fff !important; }}

.bb-sum {{ display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 8px; margin-bottom: 2px; }}
.bb-sum span {{ background: {style.SOFT}; border-radius: 999px; padding: 5px 14px; font-size: .8rem; color: {style.MUTED}; }}
.bb-sum b {{ color: {style.BRAND}; font-weight: 700; margin-left: 6px; }}
.bb-sum .on {{ background: {style.SKY_SOFT}; }}

[class*="st-key-bbcard-"] {{
  background: #fff; border: 1px solid {style.LINE}; border-radius: 14px;
  padding: 14px; gap: 10px !important; box-shadow: 0 2px 10px rgba(5, 23, 102, .04);
}}
[class*="st-key-bbcard-"] [data-testid="stImage"] img {{ border-radius: 10px; display: block; background: #000; }}
[class*="st-key-bbcard-"] [data-testid="stElementToolbar"] {{ display: none; }}
/* 정지·주행 종료 카드의 영상은 흐리게 (지금 수신 중인 카드가 눈에 띄게) */
[class*="st-key-bbcard-"]:has(.bb-state.stale) img,
[class*="st-key-bbcard-"]:has(.bb-state.ended) img {{ filter: grayscale(.85); opacity: .55; }}

.bb-top {{ display: flex; align-items: baseline; gap: 8px; }}
.bb-top .plate {{ color: {style.BRAND}; font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em; }}
.bb-top .dev {{ color: {style.MUTED}; font-size: .8rem; }}
.bb-state {{ margin-left: auto; align-self: center; padding: .2em .8em; border-radius: 999px;
             font-size: .76rem; font-weight: 700; line-height: 1.45; }}
.bb-state.live {{ background: {style.SKY_SOFT}; color: {style.BRAND}; }}
.bb-state.stale, .bb-state.ended {{ background: #eceef2; color: {style.MUTED}; }}

.bb-info {{ display: grid; grid-template-columns: 1fr 1.7fr 1fr; gap: 8px; }}
.bb-info.side {{ grid-template-columns: 1fr; }}
.bb-cell {{ background: {style.SOFT}; border-radius: 10px; padding: 8px 12px; min-width: 0; min-height: 74px; }}
.bb-cell .l {{ color: {style.MUTED}; font-size: .74rem; font-weight: 600; }}
.bb-cell .v {{ color: {style.BRAND}; font-size: 1.05rem; font-weight: 700; line-height: 1.4; }}
.bb-cell .v small {{ font-size: .76rem; font-weight: 600; color: {style.MUTED}; margin-left: 3px; }}
.bb-cell .s {{ color: {style.MUTED}; font-size: .7rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bb-dets {{ display: flex; flex-wrap: wrap; gap: 2px 10px; font-size: .78rem; color: {style.INK}; font-weight: 600;
            line-height: 1.55; padding-top: 2px; }}
.bb-dets span {{ display: inline-flex; align-items: center; gap: 4px; }}
.bb-dets i {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; }}
.bb-dets em {{ font-style: normal; color: {style.MUTED}; font-weight: 500; }}
.bb-dets .none {{ color: {style.MUTED}; font-weight: 500; }}
.bb-note {{ color: #8a5300; font-size: .74rem; }}

.bb-empty {{ background: {style.SOFT}; border-radius: 14px; padding: 28px; color: {style.MUTED}; font-size: .9rem; }}
.bb-empty b {{ color: {style.BRAND}; font-size: 1.05rem; }}
.bb-empty code {{ font-size: .78rem; }}
</style>
"""


def hexcolor(bgr: tuple[int, int, int]) -> str:
    """OpenCV 색(BGR)을 CSS 색으로. 프레임의 박스 색과 같게."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def legend() -> str:
    """범례 알약 칩 줄 (박스 색 점 + 이름)."""
    chips = "".join(f"<span class='dby-chip'><i style='background:{hexcolor(COLORS[k])}'></i>{escape(v)}</span>"
                    for k, v in LABELS.items())
    return (f"<div class='bb-legend'>{chips}<span class='note'>굵은 선은 전송하는 결함, "
            f"가는 선은 위치만 확인하는 시설물입니다</span></div>")


def read_devices() -> list[tuple[dict, bytes | None]]:
    out = []
    now = time.time()
    for js in sorted(LIVE_DIR.glob("*.json")):
        try:
            status = json.loads(js.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # 쓰는 중
        if not is_current(status, now):
            continue  # 지난 주행이 남긴 화면 (강제로 끈 주행 등)
        jpg = js.with_suffix(".jpg")
        try:
            img = jpg.read_bytes()
        except OSError:
            img = None
        out.append((status, img))
    return out


def state_of(s: dict, now: float) -> str:
    """live(수신 중) · stale(정지) · ended(주행 종료)."""
    if s.get("ended"):
        return "ended"
    return "stale" if now - s.get("updated_at", 0) > STALE_S else "live"


def pick(devices: list[tuple[dict, bytes | None]]) -> list[tuple[dict, bytes | None]]:
    """'화면 선택'으로 고른 차량만. '전체'이거나 고른 차량이 지금 없으면 모두."""
    if "bb_focus" not in st.session_state:
        st.session_state.bb_focus = st.query_params.get("bb", ALL)
    focus = st.session_state.bb_focus
    ids = [s["device_id"] for s, _ in devices]
    if len(ids) < 2 and focus in (ALL, None):
        return devices  # 고를 것이 없다
    # 고른 차량이 잠깐 화면에서 빠져도(주행을 다시 켜는 사이) 고른 것은 남긴다
    options = [ALL, *ids, *([focus] if focus not in (ALL, None, *ids) else [])]
    choice = st.segmented_control("화면 선택", options, key="bb_focus",
                                  format_func=lambda d: d if d == ALL else api.plate_of(d))
    return [d for d in devices if d[0]["device_id"] == choice] or devices


def summary(devices: list[tuple[dict, bytes | None]], now: float) -> str:
    """오른쪽 위 현황 알약: 상태별 대수, 전송 관측 합계."""
    states = [state_of(s, now) for s, _ in devices]
    sent = sum(s.get("sent", 0) for s, _ in devices)
    return (f"<div class='bb-sum'><span class='on'>수신 중<b>{states.count('live')}대</b></span>"
            f"<span>정지<b>{states.count('stale')}대</b></span>"
            f"<span>주행 종료<b>{states.count('ended')}대</b></span>"
            f"<span>전송 관측 합계<b>{sent}건</b></span></div>")


def card_head(s: dict, state: str) -> str:
    # 차량 번호 (12가 3456). 기기 이름(bb-01)은 서버에 등록된 번호로 바꾼다
    dev = s["device_id"]
    plate = api.plate_of(dev)
    dev_txt = f"<span class='dev'>{escape(dev)}</span>" if plate != dev else ""
    return (f"<div class='bb-top'><span class='plate'>{escape(plate)}</span>{dev_txt}"
            f"<span class='bb-state {state}'>{STATE_TEXT[state]}</span></div>")


def card_info(s: dict, state: str, side: bool = False) -> str:
    """라벨 위 / 값 아래 칸 3개: 영상 시간, 지금 탐지, 전송 관측."""
    t = s.get("t")
    clock = f"{int(t // 60):02d}:{t % 60:04.1f}" if isinstance(t, (int, float)) else "--:--"
    # 같은 유형은 한 줄로 묶는다 (표지판 6개가 6줄이 되어 카드 높이가 0.5초마다 바뀌지 않게)
    best: dict[str, list[float]] = {}
    for b in s.get("boxes") or []:
        best.setdefault(b["type"], []).append(b["confidence"])
    dets = "".join(
        f"<span><i style='background:{hexcolor(COLORS.get(k, (255, 255, 255)))}'></i>"
        f"{escape(LABELS.get(k, k))}{f' {len(v)}개' if len(v) > 1 else ''}<em>{max(v):.2f}</em></span>"
        for k, v in best.items()
    ) or "<span class='none'>없음</span>"
    # sent 는 관측 수다. 블랙박스가 연속 프레임의 같은 결함을 1건으로 합쳐(edge/blackbox/track_merge.py)
    # 결함이 화면에서 사라진 뒤에 보내므로 탐지 횟수(found)보다 작고 조금 늦게 오른다
    found = s.get("found", 0)
    tip = f"프레임마다 찾은 탐지 {found}회를 결함마다 1건으로 합쳐 보낸 수"
    note = s.get("note") if state != "ended" else ""  # 주행 종료는 머리의 상태 알약에 이미 있다
    src = escape(s.get("source", ""))
    return (f"<div class='bb-info{' side' if side else ''}'>"
            f"<div class='bb-cell'><div class='l'>영상 시간</div><div class='v'>{clock}</div>"
            f"<div class='s' title='{src}'>{src}</div></div>"
            f"<div class='bb-cell'><div class='l'>지금 탐지</div><div class='bb-dets'>{dets}</div>"
            + (f"<div class='bb-note'>{escape(note)}</div>" if note else "") + "</div>"
            f"<div class='bb-cell' title='{escape(tip)}'><div class='l'>전송 관측</div>"
            f"<div class='v'>{s.get('sent', 0)}<small>건</small></div><div class='s'>탐지 {found}회</div></div>"
            "</div>")


def card(s: dict, img: bytes | None, now: float, wide: bool = False) -> None:
    """차량 한 대의 흰 카드: 머리(차량 번호·상태) → 영상 → 정보 칸 3개.

    wide(한 대만 크게)면 정보 칸을 영상 오른쪽에 세로로 둔다. 아래에 두면 화면 밖으로 밀린다.
    """
    state = state_of(s, now)
    with st.container(key=f"bbcard-{s['device_id']}"):
        st.html(card_head(s, state))
        frame, side = st.columns([4, 1], gap="small") if wide else (st.container(), st.container())
        if img:
            frame.image(img, width="stretch")
        side.html(card_info(s, state, side=wide))


@st.fragment(run_every=0.5)
def view() -> None:
    devices = read_devices()
    if not devices:
        st.html("<div class='bb-empty'><b>수신 중인 블랙박스가 없습니다</b><br>"
                "가상 블랙박스를 실행하면 여기에 영상이 나타납니다. 예: <code>uv run python -m edge.blackbox.run "
                "--video data/videos/v1.mp4 --model ai/models/detector_v3.onnx</code></div>")
        return
    now = time.time()
    left, right = st.columns([1, 1], vertical_alignment="bottom")
    with left:
        shown = pick(devices)
    with right:
        st.html(summary(devices, now))
    if len(shown) == 1:
        card(*shown[0], now, wide=True)
        return
    per_row = min(3, len(shown))
    for i in range(0, len(shown), per_row):
        for col, (s, img) in zip(st.columns(per_row, gap="medium"), shown[i:i + per_row]):
            with col:
                card(s, img, now)


st.html(_CSS)
style.page_header("블랙박스 화면", "가상 블랙박스가 분석 중인 영상과 탐지 결과를 0.5초마다 새로 보여 줍니다.",
                  kicker="관리자 · 실시간")
st.html(legend())
view()
