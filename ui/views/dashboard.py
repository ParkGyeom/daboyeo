"""지도 대시보드 (관리자). 결함을 점으로 보여주고, 고른 결함의 상세와 처리 상태 변경.

배치(예약 누리집처럼): 머리 띠 → 결함 찾기 상자(라벨 위·값 아래 칸 한 줄) → 결함 현황 타일
→ 왼쪽 결함 위치 지도 | 오른쪽 결함 상세 정보 → 결함 목록 표.
찾기 조건: 시·도 → 시·군·구 → 읍·면·동(앞 선택으로 좁혀짐), 유형, 상태, 차량 번호, 기간.
현황·지도·표가 모두 같은 조건을 따른다. 거르기는 서버(GET /api/v1/defects 의 조건)가 하고,
고를 수 있는 지역은 전체 목록에서, 차량 번호는 등록된 차량에서 만든다.
risk_level 은 보여 주지 않는다 — 판별은 우리 몫이 아니다.

지도는 네이버 지도다 (ui/naver_map.py, .env 의 NAVER_MAP_KEY_ID 필요, 인터넷 필요).
키가 없으면 미리 받아 둔 오픈스트리트맵 도로선 위에 점을 찍는다 (ui/basemap.py, 인터넷 없이).
결함은 오른쪽 '조회할 결함', 목록 표의 줄, (키가 없을 때) 지도의 점으로 고른다.
5초마다 서버를 다시 불러 가상 블랙박스가 보내는 결함이 늘어나는 것이 보이게 한다.
"""

import math
from base64 import b64encode
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

import api
import basemap
import naver_map
import style

st.set_page_config(layout="wide")
style.page_header("지도 대시보드", "여러 차량이 함께 확인한 도로 결함의 위치와 처리 상태를 한눈에 봅니다.", kicker="관리자")

STATUS_SHAPE = {"미신고": "circle", "신고됨": "square", "보수 중": "triangle-up"}
NEXT_STATUS = {"reported": "under_repair"}
NEXT_LABEL = {"under_repair": "보수 중으로 변경"}  # 상태 변경 버튼 글자 (받침에 맞춰 '으로')
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
NO_DONG = ""  # 읍·면·동 고르기에서 '동 없음' (서버에 region3="" 로 보낸다)
KEYS = ("f_r1", "f_r2", "f_r3", "f_type", "f_status", "f_plate")
MAP_H = 640  # 지도 높이(px). 오른쪽 상세 정보와 높이를 맞춘다
# 처리 상태를 바꿀 수 없는 결함의 안내
NO_NEXT = {"unreported": "시민 신고가 접수되면 '보수 중'으로 바꿀 수 있습니다.",
           "under_repair": "보수 중인 결함입니다."}

B, SKY, SKY_SOFT, SOFT, LINE, MUTED = style.BRAND, style.SKY, style.SKY_SOFT, style.SOFT, style.LINE, style.MUTED
# 이 화면에만 쓰는 조각 모양. 다른 화면에도 쓰게 되면 style.py 로 옮길 것
#   찾기 상자(.st-key-card-dash-search): 입력 칸의 테두리를 없애고 칸 사이에 세로선 → 라벨 위·값 아래 칸
#   .dash-fields: 라벨/값 칸 격자 (상세 정보), .dash-cond: 걸린 조건 알약 줄
_CSS = f"""
<style>
.st-key-card-dash-search [data-testid="stColumn"] + [data-testid="stColumn"] {{
  border-left: 1px solid {LINE}; padding-left: 14px; }}
.st-key-dash-cond-row [data-testid="stColumn"] {{ border-left: 0 !important; padding-left: 0 !important; }}
.st-key-card-dash-search [data-testid="stWidgetLabel"] p {{ color: {MUTED}; font-size: .78rem; font-weight: 600; }}
/* Streamlit 1.67 의 선택 칸은 react-aria ComboBox, 날짜 칸은 stDateInputField. 버전이 바뀌면 여기를 고친다 */
.st-key-card-dash-search .react-aria-ComboBox > div,
.st-key-card-dash-search [data-testid="stDateInputField"] {{
  border: 0 !important; background: transparent !important; box-shadow: none !important; padding-left: 0 !important; }}
.st-key-card-dash-search .react-aria-ComboBox input {{ color: {B} !important; font-weight: 700; font-size: 1.02rem; }}
.st-key-card-dash-search .react-aria-DateField span {{ color: {B} !important; font-weight: 600; font-size: .9rem; }}
.st-key-card-dash-search .react-aria-ComboBox input::placeholder {{ color: {B}; opacity: 1; font-weight: 700; }}
.dash-cond {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; font-size: .84rem; color: {MUTED}; }}
.dash-cond .p {{ background: {SKY_SOFT}; color: {B}; border-radius: 999px; padding: .2em .8em; font-weight: 600; }}
.dash-cond .n {{ margin-left: 4px; }}
.dash-cond .n strong {{ color: {B}; font-size: 1rem; }}
.dash-photo {{ display: block; width: 100%; aspect-ratio: 2 / 1; object-fit: cover; border-radius: 12px;
  background: {SOFT}; }}
.dash-nophoto {{ color: {MUTED}; display: flex; align-items: center; justify-content: center; font-size: .85rem; }}
.dash-title {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: .1rem 0 .2rem; }}
.dash-title b {{ color: {B}; font-size: 1.2rem; font-weight: 700; margin-right: 4px; }}
.dash-addr {{ color: {MUTED}; font-size: .86rem; margin-bottom: .6rem; }}
.dash-fields {{ display: grid; grid-template-columns: 1fr 1fr; border-top: 1px solid {LINE}; }}
.dash-fields > div {{ padding: 9px 0 9px; border-bottom: 1px solid {LINE}; min-width: 0; }}
.dash-fields > div:nth-child(even) {{ padding-left: 14px; border-left: 1px solid {LINE}; }}
.dash-fields .wide {{ grid-column: 1 / -1; padding-left: 0 !important; border-left: 0 !important; }}
.dash-fields .k {{ color: {MUTED}; font-size: .76rem; font-weight: 600; }}
.dash-fields .v {{ color: {B}; font-size: .98rem; font-weight: 700; overflow-wrap: anywhere; line-height: 1.4; }}
.dash-move {{ display: flex; align-items: center; gap: 14px; margin: .1rem 0 .7rem; }}
.dash-move .k {{ color: {MUTED}; font-size: .76rem; font-weight: 600; }}
.dash-move .v {{ color: {B}; font-size: 1.05rem; font-weight: 700; }}
.dash-move .arrow {{ color: {SKY}; font-size: 1.3rem; font-weight: 700; padding-top: .7rem; }}
</style>
"""
st.html(_CSS)


def _clear() -> None:
    for k in KEYS:
        st.session_state[k] = None
    st.session_state["f_dates"] = ()


def _pick(col, label: str, key: str, options: list, fmt=str):
    """선택 칸. 비어 있으면(None) '전체'이고, 칸의 ⊗ 로 비운다.
    앞 선택이 바뀌어 고른 값이 목록에서 빠졌으면 비운다."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = None
    return col.selectbox(label, options, index=None, placeholder="전체", format_func=fmt, key=key)


def _filters(everything: list[dict], plates: list[str]) -> dict:
    """결함 찾기 상자 (예약 상자처럼 라벨/값 칸 한 줄). 서버 /defects 에 넘길 조건(dict)을 돌려준다.
    걸지 않은 조건은 None."""
    c = st.columns([1.05, 1.05, 1, 1.6, 0.95, 0.95, 1.15])
    r1 = _pick(c[0], "시·도", "f_r1", sorted({r["region1"] for r in everything if r.get("region1")}))
    pool = [r for r in everything if r.get("region1") and (r1 is None or r["region1"] == r1)]
    r2 = _pick(c[1], "시·군·구", "f_r2", sorted({r["region2"] for r in pool if r.get("region2")}))
    pool = [r for r in pool if r2 is None or r.get("region2") == r2]
    dongs = sorted({r.get("region3") or NO_DONG for r in pool})
    r3 = _pick(c[2], "읍·면·동", "f_r3", dongs, fmt=lambda x: x or "(동 없음)")
    days = c[3].date_input("확인 기간", value=(), format="YYYY.MM.DD", key="f_dates")
    t = _pick(c[4], "결함 유형", "f_type", list(api.TYPE_KO), fmt=api.TYPE_KO.get)
    s = _pick(c[5], "처리 상태", "f_status", list(api.STATUS_KO), fmt=api.STATUS_KO.get)
    plate = _pick(c[6], "차량 번호", "f_plate", plates)
    days = tuple(days or ())
    return {"region1": r1, "region2": r2, "region3": r3, "type": t, "status": s, "vehicle": plate,
            "date_from": days[0] if days else None, "date_to": days[1] if len(days) > 1 else None}


def _conditions(f: dict) -> list[str]:
    """지금 걸린 조건: ["서울특별시 › 강남구 › 역삼동", "균열", "12가 3456", "2026.11.18~2026.11.19"]."""
    where = " › ".join(x or "(동 없음)" for x in (f["region1"], f["region2"], f["region3"]) if x is not None)
    days = ""
    if f["date_from"]:
        days = f"{f['date_from']:%Y.%m.%d}~" + (f"{f['date_to']:%Y.%m.%d}" if f["date_to"] else "")
    parts = [where, f["type"] and api.TYPE_KO[f["type"]], f["status"] and api.STATUS_KO[f["status"]],
             f["vehicle"], days]
    return [p for p in parts if p]


def _summary(f: dict) -> str:
    """지금 걸린 조건을 한 줄로. 지도가 조건이 바뀐 것을 알아채는 데 쓴다."""
    return " · ".join(_conditions(f))


def _stats(rows: list[dict], everything: list[dict], active: bool) -> None:
    """결함 현황 타일: 전체 결함, 미신고·신고됨·보수 중."""
    n = len(rows)
    items = [("전체 결함", n, SKY, f"전체 {len(everything)}건 중" if active else "지도에 표시된 결함", "건")]
    for s, ko in api.STATUS_KO.items():
        k = sum(r["status"] == s for r in rows)
        sub = f"전체 {sum(r['status'] == s for r in everything)}건 중" if active else f"전체의 {round(100 * k / n) if n else 0}%"
        items.append((ko, k, style.STATUS_STYLE[s][0], sub, "건"))
    style.stat_cards(items)


@st.fragment(run_every=5)
def board():
    try:
        everything = api.defects()
        plates = api.plates()
    except api.ApiError as e:
        st.error(e.detail)
        return
    with style.card("dash-search"):
        style.section("결함 찾기", "지역·유형·상태·차량·기간으로 좁혀 봅니다")
        f = _filters(everything, plates)
        active = any(v is not None for v in f.values())
        try:
            rows = api.defects(**f) if active else everything
        except api.ApiError as e:
            st.error(e.detail)
            return
        if active:
            left, right = st.container(key="dash-cond-row").columns([5, 1], vertical_alignment="center")
            pills = "".join(f"<span class='p'>{escape(p)}</span>" for p in _conditions(f))
            left.html(f"<div class='dash-cond'>{pills}<span class='n'>조회 결과 <strong>{len(rows)}</strong>건"
                      f" / 전체 {len(everything)}건</span></div>")
            right.button("조건 초기화", on_click=_clear, width="stretch")

    style.section("결함 현황", "5초마다 새로 고칩니다")
    _stats(rows, everything, active)
    if not rows:
        st.info("조건에 맞는 결함이 없습니다." if active else "등록된 결함이 없습니다.")
        return

    df = pd.DataFrame(rows)
    df["유형"] = df["defect_type"].map(api.TYPE_KO)
    df["상태"] = df["status"].map(api.STATUS_KO)
    df["위치"] = [naver_map.short_place(r.get("address")) for r in rows]
    df["지역"] = [naver_map.region(r) or "-" for r in rows]
    df["차량 번호"] = [", ".join(r.get("plates") or []) or "-" for r in rows]
    df["처음 본 때"] = [api.short_when(str(r["first_seen_at"])) for r in rows]
    df["마지막 본 때"] = [api.short_when(str(r["last_seen_at"])) for r in rows]
    df["사진"] = [api.snapshot_url(r.get("snapshot_path")) for r in rows]

    _take_table_pick()  # 목록에서 고른 줄이 있으면 오른쪽 상세를 그 결함으로
    left, right = st.columns([5, 3], gap="medium")
    with left, style.card("dash-map"):
        if naver_map.key_id():
            # 네이버 지도. 점을 누르면 지도 위에 상세 말풍선, 빈 곳을 누르면 닫힘. 상태 변경은 오른쪽에서
            style.section("결함 위치", "점을 누르면 요약 · 숫자는 결함 묶음")
            naver_map.render(rows, height=MAP_H, fit=_summary(f) if active else "")
            st.caption("지도 © NAVER Corp.")
        else:
            style.section("결함 위치", "점을 누르면 오른쪽에 상세 정보")
            _altair_map(df)
    with right, style.card("dash-detail"):
        _detail(rows)
    _table(df)


def _altair_map(df: pd.DataFrame) -> None:
    """네이버 지도 키가 없을 때: 받아 둔 도로선 위에 점. 점을 누르면 오른쪽 상세가 그 결함으로 바뀐다."""
    pick = alt.selection_point(fields=["id"], name="pick")
    # 보여줄 범위: 결함들 + 여유 300m. 높이는 실제 거리 비율로, 폭은 칸에 맞춘다
    lat0 = df["lat"].mean()
    m_lat, m_lon = 300 / 111_320, 300 / (111_320 * math.cos(math.radians(lat0)))
    s, n = df["lat"].min() - m_lat, df["lat"].max() + m_lat
    w, e = df["lon"].min() - m_lon, df["lon"].max() + m_lon
    width = 760  # 1400px 화면에서 지도 칸의 대략 폭. 높이 비율 계산에만 쓴다
    height = int(min(MAP_H, max(320, width * (n - s) / ((e - w) * math.cos(math.radians(lat0))))))
    xs = alt.Scale(domain=[w, e], zero=False, nice=False)
    ys = alt.Scale(domain=[s, n], zero=False, nice=False)
    type_color = {api.TYPE_KO[k]: v for k, v in naver_map.TYPE_COLOR.items()}

    points = (
        alt.Chart(df.drop(columns=["사진"]))
        .mark_point(filled=True, size=150, opacity=1, stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X("lon:Q", title="경도", scale=xs),
            y=alt.Y("lat:Q", title="위도", scale=ys),
            color=alt.Color("유형:N", title="유형",
                            scale=alt.Scale(domain=list(type_color), range=list(type_color.values()))),
            shape=alt.Shape("상태:N", title="처리 상태",
                            scale=alt.Scale(domain=list(STATUS_SHAPE), range=list(STATUS_SHAPE.values()))),
            tooltip=["위치", "지역", "유형", "상태", "차량 번호", "observation_count"],
        )
        .add_params(pick)
    )
    roads = basemap.load_roads((s, w, n, e))
    if roads:
        rdf = pd.DataFrame(
            [{"rid": r["id"], "i": i, "lat": la, "lon": lo, "major": r["highway"] in MAJOR}
             for r in roads for i, (la, lo) in enumerate(r["coords"])]
        )
        lines = (
            alt.Chart(rdf)
            .mark_line(clip=True, color="#9aa3b2", opacity=0.55)
            .encode(x=alt.X("lon:Q", scale=xs), y=alt.Y("lat:Q", scale=ys), detail="rid:N", order="i:Q",
                    strokeWidth=alt.condition("datum.major", alt.value(3), alt.value(1.2)))
        )
        chart = alt.layer(lines, points)
    else:
        chart = points
    chart = chart.properties(height=height).configure_view(stroke=None)
    event = st.altair_chart(chart, width="stretch", on_select="rerun", key="map")
    st.caption(basemap.ATTRIBUTION if roads else "지도 배경 없음 (python -m ui.basemap fetch 로 도로를 받아 둔다)")
    sel = event.selection.get("pick") if event else None
    if sel:
        st.session_state["dash_pick"] = sel[0]["id"]


def _labels(rows: list[dict]) -> dict[str, str]:
    """고르기 목록의 이름: "봉은사로 102 균열". 같은 이름이 여럿이면 뒤에 (2), (3) 을 붙인다."""
    out, seen = {}, {}
    for r in rows:
        t = naver_map.title(r)
        seen[t] = seen.get(t, 0) + 1
        out[r["id"]] = t if seen[t] == 1 else f"{t} ({seen[t]})"
    return out


def _detail(rows: list[dict]) -> None:
    """오른쪽 결함 상세 정보: 조회할 결함 → 사진 → 제목·유형·상태 → 라벨/값 칸 → 처리 상태 변경."""
    style.section("결함 상세 정보")
    ids = [r["id"] for r in rows]
    cur = st.session_state.get("dash_pick")
    labels = _labels(rows)
    did = st.selectbox("조회할 결함", ids, index=ids.index(cur) if cur in ids else 0, format_func=labels.get)
    st.session_state["dash_pick"] = did
    d = next(r for r in rows if r["id"] == did)

    # 사진은 결함 부분을 자른 것이라 크기가 제각각이다. 같은 비율 틀에 채워 높이를 지도와 맞춘다
    photo = api.snapshot_bytes(d["snapshot_path"])
    if photo:
        st.html(f"<img class='dash-photo' alt='결함 사진' src='data:image/jpeg;base64,{b64encode(photo).decode()}'>")
    else:
        st.html("<div class='dash-photo dash-nophoto'>사진 없음</div>")

    where = naver_map.place(d.get("address")) or f"{d['lat']:.6f}, {d['lon']:.6f}"
    dong = naver_map.region(d)
    plates = ", ".join(d.get("plates") or []) or "-"
    fields = [("확인 차량", f"{d['vehicle_count']}대", ""), ("관측 횟수", f"{d['observation_count']}회", ""),
              ("처음 확인", api.when(str(d["first_seen_at"])), ""), ("최근 확인", api.when(str(d["last_seen_at"])), ""),
              ("차량 번호", plates, "wide")]
    cells = "".join(f"<div class='{c}'><div class='k'>{escape(k)}</div><div class='v'>{escape(v)}</div></div>"
                    for k, v, c in fields)
    st.html(
        f"<div class='dash-title'><b>{escape(naver_map.title(d))}</b>"
        f"{style.type_chip(d['defect_type'])}{style.badge(d['status'])}</div>"
        f"<div class='dash-addr'>{escape(where)}{escape(' · ' + dong) if dong else ''}</div>"
        f"<div class='dash-fields'>{cells}</div>"
    )

    nxt = NEXT_STATUS.get(d["status"])
    if not nxt:
        st.caption(NO_NEXT.get(d["status"], ""))
        return
    st.html(f"<div class='dash-move'><div><div class='k'>현재 상태</div><div class='v'>{escape(api.STATUS_KO[d['status']])}</div></div>"
            f"<div class='arrow'>→</div>"
            f"<div><div class='k'>바꿀 상태</div><div class='v'>{escape(api.STATUS_KO[nxt])}</div></div></div>")
    if st.button(NEXT_LABEL[nxt], key=f"st-{did}", type="primary", width="stretch"):
        try:
            api.set_status(did, nxt)
            st.rerun(scope="fragment")
        except api.ApiError as e:
            st.warning(e.detail)


def _take_table_pick() -> None:
    """목록에서 줄을 새로 고르면(on_select 콜백) 그 결함을 오른쪽 상세로 보낸다.
    고른 줄은 다시 그려도 남아 있으므로, 바뀐 때(콜백)만 반영해 '조회할 결함'으로 바꾼 것을 덮지 않는다."""
    pick = st.session_state.pop("dash_table_pick", None)
    if pick:
        st.session_state["dash_pick"] = pick


def _on_table_select() -> None:
    sel = st.session_state.get("dash_table")
    rows = sel.selection.rows if sel else []
    ids = st.session_state.get("dash_table_ids", [])
    if rows and rows[0] < len(ids):
        st.session_state["dash_table_pick"] = ids[rows[0]]


def _table(df: pd.DataFrame) -> None:
    """결함 목록 표(화면 폭 전체). 줄을 누르면 위 상세 정보가 그 결함으로 바뀐다."""
    with style.card("dash-table"):
        style.section("결함 목록", f"총 {len(df)}건 · 줄을 고르면 위에 상세 정보가 나옵니다")
        st.session_state["dash_table_ids"] = df["id"].tolist()
        cols = ["사진", "위치", "지역", "유형", "상태", "차량 번호", "vehicle_count", "observation_count",
                "confidence", "처음 본 때", "마지막 본 때"]
        # 상태 글자는 배지와 같은 색
        color = {api.STATUS_KO[s]: f"color: {fg}; font-weight: 700" for s, (fg, _) in style.STATUS_STYLE.items()}
        table = df[cols].style.map(lambda v: color.get(v, ""), subset=["상태"])
        st.dataframe(
            table,
            hide_index=True, width="stretch", key="dash_table",
            on_select=_on_table_select, selection_mode="single-row",
            row_height=46, height=min(640, 40 + 46 * len(df)),  # 줄이 적으면 빈칸 없이, 많으면 표 안에서 넘긴다
            column_config={  # 폭(px)은 1400px 화면에서 가로로 넘치지 않게 맞춘 값
                "사진": st.column_config.ImageColumn("사진", width=56),
                "위치": st.column_config.TextColumn("위치", width=130),
                "지역": st.column_config.TextColumn("지역", width=125),
                "유형": st.column_config.TextColumn("결함 유형", width=74),
                "상태": st.column_config.TextColumn("처리 상태", width=74),
                "차량 번호": st.column_config.TextColumn("확인 차량 번호", width=230),
                "vehicle_count": st.column_config.NumberColumn("차량 수", format="%d대", width=64),
                "observation_count": st.column_config.NumberColumn("관측", format="%d회", width=60),
                "confidence": st.column_config.NumberColumn(
                    "신뢰도", format="%.2f", width=64, help="같은 자리를 본 관측들의 탐지 신뢰도(0~1)"),
                "처음 본 때": st.column_config.TextColumn("처음 확인", width=100),
                "마지막 본 때": st.column_config.TextColumn("최근 확인", width=100),
            },
        )


board()
