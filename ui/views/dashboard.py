"""지도 대시보드 (관리자). 결함을 점으로 보여주고, 고른 결함의 상세와 상태 변경.

위쪽 필터로 나눠 본다: 시·도 → 시·군·구 → 읍·면·동(앞 선택으로 좁혀짐), 유형, 상태, 차량 번호, 기간.
숫자 카드·지도·표가 모두 같은 필터를 따른다. 거르기는 서버(GET /api/v1/defects 의 조건)가 하고,
고를 수 있는 지역은 전체 목록에서, 차량 번호는 등록된 차량에서 만든다.
risk_level 은 보여 주지 않는다 — 판별은 우리 몫이 아니다.

지도는 네이버 지도다 (ui/naver_map.py, .env 의 NAVER_MAP_KEY_ID 필요, 인터넷 필요).
키가 없으면 미리 받아 둔 오픈스트리트맵 도로선 위에 점을 찍는다 (ui/basemap.py, 인터넷 없이).
5초마다 서버를 다시 불러 가상 블랙박스가 보내는 결함이 늘어나는 것이 보이게 한다.
"""

import math

import altair as alt
import pandas as pd
import streamlit as st

import api
import basemap
import naver_map

st.set_page_config(layout="wide")
st.title("지도 대시보드")

STATUS_SHAPE = {"미신고": "circle", "신고됨": "square", "보수 중": "triangle-up"}
NEXT_STATUS = {"reported": "under_repair"}
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
NO_DONG = ""  # 읍·면·동 고르기에서 '동 없음' (서버에 region3="" 로 보낸다)
KEYS = ("f_r1", "f_r2", "f_r3", "f_type", "f_status", "f_plate")


def _clear() -> None:
    for k in KEYS:
        st.session_state[k] = None
    st.session_state["f_dates"] = ()


def _pick(col, label: str, key: str, options: list, fmt=str):
    """선택 상자. 비어 있으면(None) '전체'이고, 상자의 ⊗ 로 비운다.
    앞 선택이 바뀌어 고른 값이 목록에서 빠졌으면 비운다."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = None
    return col.selectbox(label, options, index=None, placeholder="전체", format_func=fmt, key=key)


def _filters(everything: list[dict], plates: list[str]) -> dict:
    """필터 줄. 서버 /defects 에 넘길 조건(dict)을 돌려준다. 걸지 않은 조건은 None."""
    a = st.columns(4)
    r1 = _pick(a[0], "시·도", "f_r1", sorted({r["region1"] for r in everything if r.get("region1")}))
    pool = [r for r in everything if r.get("region1") and (r1 is None or r["region1"] == r1)]
    r2 = _pick(a[1], "시·군·구", "f_r2", sorted({r["region2"] for r in pool if r.get("region2")}))
    pool = [r for r in pool if r2 is None or r.get("region2") == r2]
    dongs = sorted({r.get("region3") or NO_DONG for r in pool})
    r3 = _pick(a[2], "읍·면·동", "f_r3", dongs, fmt=lambda x: x or "(동 없음)")
    days = a[3].date_input("기간 (처음~끝 날짜)", value=(), format="YYYY.MM.DD", key="f_dates")
    b = st.columns(4)
    t = _pick(b[0], "결함 유형", "f_type", list(api.TYPE_KO), fmt=api.TYPE_KO.get)
    s = _pick(b[1], "상태", "f_status", list(api.STATUS_KO), fmt=api.STATUS_KO.get)
    plate = _pick(b[2], "차량 번호", "f_plate", plates)
    b[3].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)  # 버튼을 상자 높이에 맞춘다
    b[3].button("필터 모두 지우기", on_click=_clear, width="stretch")
    days = tuple(days or ())
    return {"region1": r1, "region2": r2, "region3": r3, "type": t, "status": s, "vehicle": plate,
            "date_from": days[0] if days else None, "date_to": days[1] if len(days) > 1 else None}


def _summary(f: dict) -> str:
    """지금 걸린 필터를 한 줄로: "서울특별시 › 강남구 › 역삼동 · 균열 · 12가 3456 · 11/18~11/19"."""
    where = " › ".join(x or "(동 없음)" for x in (f["region1"], f["region2"], f["region3"]) if x is not None)
    days = ""
    if f["date_from"]:
        days = f"{f['date_from']:%m/%d}~" + (f"{f['date_to']:%m/%d}" if f["date_to"] else "")
    parts = [where, f["type"] and api.TYPE_KO[f["type"]], f["status"] and api.STATUS_KO[f["status"]],
             f["vehicle"], days]
    return " · ".join(p for p in parts if p)


@st.fragment(run_every=5)
def board():
    try:
        everything = api.defects()
        plates = api.plates()
    except api.ApiError as e:
        st.error(e.detail)
        return
    f = _filters(everything, plates)
    active = any(v is not None for v in f.values())
    try:
        rows = api.defects(**f) if active else everything
    except api.ApiError as e:
        st.error(e.detail)
        return

    c = st.columns(4)
    c[0].metric("결함", len(rows), help=None if not active else f"전체 {len(everything)}건 중")
    for i, (s, ko) in enumerate(api.STATUS_KO.items(), 1):
        c[i].metric(ko, sum(r["status"] == s for r in rows))
    if active:
        st.caption(f"필터: {_summary(f)} — 전체 {len(everything)}건 중 {len(rows)}건")
    if not rows:
        st.info("조건에 맞는 결함이 없습니다." if active else "결함이 없습니다.")
        return

    df = pd.DataFrame(rows)
    df["유형"] = df["defect_type"].map(api.TYPE_KO)
    df["상태"] = df["status"].map(api.STATUS_KO)
    df["위치"] = [naver_map.short_place(r.get("address")) for r in rows]
    df["지역"] = [naver_map.region(r) or "-" for r in rows]
    df["차량 번호"] = [", ".join(r.get("plates") or []) or "-" for r in rows]
    df["처음 본 때"] = [api.short_when(str(r["first_seen_at"])) for r in rows]
    df["마지막 본 때"] = [api.short_when(str(r["last_seen_at"])) for r in rows]
    if naver_map.key_id():
        # 네이버 지도. 점을 누르면 지도 위에 상세 말풍선, 빈 곳을 누르면 닫힘. 상태 변경은 아래 '결함 고르기'에서
        naver_map.render(rows, fit=_summary(f) if active else "")
        st.caption("지도 © NAVER Corp. · 점을 누르면 상세가 뜨고, 빈 곳을 누르면 닫힙니다. 상태 변경은 아래에서 결함을 고릅니다.")
    else:
        _altair_map(df)
    _detail(df, rows)


def _altair_map(df: pd.DataFrame) -> None:
    """네이버 지도 키가 없을 때: 받아 둔 도로선 위에 점. 점을 누르면 아래 상세가 그 결함으로 바뀐다."""
    pick = alt.selection_point(fields=["id"], name="pick")
    # 보여줄 범위: 결함들 + 여유 300m. 가로세로 비율을 실제 거리에 맞춘다
    lat0 = df["lat"].mean()
    m_lat, m_lon = 300 / 111_320, 300 / (111_320 * math.cos(math.radians(lat0)))
    s, n = df["lat"].min() - m_lat, df["lat"].max() + m_lat
    w, e = df["lon"].min() - m_lon, df["lon"].max() + m_lon
    width = 760
    height = int(min(720, max(320, width * (n - s) / ((e - w) * math.cos(math.radians(lat0))))))
    xs = alt.Scale(domain=[w, e], zero=False, nice=False)
    ys = alt.Scale(domain=[s, n], zero=False, nice=False)
    type_color = {api.TYPE_KO[k]: v for k, v in naver_map.TYPE_COLOR.items()}

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=160, opacity=0.9, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("lon:Q", title="경도", scale=xs),
            y=alt.Y("lat:Q", title="위도", scale=ys),
            color=alt.Color("유형:N", title="유형",
                            scale=alt.Scale(domain=list(type_color), range=list(type_color.values()))),
            shape=alt.Shape("상태:N", title="상태",
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
            .mark_line(clip=True, color="#8a8f98", opacity=0.55)
            .encode(x=alt.X("lon:Q", scale=xs), y=alt.Y("lat:Q", scale=ys), detail="rid:N", order="i:Q",
                    strokeWidth=alt.condition("datum.major", alt.value(3), alt.value(1.2)))
        )
        chart = alt.layer(lines, points)
    else:
        chart = points
    chart = chart.properties(width=width, height=height)
    event = st.altair_chart(chart, width="content", on_select="rerun", key="map")
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


def _detail(df: pd.DataFrame, rows: list[dict]) -> None:
    """결함 표(화면 폭 전체)와 그 아래 고른 결함의 상세·상태 변경."""
    st.dataframe(
        df[["위치", "지역", "유형", "상태", "차량 번호", "observation_count", "처음 본 때", "마지막 본 때"]]
        .rename(columns={"observation_count": "관측 수"}),
        hide_index=True, width="stretch",
    )
    right, photo_col = st.columns([3, 2])
    ids = df["id"].tolist()
    cur = st.session_state.get("dash_pick")
    labels = _labels(rows)
    did = right.selectbox("결함 고르기", ids, index=ids.index(cur) if cur in ids else 0, format_func=labels.get)
    st.session_state["dash_pick"] = did
    d = next(r for r in rows if r["id"] == did)
    photo = api.snapshot_bytes(d["snapshot_path"])
    if photo:
        photo_col.image(photo, width=360)
    where = naver_map.place(d.get("address")) or f"{d['lat']:.6f}, {d['lon']:.6f}"
    dong = naver_map.region(d)
    plates = ", ".join(d.get("plates") or []) or "-"
    first, last = api.when(str(d["first_seen_at"])), api.when(str(d["last_seen_at"]))
    right.markdown(
        f"**{naver_map.title(d)}** · **{api.STATUS_KO[d['status']]}**  \n"
        f"{where}" + (f" · {dong}" if dong else "") + "  \n"
        f"차량 {plates} · 관측 {d['observation_count']}건 · 신뢰도 {d['confidence']:.2f}  \n"
        f"처음 {first}" + (f" · 마지막 {last}" if last != first else "")
    )
    nxt = NEXT_STATUS.get(d["status"])
    if nxt and right.button(f"'{api.STATUS_KO[nxt]}'(으)로 변경", key=f"st-{did}"):
        try:
            api.set_status(did, nxt)
            st.rerun(scope="fragment")
        except api.ApiError as e:
            right.warning(e.detail)


board()
