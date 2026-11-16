"""지도 대시보드 (관리자). 결함을 점으로 보여주고, 고른 결함의 상세와 상태 변경.

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
SEV_COLOR = {"빠른 보수": "#d62728", "정기 보수": "#f2a900"}
NEXT_STATUS = {"reported": "under_repair"}
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}

f1, f2 = st.columns(2)
type_f = f1.selectbox("유형", [None, *api.TYPE_KO], format_func=lambda t: "전체" if t is None else api.TYPE_KO[t])
status_f = f2.selectbox("상태", [None, *api.STATUS_KO], format_func=lambda s: "전체" if s is None else api.STATUS_KO[s])


@st.fragment(run_every=5)
def board():
    try:
        rows = api.defects(type_f, status_f)
    except api.ApiError as e:
        st.error(e.detail)
        return
    c = st.columns(4)
    c[0].metric("결함", len(rows))
    for i, (s, ko) in enumerate(api.STATUS_KO.items(), 1):
        c[i].metric(ko, sum(r["status"] == s for r in rows))
    if not rows:
        st.info("결함이 없습니다.")
        return

    df = pd.DataFrame(rows)
    df["유형"] = df["defect_type"].map(api.TYPE_KO)
    df["severity"] = df["risk_level"].map(api.RISK_KO)
    df["상태"] = df["status"].map(api.STATUS_KO)
    df["번호"] = df["id"].map(api.short)
    if naver_map.key_id():
        # 네이버 지도. 점을 누르면 지도 위에 상세 말풍선, 상태 변경은 아래 '결함 번호'에서
        naver_map.render(rows)
        st.caption("지도 © NAVER Corp. · 점을 누르면 상세가 뜹니다. 상태 변경은 아래에서 결함 번호를 고릅니다.")
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

    points = (
        alt.Chart(df)
        .mark_point(filled=True, size=160, opacity=0.9, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("lon:Q", title="경도", scale=xs),
            y=alt.Y("lat:Q", title="위도", scale=ys),
            color=alt.Color("severity:N", title="위험도",
                            scale=alt.Scale(domain=list(SEV_COLOR), range=list(SEV_COLOR.values()))),
            shape=alt.Shape("상태:N", title="상태",
                            scale=alt.Scale(domain=list(STATUS_SHAPE), range=list(STATUS_SHAPE.values()))),
            tooltip=["번호", "유형", "severity", "상태", "vehicle_count", "confidence"],
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


def _detail(df: pd.DataFrame, rows: list[dict]) -> None:
    """결함 표와 고른 결함의 상세·상태 변경."""
    left, right = st.columns([3, 2])
    left.dataframe(
        df[["번호", "유형", "severity", "상태", "vehicle_count", "observation_count", "confidence", "jurisdiction"]]
        .rename(columns={"severity": "위험도", "vehicle_count": "차량 수",
                         "observation_count": "관측 수", "confidence": "최고 신뢰도", "jurisdiction": "관할"}),
        hide_index=True, width="stretch",
    )
    ids = df["id"].tolist()
    cur = st.session_state.get("dash_pick")
    did = right.selectbox("결함 번호", ids, index=ids.index(cur) if cur in ids else 0, format_func=api.short)
    d = next(r for r in rows if r["id"] == did)
    photo = api.snapshot_bytes(d["snapshot_path"])
    if photo:
        right.image(photo, width=260)
    where = naver_map.place(d.get("address")) or f"{d['lat']:.6f}, {d['lon']:.6f}"
    right.markdown(
        f"**#{api.short(d['id'])} {api.TYPE_KO[d['defect_type']]}** · {api.RISK_KO[d['risk_level']]} · "
        f"**{api.STATUS_KO[d['status']]}**  \n"
        f"{where} · {d['jurisdiction']}  \n"
        f"관측 {d['observation_count']}건 · 차량 {d['vehicle_count']}대 · 신뢰도 {d['confidence']:.2f}"
    )
    nxt = NEXT_STATUS.get(d["status"])
    if nxt and right.button(f"'{api.STATUS_KO[nxt]}'(으)로 변경", key=f"st-{did}"):
        try:
            api.set_status(did, nxt)
            st.rerun(scope="fragment")
        except api.ApiError as e:
            right.warning(e.detail)


board()
