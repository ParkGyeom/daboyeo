"""지도 대시보드 (관리자). 결함을 점으로 보여주고, 고른 결함의 상세와 상태 변경.

지도 배경은 미리 받아 둔 오픈스트리트맵 도로선이다 (web/basemap.py, B10). 인터넷 없이 그린다.
받아 둔 것이 없으면 배경 없이 점만 찍는다.
5초마다 서버를 다시 불러 가상 블랙박스가 보내는 결함이 늘어나는 것이 보이게 한다.
"""

import math

import altair as alt
import pandas as pd
import streamlit as st

import api
import basemap

st.set_page_config(layout="wide")
st.title("지도 대시보드")

STATUS_SHAPE = {"미신고": "circle", "신고됨": "square", "보수 중": "triangle-up"}
SEV_COLOR = {"빠른 보수": "#d62728", "정기 보수": "#f2a900"}
NEXT_STATUS = {"신고됨": "보수 중"}
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}

f1, f2 = st.columns(2)
type_f = f1.selectbox("유형", [None, "pothole", "crack", "lane_wear"], format_func=lambda t: "전체" if t is None else api.TYPE_KO[t])
status_f = f2.selectbox("상태", [None, "미신고", "신고됨", "보수 중"], format_func=lambda s: "전체" if s is None else s)


@st.fragment(run_every=5)
def board():
    try:
        rows = api.defects(type_f, status_f)
    except api.ApiError as e:
        st.error(e.detail)
        return
    c = st.columns(4)
    c[0].metric("결함", len(rows))
    for i, s in enumerate(["미신고", "신고됨", "보수 중"], 1):
        c[i].metric(s, sum(r["status"] == s for r in rows))
    if not rows:
        st.info("결함이 없습니다.")
        return

    df = pd.DataFrame(rows)
    df["유형"] = df["type"].map(api.TYPE_KO)
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
            shape=alt.Shape("status:N", title="상태",
                            scale=alt.Scale(domain=list(STATUS_SHAPE), range=list(STATUS_SHAPE.values()))),
            tooltip=["id", "유형", "severity", "status", "device_count", "max_confidence"],
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

    left, right = st.columns([3, 2])
    left.dataframe(
        df[["id", "유형", "severity", "status", "device_count", "observation_count", "max_confidence", "jurisdiction"]]
        .rename(columns={"severity": "위험도", "status": "상태", "device_count": "기기 수",
                         "observation_count": "관측 수", "max_confidence": "최고 신뢰도", "jurisdiction": "관할"}),
        hide_index=True, width="stretch",
    )
    ids = df["id"].tolist()
    cur = st.session_state.get("dash_pick")
    did = right.selectbox("결함 번호", ids, index=ids.index(cur) if cur in ids else 0)
    d = next(r for r in rows if r["id"] == did)
    photo = api.snapshot_bytes(d["best_snapshot_path"])
    if photo:
        right.image(photo, width=260)
    right.markdown(
        f"**#{d['id']} {api.TYPE_KO[d['type']]}** · {d['severity']} · **{d['status']}**  \n"
        f"{d['lat']:.6f}, {d['lon']:.6f} · {d['jurisdiction']}  \n"
        f"관측 {d['observation_count']}건 · 기기 {d['device_count']}대 · 신뢰도 {d['max_confidence']:.2f}"
    )
    nxt = NEXT_STATUS.get(d["status"])
    if nxt and right.button(f"'{nxt}'(으)로 변경", key=f"st-{did}"):
        try:
            api.set_status(did, nxt)
            st.rerun(scope="fragment")
        except api.ApiError as e:
            right.warning(e.detail)


board()
