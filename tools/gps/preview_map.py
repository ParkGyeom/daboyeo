"""재구성 경로 미리보기 (docs/plan.md 0.7): 경로가 도로 위에 있는지, 기준점이 맞는지 눈으로 본다.

인터넷 없이 열리는 HTML 한 파일(SVG)을 만든다. 배경 도로선은 ui/basemap.py 가 쓰는 data/basemap/*.json 에서
경로 주변만 가져온다. 그 구간 도로가 아직 없으면 --fetch-roads 로 한 번 받는다(인터넷 필요, OSM Overpass).
받은 파일(data/basemap/gps_<이름>.json)은 관리자 화면의 오프라인 도로선 지도에도 같이 나온다.

  uv run python -m tools.gps.preview_map gn_a                  # → data/videos/gn_a.preview.html
  uv run python -m tools.gps.preview_map gn_a --fetch-roads    # 주변 도로를 받고 그린다
  uv run python -m tools.gps.preview_map --gpx a.gpx --anchors a.csv --gps a.gps.csv --name synth --out a.html

그림: 회색 = 도로(OSM), 파랑 = 그린 경로, 작은 점 = 재구성 CSV(10초마다), 빨강 = 기준점(번호),
주황 X = 그 기준점을 빼고 보간한 위치(hold-out). 휠로 확대, 끌어서 이동.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path

from daboyeo_contracts import GpsTrack

from tools.gps.holdout_error import Holdout, holdout
from tools.gps.reconstruct import (
    VIDEO_DIR,
    AnchorError,
    Route,
    Snapped,
    add_input_args,
    fmt_time,
    load_anchors,
    load_gpx,
    resolve_inputs,
    snap_anchors,
)

MARGIN_M = 150
DOT_EVERY_S = 10.0
SIZE = 900  # SVG 긴 변 (px)
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}


def route_bbox(latlon: list[tuple[float, float]], margin_m: float = MARGIN_M) -> tuple[float, float, float, float]:
    """(남, 서, 북, 동) — ui/basemap.py 와 같은 순서."""
    lats, lons = [p[0] for p in latlon], [p[1] for p in latlon]
    dlat = margin_m / 111_320
    dlon = margin_m / (111_320 * math.cos(math.radians(sum(lats) / len(lats))))
    return min(lats) - dlat, min(lons) - dlon, max(lats) + dlat, max(lons) + dlon


def fetch_roads(bbox: tuple[float, float, float, float], name: str) -> Path:
    """경로 주변 도로를 OSM Overpass 에서 받아 data/basemap/gps_<name>.json 에 ui/basemap.py 형식으로 둔다."""
    import httpx

    from ui.basemap import ATTRIBUTION, BASEMAP_DIR, HIGHWAYS, OVERPASS_URL, USER_AGENT

    q = f'[out:json][timeout:60];way["highway"~"^({HIGHWAYS})$"]({",".join(f"{v:.6f}" for v in bbox)});out geom;'
    for attempt in range(3):  # 공용 서버가 바쁘면 429·502·504
        r = httpx.post(OVERPASS_URL, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=120)
        if r.status_code not in (429, 502, 503, 504):
            break
        print(f"Overpass 가 바쁨({r.status_code}), {20 * (attempt + 1)}초 뒤 다시 시도")
        time.sleep(20 * (attempt + 1))
    r.raise_for_status()
    roads = [
        {"id": el["id"], "highway": el.get("tags", {}).get("highway", ""), "name": el.get("tags", {}).get("name", ""),
         "coords": [[round(p["lat"], 6), round(p["lon"], 6)] for p in el["geometry"]]}
        for el in r.json().get("elements", []) if el.get("type") == "way" and el.get("geometry")
    ]
    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    path = BASEMAP_DIR / f"gps_{name}.json"
    path.write_text(json.dumps({"attribution": ATTRIBUTION, "bbox": bbox, "roads": roads}, ensure_ascii=False))
    print(f"{path}: 도로 {len(roads)}개")
    return path


def load_roads(bbox: tuple[float, float, float, float]) -> list[dict]:
    """data/basemap 의 도로 중 bbox 에 걸치는 것. 파일이 없으면 빈 목록."""
    from ui.basemap import BASEMAP_DIR

    s, w, n, e = bbox
    seen, out = set(), []
    for f in sorted(BASEMAP_DIR.glob("*.json")) if BASEMAP_DIR.exists() else []:
        for r in json.loads(f.read_text(encoding="utf-8")).get("roads", []):
            if r["id"] not in seen and any(s <= la <= n and w <= lo <= e for la, lo in r["coords"]):
                seen.add(r["id"])
                out.append(r)
    return out


class Projector:
    """위경도 → SVG 좌표 (북쪽이 위, 가로·세로 축척 같음)."""

    def __init__(self, bbox: tuple[float, float, float, float]):
        s, w, n, e = bbox
        self.s, self.w = s, w
        self.kx = 111_320 * math.cos(math.radians((s + n) / 2))  # m / 경도 1도
        self.ky = 111_320
        width_m, height_m = (e - w) * self.kx, (n - s) * self.ky
        self.px_per_m = SIZE / max(width_m, height_m)
        self.width, self.height = width_m * self.px_per_m, height_m * self.px_per_m

    def __call__(self, lat: float, lon: float) -> tuple[float, float]:
        return ((lon - self.w) * self.kx * self.px_per_m, self.height - (lat - self.s) * self.ky * self.px_per_m)

    def path(self, latlon) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (self(la, lo) for la, lo in latlon))


def render(name: str, route: Route, snapped: list[Snapped], results: list[Holdout],
           track: GpsTrack | None, roads: list[dict], attribution: bool) -> str:
    bbox = route_bbox(route.latlon + [(p.lat, p.lon) for p in snapped])
    pj = Projector(bbox)
    svg: list[str] = []
    for r in roads:
        major = r.get("highway") in MAJOR
        title = html.escape(r.get("name") or r.get("highway", ""))
        svg.append(f'<polyline class="road{" major" if major else ""}" points="{pj.path(r["coords"])}">'
                   f'<title>{title}</title></polyline>')
    svg.append(f'<polyline class="route" points="{pj.path(route.latlon)}"/>')
    x, y = pj(*route.latlon[0])
    svg.append(f'<circle class="start" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>경로 시작</title></circle>')
    if track:
        last = -math.inf
        for p in track.points:
            if p.t_s - last >= DOT_EVERY_S:
                last = p.t_s
                x, y = pj(p.lat, p.lon)
                svg.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="2.2"><title>CSV t_s={p.t_s:.1f}</title></circle>')
    by_t = {h.snapped.anchor.t_s: h for h in results}
    for k, p in enumerate(snapped, 1):
        x, y = pj(p.lat, p.lon)
        sx, sy = pj(*route.point_at(p.s))
        a = p.anchor
        tip = html.escape(f"{k}. {fmt_time(a.t_s)} {a.memo} — 경로에서 {p.snap_m:.1f} m")
        if (h := by_t.get(a.t_s)) is not None:
            hx, hy = pj(h.lat, h.lon)
            svg.append(f'<line class="err" x1="{x:.1f}" y1="{y:.1f}" x2="{hx:.1f}" y2="{hy:.1f}"/>'
                       f'<path class="hold" d="M{hx - 4:.1f},{hy - 4:.1f}l8,8m0,-8l-8,8">'
                       f'<title>{k}번을 빼고 보간: {h.error_m:.1f} m 차이</title></path>')
        svg.append(f'<line class="snap" x1="{x:.1f}" y1="{y:.1f}" x2="{sx:.1f}" y2="{sy:.1f}"/>'
                   f'<circle class="anchor" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>{tip}</title></circle>'
                   f'<text class="label" x="{x + 7:.1f}" y="{y - 7:.1f}">{k}</text>')
    # 축척 막대: 가장 가까운 1·2·5 단위
    want = pj.width / 5 / pj.px_per_m
    unit = 10 ** math.floor(math.log10(want))
    bar_m = max(u * unit for u in (1, 2, 5) if u * unit <= want)
    bar_px = bar_m * pj.px_per_m
    svg.append(f'<g class="scale"><line x1="12" y1="{pj.height - 14:.1f}" x2="{12 + bar_px:.1f}" y2="{pj.height - 14:.1f}"/>'
               f'<text x="12" y="{pj.height - 20:.1f}">{bar_m:.0f} m</text></g>')

    rows = []
    for k, p in enumerate(snapped, 1):
        h = by_t.get(p.anchor.t_s)
        rows.append(f"<tr><td>{k}</td><td>{fmt_time(p.anchor.t_s)}</td><td>{html.escape(p.anchor.memo)}</td>"
                    f"<td>{p.s:.0f}</td><td>{p.snap_m:.1f}</td><td>{'' if h is None else f'{h.error_m:.1f}'}</td></tr>")
    errs = [h.error_m for h in results]
    stats = (f"hold-out 오차 평균 {sum(errs) / len(errs):.1f} m · 최대 {max(errs):.1f} m ({len(errs)}개)"
             if errs else "hold-out 오차: 가운데 기준점이 없어 재지 못함")
    csv_note = f"재구성 CSV {len(track.points)}점, t_s {track.start_s:.1f}~{track.end_s:.1f}" if track else "재구성 CSV 없음"
    road_note = (f"배경 도로 {len(roads)}개 — 지도 데이터 © OpenStreetMap contributors (ODbL)" if attribution
                 else "배경 도로 없음 — --fetch-roads 로 받으면 도로 위에 있는지 볼 수 있다")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} 경로 미리보기</title>
<style>
body {{ margin: 0; padding: 16px; font: 14px/1.5 -apple-system, "Apple SD Gothic Neo", sans-serif; background: #fafafa; color: #222; }}
h1 {{ font-size: 18px; margin: 0 0 4px; }}
.meta {{ color: #555; margin: 0 0 8px; }}
.wrap {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
svg {{ background: #fff; border: 1px solid #ccc; max-width: 100%; height: auto; cursor: grab; touch-action: none; }}
.road {{ fill: none; stroke: #c8c8c8; stroke-width: 1.5; vector-effect: non-scaling-stroke; }}
.road.major {{ stroke: #a9a9a9; stroke-width: 4; }}
.route {{ fill: none; stroke: #1f6fd1; stroke-width: 3; stroke-opacity: .75; vector-effect: non-scaling-stroke; }}
.start {{ fill: #fff; stroke: #1f6fd1; stroke-width: 2; }}
.dot {{ fill: #0a3d7a; }}
.anchor {{ fill: #d62828; stroke: #fff; stroke-width: 1.5; }}
.snap {{ stroke: #d62828; stroke-width: 1; stroke-dasharray: 2 2; vector-effect: non-scaling-stroke; }}
.err {{ stroke: #f08c00; stroke-width: 1; vector-effect: non-scaling-stroke; }}
.hold {{ stroke: #f08c00; stroke-width: 2.5; fill: none; }}
.label {{ font-size: 12px; font-weight: 600; fill: #d62828; paint-order: stroke; stroke: #fff; stroke-width: 3; }}
.scale line {{ stroke: #222; stroke-width: 2; }} .scale text {{ font-size: 12px; fill: #222; }}
table {{ border-collapse: collapse; background: #fff; }}
td, th {{ border: 1px solid #ddd; padding: 2px 8px; text-align: right; }}
td:nth-child(3) {{ text-align: left; }}
.legend span {{ margin-right: 12px; }}
</style></head><body>
<h1>{html.escape(name)} — 재구성 경로 미리보기</h1>
<p class="meta">경로 {route.length_m:.0f} m · 기준점 {len(snapped)}개 · {html.escape(csv_note)} · {html.escape(stats)}<br>
{html.escape(road_note)}</p>
<p class="legend"><span style="color:#1f6fd1">━ 그린 경로</span><span style="color:#0a3d7a">● CSV ({DOT_EVERY_S:.0f}초마다)</span>
<span style="color:#d62828">● 기준점 (점선 = 경로에 붙인 곳)</span><span style="color:#f08c00">✕ 빼고 보간한 위치</span>
<span>휠: 확대 · 끌기: 이동 · 더블클릭: 처음으로</span></p>
<div class="wrap">
<svg id="map" viewBox="0 0 {pj.width:.0f} {pj.height:.0f}" width="{pj.width:.0f}" height="{pj.height:.0f}" xmlns="http://www.w3.org/2000/svg">
{chr(10).join(svg)}
</svg>
<table><tr><th>#</th><th>시각</th><th>설명</th><th>경로 m</th><th>경로까지 m</th><th>hold-out m</th></tr>
{chr(10).join(rows)}
</table>
</div>
<script>
(() => {{
  const svg = document.getElementById('map');
  const full = svg.viewBox.baseVal; const home = [full.x, full.y, full.width, full.height];
  let vb = [...home], drag = null;
  const set = () => svg.setAttribute('viewBox', vb.join(' '));
  const at = (e) => {{ const r = svg.getBoundingClientRect();
    return [vb[0] + (e.clientX - r.left) / r.width * vb[2], vb[1] + (e.clientY - r.top) / r.height * vb[3]]; }};
  svg.addEventListener('wheel', (e) => {{ e.preventDefault(); const [x, y] = at(e); const k = e.deltaY < 0 ? 0.8 : 1.25;
    vb = [x - (x - vb[0]) * k, y - (y - vb[1]) * k, vb[2] * k, vb[3] * k]; set(); }}, {{ passive: false }});
  svg.addEventListener('pointerdown', (e) => {{ drag = [e.clientX, e.clientY, vb[0], vb[1]]; svg.setPointerCapture(e.pointerId); }});
  svg.addEventListener('pointermove', (e) => {{ if (!drag) return; const r = svg.getBoundingClientRect();
    vb[0] = drag[2] - (e.clientX - drag[0]) / r.width * vb[2]; vb[1] = drag[3] - (e.clientY - drag[1]) / r.height * vb[3]; set(); }});
  svg.addEventListener('pointerup', () => {{ drag = null; }});
  svg.addEventListener('dblclick', () => {{ vb = [...home]; set(); }});
}})();
</script>
</body></html>
"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--gps", help="재구성 CSV (기본 data/videos/<video_id>.gps.csv, 없으면 경로·기준점만)")
    ap.add_argument("--name", help="제목·도로 파일 이름 (기본 video_id)")
    ap.add_argument("--out", help="출력 HTML (기본 data/videos/<video_id>.preview.html)")
    ap.add_argument("--fetch-roads", action="store_true", help="경로 주변 도로를 OSM 에서 받는다 (인터넷 필요)")
    args = ap.parse_args(argv)
    gpx, anchors = resolve_inputs(ap, args)
    name = args.name or args.video_id or gpx.stem
    out = Path(args.out) if args.out else VIDEO_DIR / f"{name}.preview.html"
    gps = Path(args.gps) if args.gps else (VIDEO_DIR / f"{args.video_id}.gps.csv" if args.video_id else None)

    try:
        route = Route(load_gpx(gpx))
        snapped = snap_anchors(route, load_anchors(anchors, args.video_id), args.max_snap_m)
    except AnchorError as e:
        sys.exit(f"오류: {e}")
    track = GpsTrack.from_csv(gps) if gps and gps.exists() else None
    bbox = route_bbox(route.latlon + [(p.lat, p.lon) for p in snapped])
    if args.fetch_roads:
        fetch_roads(route_bbox(route.latlon, 400), name)
    roads = load_roads(bbox)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(name, route, snapped, holdout(route, snapped), track, roads, bool(roads)), encoding="utf-8")
    print(f"{out}: 도로 {len(roads)}개, 기준점 {len(snapped)}개{'' if track else ', 재구성 CSV 없음'}")
    print(f"  열기: open {out}")


if __name__ == "__main__":
    main()
