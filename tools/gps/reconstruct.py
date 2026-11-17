"""GPS 가 없는 공개 영상의 경로 재구성 (PRD 8.5, docs/plan.md 0.7).

입력 두 개로 가상 블랙박스가 읽는 GPS CSV(`t_s,lat,lon,speed_kmh`, contracts 의 GpsTrack)를 만든다.

  routes/<video_id>.gpx    gpx.studio 에서 도로를 따라 그린 경로 (주행 방향 순서)
  anchors/<video_id>.csv   기준점: 영상 시각과 그 순간 위치 (video_id,t_s,lat,lon,memo)

기준점 위치는 경로 위 가장 가까운 점으로 붙이고, 기준점 사이는 경로를 따라 **거리 비례**로
(= 그 구간을 같은 속도로 달렸다고 보고) 보간한다. 결과는 첫 기준점부터 마지막 기준점까지만 있다.

  t_s 는 원본 영상 시각. 초(1250.5)나 분:초(20:50.5), 시:분:초(0:20:50) 로 쓴다.
  lat,lon 을 비우면 "앞 기준점과 같은 곳" = 신호 대기처럼 멈춰 있던 구간의 끝.
  기준점이 경로 순서와 어긋나거나, 시간이 거꾸로이거나, 경로에서 멀면(--max-snap-m) 오류.

  uv run python -m tools.gps.reconstruct gn_a                       # → data/videos/gn_a.gps.csv
  uv run python -m tools.gps.reconstruct gn_a --clip-start 1247.646 --out data/videos/gn_a_2050-2510.gps.csv  # fetch_log 값
  uv run python -m tools.gps.reconstruct --gpx a.gpx --anchors a.csv --out a.gps.csv

--clip-start: 잘라 낸 클립의 t=0 이 원본의 몇 초인지 (tools/video/fetch.py 의 fetch_log.jsonl
clip_start_in_source_s). 모든 시각에서 빼고, 클립 시작 전 점은 버린다.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from daboyeo_contracts import GpsTrack
from daboyeo_contracts.gps import EARTH_RADIUS_M, GpsPoint

ROOT = Path(__file__).resolve().parents[2]
GPS_DIR = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "videos"
MAX_SNAP_M = 50.0  # 기준점이 경로에서 이보다 멀면 잘못 찍은 것으로 본다
ORDER_TOL_M = 5.0  # 뒤쪽 경로가 앞쪽보다 이만큼 이상 가까우면 순서 어긋남으로 본다
FAST_KMH = 120.0  # 구간 평균 속도가 이보다 빠르면 경고 (시내 도로)


class AnchorError(ValueError):
    """기준점·경로 입력이 잘못됨. 메시지는 사람이 고칠 수 있게 쓴다."""


def parse_time(text: str) -> float:
    """'1250.5', '20:50.5', '0:20:50' → 초."""
    text = text.strip()
    if not text:
        raise ValueError("시각이 비어 있다")
    sec = 0.0
    for part in text.split(":"):
        sec = sec * 60 + float(part)
    return sec


def fmt_time(t_s: float) -> str:
    m, s = divmod(t_s, 60)
    return f"{int(m)}:{s:04.1f}"


# ---------- 입력 ----------

def load_gpx(path: str | Path) -> list[tuple[float, float]]:
    """GPX 의 trkpt(없으면 rtept) 를 파일 순서대로. 연달아 같은 점은 하나로."""
    root = ET.parse(path).getroot()
    pts: list[tuple[float, float]] = []
    for tag in ("trkpt", "rtept"):
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1] == tag:
                p = (float(el.get("lat")), float(el.get("lon")))
                if not pts or pts[-1] != p:
                    pts.append(p)
        if pts:
            break
    if len(pts) < 2:
        raise AnchorError(f"{path}: 경로 점이 2개 미만이다 (gpx.studio 에서 경로를 그려 내보낸다)")
    return pts


@dataclass(frozen=True, slots=True)
class Anchor:
    t_s: float
    lat: float | None  # None: 앞 기준점과 같은 곳 (정지)
    lon: float | None
    memo: str = ""


def memo_of(row: dict) -> str:
    """memo 에 따옴표 없이 쉼표가 들어가 열이 넘치면 넘친 부분도 memo 로 이어 붙인다."""
    parts = [row.get("memo") or "", *(row.get(None) or [])]
    return ",".join(parts).strip()


def load_anchors(path: str | Path, video_id: str | None = None) -> list[Anchor]:
    """기준점 CSV (video_id,t_s,lat,lon,memo — video_id·memo 는 없어도 된다). 파일 순서 그대로.
    video_id 열이 있고 값이 있으면 주어진 video_id 의 행만. '#' 으로 시작하는 줄은 건너뛴다."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    missing = [c for c in ("t_s", "lat", "lon") if c not in (reader.fieldnames or [])]
    if missing:
        raise AnchorError(f"{path}: 열 {missing} 이 없다 (video_id,t_s,lat,lon,memo)")
    out = []
    for n, row in enumerate(reader, start=2):
        vid = (row.get("video_id") or "").strip()
        if video_id and vid and vid != video_id:
            continue
        lat, lon = (row.get("lat") or "").strip(), (row.get("lon") or "").strip()
        if bool(lat) != bool(lon):
            raise AnchorError(f"{path}:{n}: lat·lon 은 둘 다 쓰거나 둘 다 비운다")
        try:
            out.append(Anchor(parse_time(row["t_s"]), float(lat) if lat else None,
                              float(lon) if lon else None, memo_of(row)))
        except ValueError as e:
            raise AnchorError(f"{path}:{n}: {e}") from None
    return out


# ---------- 경로 ----------

class Route:
    """위경도 꺾은선. 짧은 구간(수 km)이라 한 점 기준 평면(m)으로 펴서 계산한다."""

    def __init__(self, latlon: list[tuple[float, float]]):
        if len(latlon) < 2:
            raise AnchorError("경로 점이 2개 미만이다")
        self.latlon = latlon
        self._lat0 = math.radians(sum(p[0] for p in latlon) / len(latlon))
        self._ref = latlon[0]
        self.xy = [self._to_xy(la, lo) for la, lo in latlon]
        self.cum = [0.0]
        for (x1, y1), (x2, y2) in pairwise(self.xy):
            self.cum.append(self.cum[-1] + math.hypot(x2 - x1, y2 - y1))

    @property
    def length_m(self) -> float:
        return self.cum[-1]

    def _to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (math.radians(lon - self._ref[1]) * EARTH_RADIUS_M * math.cos(self._lat0),
                math.radians(lat - self._ref[0]) * EARTH_RADIUS_M)

    def _to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return (self._ref[0] + math.degrees(y / EARTH_RADIUS_M),
                self._ref[1] + math.degrees(x / (EARTH_RADIUS_M * math.cos(self._lat0))))

    def locate(self, lat: float, lon: float, min_s: float = 0.0) -> tuple[float, float]:
        """경로 위(거리 min_s 이후)에서 가장 가까운 점의 (출발점부터 거리 m, 떨어진 거리 m)."""
        px, py = self._to_xy(lat, lon)
        best = (math.inf, 0.0)
        for i, ((x1, y1), (x2, y2)) in enumerate(pairwise(self.xy)):
            s1, seg = self.cum[i], self.cum[i + 1] - self.cum[i]
            if self.cum[i + 1] < min_s or seg == 0:
                continue
            r = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / seg**2
            r = min(max(r, (min_s - s1) / seg, 0.0), 1.0)
            d = math.hypot(x1 + (x2 - x1) * r - px, y1 + (y2 - y1) * r - py)
            if d < best[0]:
                best = (d, s1 + seg * r)
        return best[1], best[0]

    def point_at(self, s: float) -> tuple[float, float]:
        """출발점부터 경로를 따라 s m 간 곳의 (lat, lon)."""
        s = min(max(s, 0.0), self.length_m)
        i = min(bisect_right(self.cum, s), len(self.cum) - 1)
        s1, s2 = self.cum[i - 1], self.cum[i]
        r = 0.0 if s2 == s1 else (s - s1) / (s2 - s1)
        (x1, y1), (x2, y2) = self.xy[i - 1], self.xy[i]
        return self._to_latlon(x1 + (x2 - x1) * r, y1 + (y2 - y1) * r)


@dataclass(frozen=True, slots=True)
class Snapped:
    anchor: Anchor
    lat: float  # 기준점이 가리키는 위치 (비웠으면 앞 기준점 위치)
    lon: float
    s: float  # 경로 위 거리 (m)
    snap_m: float  # 기준점이 경로에서 떨어진 거리 (m)


def snap_anchors(route: Route, anchors: list[Anchor], max_snap_m: float = MAX_SNAP_M) -> list[Snapped]:
    """기준점을 경로 위에 붙인다. 순서·시간·거리가 맞지 않으면 AnchorError."""
    if len(anchors) < 2:
        raise AnchorError(f"기준점이 2개 이상 있어야 한다 (지금 {len(anchors)}개)")
    out: list[Snapped] = []
    for k, a in enumerate(anchors):
        name = f"기준점 {k + 1} (t={fmt_time(a.t_s)} {a.memo})".rstrip()
        if out and a.t_s <= out[-1].anchor.t_s:
            raise AnchorError(f"{name}: 시각이 앞 기준점({fmt_time(out[-1].anchor.t_s)})보다 늦지 않다. 시간 순으로 적는다")
        if a.lat is None:
            if not out:
                raise AnchorError(f"{name}: 첫 기준점은 위치를 비울 수 없다")
            p = out[-1]
            out.append(Snapped(a, p.lat, p.lon, p.s, p.snap_m))
            continue
        prev_s = out[-1].s if out else 0.0
        s, d = route.locate(a.lat, a.lon, min_s=prev_s)
        s_any, d_any = route.locate(a.lat, a.lon)
        if s_any < prev_s and d - d_any > ORDER_TOL_M:
            hint = " 경로를 주행 반대 방향으로 그렸는지 확인한다" if k == 1 else ""
            raise AnchorError(
                f"{name}: 경로 순서와 어긋난다 — 앞 기준점보다 {prev_s - s_any:.0f} m 뒤에 있다.{hint}")
        if d > max_snap_m:
            raise AnchorError(f"{name}: 경로에서 {d:.0f} m 떨어져 있다 (한도 {max_snap_m:.0f} m). 위치나 경로를 고친다")
        out.append(Snapped(a, a.lat, a.lon, s, d))
    return out


# ---------- 보간 ----------

def s_at(snapped: list[Snapped], t_s: float) -> tuple[float, float]:
    """시각 t_s 의 (경로 위 거리 m, 그 구간 평균 속도 km/h). 첫·마지막 기준점 밖이면 ValueError."""
    if not snapped[0].anchor.t_s <= t_s <= snapped[-1].anchor.t_s:
        raise ValueError(f"t_s={t_s} 가 기준점 범위 밖이다")
    times = [p.anchor.t_s for p in snapped]
    i = min(max(bisect_right(times, t_s), 1), len(snapped) - 1)
    a, b = snapped[i - 1], snapped[i]
    dt = b.anchor.t_s - a.anchor.t_s
    return a.s + (b.s - a.s) * (t_s - a.anchor.t_s) / dt, (b.s - a.s) / dt * 3.6


def position_at(route: Route, snapped: list[Snapped], t_s: float) -> tuple[float, float]:
    return route.point_at(s_at(snapped, t_s)[0])


def interpolate(route: Route, snapped: list[Snapped], step_s: float = 0.5, clip_start_s: float = 0.0) -> GpsTrack:
    """첫 기준점부터 마지막 기준점까지 step_s 간격 (마지막 기준점 시각도 넣는다). 시각은 clip_start_s 를 뺀 값."""
    if step_s <= 0:
        raise ValueError("step_s 는 0 보다 커야 한다")
    t0, t1 = snapped[0].anchor.t_s, snapped[-1].anchor.t_s
    times = [t0 + k * step_s for k in range(int((t1 - t0) / step_s + 1e-9) + 1)]
    if t1 - times[-1] > 1e-6:
        times.append(t1)
    points = []
    for t in times:
        if t - clip_start_s < 0:
            continue
        s, v = s_at(snapped, t)
        lat, lon = route.point_at(s)
        points.append(GpsPoint(round(t - clip_start_s, 3), lat, lon, v))
    return GpsTrack(points)


def reconstruct(gpx: str | Path, anchors: str | Path, video_id: str | None = None, *,
                step_s: float = 0.5, clip_start_s: float = 0.0, max_snap_m: float = MAX_SNAP_M,
                ) -> tuple[GpsTrack, Route, list[Snapped]]:
    route = Route(load_gpx(gpx))
    snapped = snap_anchors(route, load_anchors(anchors, video_id), max_snap_m)
    return interpolate(route, snapped, step_s, clip_start_s), route, snapped


def default_paths(video_id: str) -> tuple[Path, Path]:
    return GPS_DIR / "routes" / f"{video_id}.gpx", GPS_DIR / "anchors" / f"{video_id}.csv"


def add_input_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("video_id", nargs="?", help="tools/gps/routes/<id>.gpx 와 anchors/<id>.csv 를 쓴다")
    ap.add_argument("--gpx", help="경로 GPX (video_id 대신)")
    ap.add_argument("--anchors", help="기준점 CSV (video_id 대신)")
    ap.add_argument("--max-snap-m", type=float, default=MAX_SNAP_M, help="기준점-경로 거리 한도 (m)")


def resolve_inputs(ap: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[Path, Path]:
    if args.video_id:
        gpx, anchors = default_paths(args.video_id)
    else:
        gpx, anchors = None, None
    gpx = Path(args.gpx) if args.gpx else gpx
    anchors = Path(args.anchors) if args.anchors else anchors
    if gpx is None or anchors is None:
        ap.error("video_id 나 --gpx·--anchors 를 준다")
    return gpx, anchors


def summary(route: Route, snapped: list[Snapped]) -> list[str]:
    span = f"{fmt_time(snapped[0].anchor.t_s)} ~ {fmt_time(snapped[-1].anchor.t_s)}"
    lines = [f"경로 {route.length_m:.0f} m, 기준점 {len(snapped)}개, {span}"]
    for a, b in pairwise(snapped):
        v = (b.s - a.s) / (b.anchor.t_s - a.anchor.t_s) * 3.6
        warn = "  ← 너무 빠름: 기준점 시각·위치 확인 (영상 편집으로 건너뛴 곳이면 그대로 둔다)" if v > FAST_KMH else ""
        lines.append(f"  {fmt_time(a.anchor.t_s):>8} → {fmt_time(b.anchor.t_s):>8}  {b.s - a.s:6.0f} m  "
                     f"{v:5.1f} km/h  {b.anchor.memo}{warn}")
    far = [p for p in snapped if p.snap_m > 15]
    for p in far:
        lines.append(f"  주의: {fmt_time(p.anchor.t_s)} {p.anchor.memo} 기준점이 경로에서 {p.snap_m:.0f} m")
    return lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(ap)
    ap.add_argument("--out", help="출력 CSV (기본 data/videos/<video_id>.gps.csv)")
    ap.add_argument("--step", type=float, default=0.5, help="출력 간격 (초)")
    ap.add_argument("--clip-start", default="0", help="클립 t=0 의 원본 시각 (초 또는 분:초)")
    args = ap.parse_args(argv)
    gpx, anchors = resolve_inputs(ap, args)
    out = Path(args.out) if args.out else (VIDEO_DIR / f"{args.video_id}.gps.csv" if args.video_id else None)
    if out is None:
        ap.error("--gpx·--anchors 를 쓰면 --out 도 준다")
    try:
        track, route, snapped = reconstruct(gpx, anchors, args.video_id, step_s=args.step,
                                            clip_start_s=parse_time(args.clip_start), max_snap_m=args.max_snap_m)
    except AnchorError as e:
        sys.exit(f"오류: {e}")
    out.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(out)
    print("\n".join(summary(route, snapped)))
    print(f"{out}: {len(track.points)}점, t_s {track.start_s:.1f} ~ {track.end_s:.1f}")


if __name__ == "__main__":
    main()
