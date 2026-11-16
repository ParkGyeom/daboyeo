"""재구성 GPS 오차 재기 (docs/plan.md 0.8, AC-13).

기준점을 하나씩 빼고 나머지로 보간한 위치와, 뺀 기준점의 실제 위치 사이 거리(m)를 잰다.
첫·마지막 기준점은 빼면 보간할 수 없어 재지 않는다. 위치를 비운 기준점(정지)은 앞 기준점 위치가 정답이다.

멈춤의 시작·끝이나 편집 컷 양쪽 기준점을 빼면 그 멈춤·건너뜀을 모른 채 보간해서 크게 나오므로,
그것을 뺀 평균(mean_moving_m)도 같이 쓴다. 편집 컷 = 기준점 사이 평균 속도가 JUMP_KMH 를 넘는 구간.
기준점 사이가 두 배로 벌어진 상태에서 재므로, 모든 기준점을 쓴 실제 재구성 오차보다 크게 나온다 (보수적).
결과는 영상마다 한 줄씩 experiments/gps_error.csv 에 쓴다 (같은 video_id 줄은 바꾸고 다른 줄은 둔다).

  uv run python -m tools.gps.holdout_error gn_a gn_b
  uv run python -m tools.gps.holdout_error --gpx a.gpx --anchors a.csv --name synth --out /tmp/gps_error.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from daboyeo_contracts.gps import haversine_m

from tools.gps.reconstruct import (
    MAX_SNAP_M,
    ROOT,
    AnchorError,
    Route,
    Snapped,
    default_paths,
    fmt_time,
    load_anchors,
    load_gpx,
    position_at,
    snap_anchors,
)

OUT = ROOT / "experiments" / "gps_error.csv"
JUMP_KMH = 150.0  # 기준점 사이가 이보다 빠르면 영상 편집으로 건너뛴 구간으로 본다
COLUMNS = ["video_id", "n_anchors", "n_holdout", "mean_m", "median_m", "max_m", "max_at", "route_m",
           "n_moving", "mean_moving_m"]


@dataclass(frozen=True, slots=True)
class Holdout:
    snapped: Snapped
    lat: float  # 빼고 보간한 위치
    lon: float
    error_m: float
    stop_edge: bool  # 멈춤의 시작·끝 또는 편집 컷 양쪽 기준점 (빼면 그것을 모른 채 보간해서 오차가 크게 나온다)


def holdout(route: Route, snapped: list[Snapped]) -> list[Holdout]:
    """가운데 기준점마다: 그것만 빼고 보간한 위치와 기준점 위치의 거리."""
    # 편집 컷: 바로 앞 기준점과의 평균 속도가 JUMP_KMH 보다 빠른 기준점 k (k-1 과 k 사이를 영상이 건너뜀)
    jumps = {k for k in range(1, len(snapped))
             if (snapped[k].s - snapped[k - 1].s) / (snapped[k].anchor.t_s - snapped[k - 1].anchor.t_s) * 3.6 > JUMP_KMH}
    jumps |= {k - 1 for k in jumps}
    out = []
    for k in range(1, len(snapped) - 1):
        rest = snapped[:k] + snapped[k + 1:]
        p = snapped[k]
        lat, lon = position_at(route, rest, p.anchor.t_s)
        edge = p.anchor.lat is None or snapped[k + 1].anchor.lat is None or k in jumps
        out.append(Holdout(p, lat, lon, haversine_m(lat, lon, p.lat, p.lon), edge))
    return out


def summarize(video_id: str, route: Route, snapped: list[Snapped], results: list[Holdout]) -> dict:
    errs = [h.error_m for h in results]
    moving = [h.error_m for h in results if not h.stop_edge]
    worst = max(results, key=lambda h: h.error_m, default=None)
    return {
        "video_id": video_id,
        "n_anchors": len(snapped),
        "n_holdout": len(results),
        "mean_m": f"{statistics.fmean(errs):.1f}" if errs else "",
        "median_m": f"{statistics.median(errs):.1f}" if errs else "",
        "max_m": f"{worst.error_m:.1f}" if worst else "",
        "max_at": f"{fmt_time(worst.snapped.anchor.t_s)} {worst.snapped.anchor.memo.split(' (')[0]}".strip() if worst else "",
        "route_m": f"{route.length_m:.0f}",
        "n_moving": len(moving),
        "mean_moving_m": f"{statistics.fmean(moving):.1f}" if moving else "",
    }


def evaluate(gpx: str | Path, anchors: str | Path, video_id: str | None = None,
             max_snap_m: float = MAX_SNAP_M) -> tuple[Route, list[Snapped], list[Holdout]]:
    route = Route(load_gpx(gpx))
    snapped = snap_anchors(route, load_anchors(anchors, video_id), max_snap_m)
    return route, snapped, holdout(route, snapped)


def write_rows(path: Path, rows: list[dict]) -> None:
    """같은 video_id 줄은 새 값으로 바꾸고, 나머지 줄은 그대로 둔다."""
    old = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
    new_ids = {r["video_id"] for r in rows}
    merged = [r for r in old if r.get("video_id") not in new_ids] + rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_ids", nargs="*", help="tools/gps/routes/<id>.gpx 와 anchors/<id>.csv 를 쓴다")
    ap.add_argument("--gpx", help="경로 GPX (video_id 대신, 한 개)")
    ap.add_argument("--anchors", help="기준점 CSV (video_id 대신, 한 개)")
    ap.add_argument("--name", help="--gpx 로 줄 때 표에 쓸 video_id")
    ap.add_argument("--max-snap-m", type=float, default=MAX_SNAP_M)
    ap.add_argument("--out", default=str(OUT), help=f"결과 CSV (기본 {OUT.relative_to(ROOT)})")
    args = ap.parse_args(argv)

    jobs = [(vid, *default_paths(vid)) for vid in args.video_ids]
    if args.gpx or args.anchors:
        if not (args.gpx and args.anchors and args.name):
            ap.error("--gpx·--anchors·--name 을 함께 준다")
        jobs.append((args.name, Path(args.gpx), Path(args.anchors)))
    if not jobs:
        ap.error("video_id 나 --gpx·--anchors·--name 을 준다")

    rows = []
    for vid, gpx, anchors in jobs:
        try:
            route, snapped, results = evaluate(gpx, anchors, vid, args.max_snap_m)
        except (AnchorError, FileNotFoundError) as e:
            sys.exit(f"[{vid}] 오류: {e}")
        print(f"[{vid}] 경로 {route.length_m:.0f} m, 기준점 {len(snapped)}개")
        for h in results:
            edge = "  (멈춤·컷 경계)" if h.stop_edge else ""
            print(f"  {fmt_time(h.snapped.anchor.t_s):>8}  {h.error_m:6.1f} m  {h.snapped.anchor.memo}{edge}")
        row = summarize(vid, route, snapped, results)
        print(f"  평균 {row['mean_m']} m, 중앙값 {row['median_m']} m, 최대 {row['max_m']} m ({row['max_at']})")
        print(f"  멈춤·컷 경계를 뺀 {row['n_moving']}개 평균 {row['mean_moving_m'] or '-'} m")
        if len(snapped) < 8:
            print(f"  주의: 기준점 {len(snapped)}개 — 영상당 8개 이상을 권장한다 (tools/gps/README.md)")
        rows.append(row)
    write_rows(Path(args.out), rows)
    print(f"{args.out}: {len(rows)}개 영상")


if __name__ == "__main__":
    main()
