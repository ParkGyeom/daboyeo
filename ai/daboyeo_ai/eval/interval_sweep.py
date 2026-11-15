"""분석 간격 ↔ 결함 누락률 (계획 4.3, PRD 12장 '분석 간격' 행 그래프).

    uv run python -m daboyeo_ai.eval.interval_sweep data/videos/v1.mp4 data/videos/v2.mp4 \\
        --model ai/models/detector_v2.onnx --cache-dir data/eval_cache
    # 그래프(PNG)까지: matplotlib 은 의존성에 없어서 실행할 때만 붙인다
    uv run --with matplotlib python -m daboyeo_ai.eval.interval_sweep ... (같은 인자)

- 탐지기·차선 측정을 **모든 프레임에 한 번** 돌려 캐시하고(`--cache-dir`, 같은 캐시를 `eval_detector` 도 쓴다),
  간격 N(기본 1/3/5/10/15 프레임)마다 프레임 번호가 N 의 배수인 것만 골라 다시 잰다. 그래서 간격을 늘려도 싸다.
- 누락률: 정답 이벤트 창 `[t_start − tol, t_end + tol]` 안의 **분석한 프레임** 중 같은 유형 탐지(conf 기준 이상)가
  하나도 없으면 놓침. 차선 마모는 간격마다 새 `LaneWearTracker` 에 고른 프레임만 넣어 `confirmed` 가 창 안에 있어야 찾음
  (간격이 길면 5 중 3 투표가 모이는 데 시간이 더 걸린다 — 그것도 누락률에 들어간다).
- 비용: 분석한 프레임 수 (`frames_analyzed`), 영상 1분당 분석 프레임 수. 실제 ms 는 `bench_cpu`(4.2) 결과를 곱한다.
- conf 기준은 하나 (`--conf`, 기본 0.25) 또는 클래스별 `--conf-json`.
- 출력: `<out>/sweep.csv` (+ matplotlib 가 있으면 `<out>/sweep.png`). 기본 `experiments/interval_sweep/`.
- 시설물(guardrail_damage·sign_damage)은 위치 찾기로만 재고 결함 합계(`pothole+crack+lane_wear`)에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .eval_detector import write_csv
from .video_events import (
    DEFECT_TYPES,
    EVAL_TYPES,
    TypeCounts,
    VideoRecords,
    evaluate_video,
    find_gt,
    fmt,
    lane_wear_hits,
    load_conf_json,
    load_gt,
    scan_or_load,
    subsample,
    type_group,
)

DEFAULT_INTERVALS = (1, 3, 5, 10, 15)
DEFECTS_TOTAL = "+".join(DEFECT_TYPES)  # pothole+crack+lane_wear
FIELDS = ["interval", "group", "defect_type", "n_gt", "n_missed", "miss_rate", "n_pred", "precision",
          "frames_analyzed", "frames_total", "analyzed_per_min", "missed_ids"]


def sweep(
    videos: Sequence[VideoRecords],
    gts: dict[str, list],
    intervals: Sequence[int],
    conf: float | dict[str, float],
    tol_s: float = 0.5,
    merge_gap_s: float = 1.0,
    lane: bool = True,
) -> list[dict]:
    """간격마다 전체 영상 합의 유형별 누락률. videos 의 기록은 every=1 이어야 한다 (간격의 약수면 된다)."""
    rows = []
    frames_total = sum(round(v.duration_s * v.fps) for v in videos)
    minutes = sum(v.duration_s for v in videos) / 60
    for n in intervals:
        bad = [v.video_id for v in videos if n % v.every]
        if bad:
            raise ValueError(f"간격 {n} 은 기록 간격의 배수가 아니다: {bad}")
        total: dict[str, TypeCounts] = {}
        analyzed = 0
        for v in videos:
            recs = subsample(v.records, n)
            analyzed += len(recs)
            per = evaluate_video(v.video_id, recs, gts[v.video_id], conf, v.duration_s, tol_s, merge_gap_s,
                                 lane_hits=lane_wear_hits(recs) if lane else None)
            for d, c in per.items():
                total.setdefault(d, TypeCounts()).add(c)
        defects = TypeCounts()
        for d in DEFECT_TYPES:
            if d in total:
                defects.add(total[d])
        for d, c in [(d, total[d]) for d in EVAL_TYPES if d in total] + [(DEFECTS_TOTAL, defects)]:
            rows.append({
                "interval": n, "group": "defect" if d == DEFECTS_TOTAL else type_group(d), "defect_type": d,
                "n_gt": c.n_gt, "n_missed": c.n_gt - c.n_gt_found, "miss_rate": fmt(c.miss_rate), "n_pred": c.n_pred,
                "precision": fmt(c.precision), "frames_analyzed": analyzed, "frames_total": frames_total,
                "analyzed_per_min": fmt(analyzed / minutes if minutes else None, 1), "missed_ids": " ".join(c.missed),
            })
    return rows


def plot(rows: Sequence[dict], path: Path) -> bool:
    """누락률(유형별 선) + 1분당 분석 프레임(오른쪽 축). matplotlib 이 없으면 False."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    intervals = sorted({r["interval"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    types = [t for t in [*DEFECT_TYPES, DEFECTS_TOTAL] if any(r["defect_type"] == t and r["n_gt"] for r in rows)]
    for t in types:
        pts = [(r["interval"], float(r["miss_rate"])) for r in rows if r["defect_type"] == t and r["miss_rate"] != ""]
        lw, style = (2.5, "-") if t == DEFECTS_TOTAL else (1.5, "--")
        ax.plot(*zip(*pts), style, marker="o", lw=lw, label=t)
    ax.set_xlabel("analysis interval (frames)")
    ax.set_ylabel("GT event miss rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(intervals)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    cost = sorted({(r["interval"], float(r["analyzed_per_min"])) for r in rows if r["analyzed_per_min"] != ""})
    if cost:
        ax2.plot(*zip(*cost), ":", color="gray", marker="s", label="frames analyzed / min")
        ax2.set_ylabel("frames analyzed per video minute")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax.set_title("Analysis interval vs miss rate")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: Sequence[str] | None = None, detector_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=Path("ai/models/detector_v2.onnx"))
    ap.add_argument("--gt-dir", type=Path, default=Path("experiments/gt_events"))
    ap.add_argument("--intervals", default=",".join(map(str, DEFAULT_INTERVALS)), help="쉼표로 구분 (기본 1,3,5,10,15)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-json", type=Path, help="클래스별 conf JSON (주면 --conf 대신)")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--point-half", type=float, default=1.0)
    ap.add_argument("--no-lane", action="store_true")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache-dir", type=Path, help="프레임별 탐지 캐시 (예: data/eval_cache). 두 번째 실행부터 빠르다")
    ap.add_argument("--out", type=Path, default=Path("experiments/interval_sweep"))
    a = ap.parse_args(argv)
    intervals = sorted({int(x) for x in a.intervals.split(",") if x.strip()})
    if not intervals or intervals[0] < 1:
        ap.error("--intervals 는 1 이상의 정수")

    conf: float | dict[str, float] = load_conf_json(a.conf_json) if a.conf_json else a.conf
    floor = min(conf.values()) if isinstance(conf, dict) else conf
    lane = not a.no_lane
    if detector_factory is None:
        def detector_factory(c: float):
            from daboyeo_ai.inference import OnnxDetector

            return OnnxDetector(a.model, threads=a.threads, conf=c)

    videos, gts = [], {}
    for v in a.videos:
        gt_path = find_gt(v, a.gt_dir)
        if gt_path is None:
            print(f"[{v.stem}] 정답 {a.gt_dir / (v.stem + '.csv')} 이 없어 건너뜀", file=sys.stderr)
            continue
        vr = scan_or_load(v, 1, floor, detector_factory, lane, a.cache_dir, a.fps, a.model.name)
        videos.append(vr)
        gts[vr.video_id] = load_gt(gt_path, vr.video_id, a.point_half)
    if not videos:
        print("평가할 영상이 없다 (정답 CSV 확인)", file=sys.stderr)
        return 1

    rows = sweep(videos, gts, intervals, conf, a.tol, a.merge_gap, lane)
    write_csv(a.out / "sweep.csv", rows, FIELDS)
    print(f"{'간격':>4} {'type':>24} {'GT':>4} {'놓침':>4} {'누락률':>6} {'분석/분':>8}")
    for r in rows:
        print(f"{r['interval']:>4} {r['defect_type']:>24} {r['n_gt']:>4} {r['n_missed']:>4} {r['miss_rate']:>6} "
              f"{r['analyzed_per_min']:>8}")
    print(f"\n저장: {a.out / 'sweep.csv'}")
    if plot(rows, a.out / "sweep.png"):
        print(f"그래프: {a.out / 'sweep.png'}")
    else:
        print("matplotlib 이 없어 그래프는 건너뜀 — `uv run --with matplotlib python -m daboyeo_ai.eval.interval_sweep ...`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
