"""실제 주행 영상에서 탐지 성능 재기 — 이벤트 단위 정밀도·재현율·F1 (계획 4.1, PRD 12장 탐지 행).

    uv run python -m daboyeo_ai.eval.eval_detector data/videos/v1.mp4 data/videos/v2.mp4 \\
        --model ai/models/detector_v2.onnx --every 3 --cache-dir data/eval_cache

- 정답: `experiments/gt_events/<영상 파일 이름>.csv` (`--gt-dir`). 형식은 `video_events` 모듈 설명 · `docs/ai-handoff.md` 12장.
  정답 파일이 없는 영상은 건너뛴다.
- 탐지기를 `--every` 프레임마다 돌리고 (가장 낮은 conf 로 한 번만), conf 기준마다 다시 걸러 잰다.
  `--confs 0.25,0.3,...` 와 `--conf-json`(클래스별 conf, 다른 세션이 만드는 파일) 을 함께 쓸 수 있다.
- 차선 마모: 같은 프레임에 `LaneWearMeter` + 새 `LaneWearTracker` 를 돌려 `confirmed` 를 lane_wear 탐지로 쓴다
  (conf 기준과 무관 — conf 열은 `vote`). `--no-lane` 으로 끈다.
- 시설물(guardrail_damage·sign_damage): 파손 판정을 안 하므로 (PRD v0.5) **위치 찾기**(facility_presence) 로만 재고
  결함 합계(`pothole+crack`)에 넣지 않는다. 라벨러가 보이는 시설물을 모두 적지 않았다면 정밀도는 낮게 나온다.
- 출력 (요약만): `<out>/summary.csv`, `<out>/summary.json` (기본 `experiments/eval_detector/`).
  summary.json 에는 공개 테스트셋 지표(`<model>.json` 의 `metrics.test`) 를 같이 넣어 12장 표에 나란히 쓴다.
- `--cache-dir` 를 주면 프레임별 탐지를 `<cache-dir>/<영상>.jsonl` 로 남겨 다음 실행·`interval_sweep` 이 다시 쓴다.
  캐시는 크고 영상에서 나온 것이라 `data/` 아래(git 제외)에 둔다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .video_events import (
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
    type_group,
)

DEFAULT_CONFS = (0.25, 0.3, 0.35, 0.4, 0.5, 0.6)
LANE = "lane_wear"
DETECTOR_DEFECTS = ("pothole", "crack")  # 합계 행 pothole+crack (차선 마모는 conf 와 무관해 따로)
FIELDS = ["scope", "conf", "group", "defect_type", "n_gt", "n_gt_found", "n_pred", "n_pred_ok",
          "precision", "recall", "f1", "miss_rate", "pred_per_min", "minutes", "missed_ids"]


def evaluate(
    videos: Sequence[VideoRecords],
    gts: dict[str, list],
    confs: Sequence[tuple[str, float | dict[str, float]]],
    tol_s: float = 0.5,
    merge_gap_s: float = 1.0,
    lane: bool = True,
) -> list[dict]:
    """영상별·전체 행. confs = [(이름, conf 또는 클래스별 dict)]."""
    rows: list[dict] = []
    lane_cache = {v.video_id: lane_wear_hits(v.records) for v in videos} if lane else {}
    for name, conf in confs:
        total: dict[str, TypeCounts] = {}
        for v in videos:
            per = evaluate_video(v.video_id, v.records, gts[v.video_id], conf, v.duration_s, tol_s, merge_gap_s,
                                 lane_hits=None)
            for dtype, c in per.items():
                rows.append(_row(v.video_id, name, dtype, c))
                total.setdefault(dtype, TypeCounts()).add(c)
        rows.extend(_with_totals("all", name, total))
    if lane:
        total = {}
        for v in videos:
            per = evaluate_video(v.video_id, [], gts[v.video_id], 1.0, v.duration_s, tol_s, merge_gap_s,
                                 lane_hits=lane_cache[v.video_id])
            c = per[LANE]
            rows.append(_row(v.video_id, "vote", LANE, c))
            total.setdefault(LANE, TypeCounts()).add(c)
        rows.extend(_with_totals("all", "vote", total))
    return rows


def _with_totals(scope: str, name: str, total: dict[str, TypeCounts]) -> list[dict]:
    rows = [_row(scope, name, d, total[d]) for d in EVAL_TYPES if d in total]
    defects = [d for d in DETECTOR_DEFECTS if d in total]
    if len(defects) > 1:
        agg = TypeCounts()
        for d in defects:
            agg.add(total[d])
        agg.minutes = total[defects[0]].minutes
        rows.append(_row(scope, name, "pothole+crack", agg))
    return rows


def _row(scope: str, conf: str, dtype: str, c: TypeCounts) -> dict:
    return {
        "scope": scope, "conf": conf, "group": "defect" if dtype == "pothole+crack" else type_group(dtype),
        "defect_type": dtype, "n_gt": c.n_gt, "n_gt_found": c.n_gt_found, "n_pred": c.n_pred,
        "n_pred_ok": c.n_pred_ok, "precision": fmt(c.precision), "recall": fmt(c.recall), "f1": fmt(c.f1),
        "miss_rate": fmt(c.miss_rate), "pred_per_min": fmt(c.per_min, 2), "minutes": fmt(c.minutes, 2),
        "missed_ids": " ".join(c.missed),
    }


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)


def public_metrics(model: Path) -> dict | None:
    card = model.with_suffix(".json")
    if not card.exists():
        return None
    d = json.loads(card.read_text(encoding="utf-8"))
    return {"source": str(card), "split": "test", "metrics": d.get("metrics", {}).get("test")}


def print_table(rows: Sequence[dict], public: dict | None) -> None:
    allrows = [r for r in rows if r["scope"] == "all"]
    print("\n== 실제 주행 영상 (이벤트 단위, 전체 영상 합) ==")
    print(f"{'conf':>9} {'group':>17} {'type':>16} {'GT':>4} {'찾음':>4} {'예측':>5} {'P':>6} {'R':>6} {'F1':>6} {'/분':>6}")
    for r in allrows:
        print(f"{r['conf']:>9} {r['group']:>17} {r['defect_type']:>16} {r['n_gt']:>4} {r['n_gt_found']:>4} "
              f"{r['n_pred']:>5} {r['precision']:>6} {r['recall']:>6} {r['f1']:>6} {r['pred_per_min']:>6}")
    if public and public.get("metrics"):
        print(f"\n== 공개 테스트셋 (박스 단위, {public['source']}) ==")
        for k, m in public["metrics"].items():
            print(f"{k:>10}  P {m.get('P')}  R {m.get('R')}  mAP50 {m.get('mAP50')}")
    print("\n주의: 시설물(guardrail_damage·sign_damage)은 위치 찾기만 잰다 — 결함 합계에 넣지 않음.")


def parse_confs(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv: Sequence[str] | None = None, detector_factory=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=Path("ai/models/detector_v2.onnx"))
    ap.add_argument("--gt-dir", type=Path, default=Path("experiments/gt_events"))
    ap.add_argument("--every", type=int, default=3, help="몇 프레임마다 분석하나 (기본 3)")
    ap.add_argument("--confs", type=parse_confs, default=list(DEFAULT_CONFS), help="쉼표로 구분 (기본 0.25~0.6)")
    ap.add_argument("--conf-json", type=Path, help="클래스별 conf JSON (예: {\"pothole\": 0.3, ...})")
    ap.add_argument("--tol", type=float, default=0.5, help="정답 창 앞뒤 여유 (초)")
    ap.add_argument("--merge-gap", type=float, default=1.0, help="hit 사이가 이 초 이하면 같은 이벤트")
    ap.add_argument("--point-half", type=float, default=1.0, help="t_s 만 있는 정답의 창 반폭 (초)")
    ap.add_argument("--no-lane", action="store_true", help="차선 마모를 재지 않는다")
    ap.add_argument("--fps", type=float, help="영상 fps 를 덮어쓴다 (메타데이터가 틀릴 때)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache-dir", type=Path, help="프레임별 탐지 캐시 폴더 (예: data/eval_cache)")
    ap.add_argument("--out", type=Path, default=Path("experiments/eval_detector"))
    a = ap.parse_args(argv)
    if a.every < 1:
        ap.error("--every 는 1 이상")

    confs: list[tuple[str, float | dict[str, float]]] = [(f"{c:g}", c) for c in a.confs]
    if a.conf_json:
        per = load_conf_json(a.conf_json)
        confs.append((f"per_class:{a.conf_json.name}", per))
    floor = min([*a.confs, *(per.values() if a.conf_json else ()), 1.0])
    lane = not a.no_lane

    if detector_factory is None:
        def detector_factory(conf: float):
            from daboyeo_ai.inference import OnnxDetector

            return OnnxDetector(a.model, threads=a.threads, conf=conf)

    videos, gts = [], {}
    for v in a.videos:
        gt_path = find_gt(v, a.gt_dir)
        if gt_path is None:
            print(f"[{v.stem}] 정답 {a.gt_dir / (v.stem + '.csv')} 이 없어 건너뜀", file=sys.stderr)
            continue
        vr = scan_or_load(v, a.every, floor, detector_factory, lane, a.cache_dir, a.fps, a.model.name)
        videos.append(vr)
        gts[vr.video_id] = load_gt(gt_path, vr.video_id, a.point_half)
        print(f"[{vr.video_id}] {vr.duration_s / 60:.1f} 분, 분석 프레임 {len(vr.records)}, 정답 {len(gts[vr.video_id])}건")
    if not videos:
        print("평가할 영상이 없다 (정답 CSV 확인)", file=sys.stderr)
        return 1

    rows = evaluate(videos, gts, confs, a.tol, a.merge_gap, lane)
    public = public_metrics(a.model)
    write_csv(a.out / "summary.csv", rows, FIELDS)
    summary = {
        "model": a.model.name, "every": a.every, "tol_s": a.tol, "merge_gap_s": a.merge_gap,
        "point_half_s": a.point_half, "lane": lane,
        "conf_json": per if a.conf_json else None,
        "videos": [{"video_id": v.video_id, "minutes": round(v.duration_s / 60, 3), "fps": v.fps,
                    "frames_analyzed": len(v.records), "n_gt": len(gts[v.video_id])} for v in videos],
        "facility_note": "guardrail_damage·sign_damage 는 파손이 아니라 위치 찾기(facility_presence)로만 잼 (PRD v0.5)",
        "rows": rows, "public_testset": public,
    }
    (a.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print_table(rows, public)
    print(f"\n저장: {a.out / 'summary.csv'}, {a.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
