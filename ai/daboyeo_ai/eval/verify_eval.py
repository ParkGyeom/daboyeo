"""서버 검증기 V0 평가 — 크롭에 detector_v2 를 다시 돌리면 오탐을 '신뢰도 올리기' 보다 잘 지우는가.

    uv run python -m daboyeo_ai.eval.verify_eval \\
        --model /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \\
        --parts-dir /Users/parkgyeom/daboyeo/data/aihub/parts \\
        --records /Users/parkgyeom/daboyeo/data/verify_v0/records.csv

    # 탐지·크롭은 다시 돌리지 않고 기록만 다시 분석
    uv run python -m daboyeo_ai.eval.verify_eval --from-records /Users/parkgyeom/daboyeo/data/verify_v0/records.csv

Set T = `conf_thresholds` 와 같은 test 데이터·같은 클래스별 출처 (`CLASS_SOURCES`) 와 같은 매칭 (IoU 0.5, 점수 순
탐욕). 순서:

1. 전체 이미지에 탐지기를 클래스별 초안 기준(`class_conf_draft`: pothole 0.27, crack 0.29, guardrail 0.26,
   sign 0.24) 으로 돌린다 → 탐지마다 TP/FP.
2. 탐지마다 블랙박스와 똑같이 크롭 JPEG 를 만든다 (`eval.snapshot.crop_snapshot`, edge 코드 사본).
3. `Verifier`(1 스레드) 로 크롭 점수 v 를 매기고 시간을 잰다.
4. τ 를 훑으며: v ≥ τ 인 탐지만 남길 때 TP 유지율, FP 제거율, 남은 것의 정밀도.
   **비교 기준(baseline)**: 같은 수의 TP 를 남기도록 탐지 신뢰도 기준만 올렸을 때 지워지는 FP. V0 는
   이것보다 FP 를 더 지워야 쓸모가 있다.
5. 같은 계산을 '낮은 신뢰도 구간' (초안 기준 ≤ conf < 0.6) 에만 다시 한다. 서버는 conf ≥ 0.6 은 혼자서도
   보여주고 그 아래는 차량 2대 이상일 때만 보여주므로, 검증기가 필요한 곳은 주로 이 구간이다.

출력: `experiments/verify_v0.json` (클래스별 추천 τ·요약) 과 `experiments/verify_v0.csv` (τ 곡선).
탐지별 기록(`--records`)에는 AI Hub 이미지 이름이 들어가므로 레포 밖(`data/`)에 둔다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .conf_thresholds import CLASS_SOURCES, extract_test, iter_images, match_image, read_labels
from .snapshot import crop_snapshot, pixel_box

LOW_BAND_MAX = 0.6  # 서버가 차량 1대만으로도 보여주는 신뢰도
TARGET_RETENTION = 0.9
TARGET_FP_REMOVAL = 0.4
TARGET_MS = 200.0
RECORD_FIELDS = ["source", "image", "class", "conf", "tp", "v", "v_any", "crop_w", "crop_h", "ms"]


# ---------- 분석 (순수 함수, 테스트 대상) ----------

def baseline_fp_removed(conf: np.ndarray, tp: np.ndarray, keep_tp: int) -> tuple[int, float | None]:
    """신뢰도 기준만 올려서 TP 를 keep_tp 개 이상 남길 때 지워지는 FP 수와 그 기준.

    기준 = TP 신뢰도 중 keep_tp 번째로 높은 값 (그보다 낮은 탐지를 지운다). keep_tp == 0 이면 모든 FP 를 지울 수
    있다 (기준 = 무한대). 동점은 남긴다 → TP 가 keep_tp 보다 조금 더 남을 수 있다 (baseline 에 유리한 쪽).
    """
    conf, tp = np.asarray(conf, float), np.asarray(tp, bool)
    fp_conf = conf[~tp]
    if keep_tp <= 0:
        return int(len(fp_conf)), None
    tps = np.sort(conf[tp])[::-1]
    if keep_tp > len(tps):
        raise ValueError(f"keep_tp {keep_tp} > TP {len(tps)}")
    c = float(tps[keep_tp - 1])
    return int((fp_conf < c).sum()), c


def baseline_tp_kept(conf: np.ndarray, tp: np.ndarray, remove_fp: int) -> int:
    """반대 방향 비교: 신뢰도 기준만 올려서 FP 를 remove_fp 개 이상 지울 때 남는 TP 수 (FP 신뢰도 아래쪽부터 지운다)."""
    conf, tp = np.asarray(conf, float), np.asarray(tp, bool)
    if remove_fp <= 0:
        return int(tp.sum())
    cut = float(np.sort(conf[~tp])[remove_fp - 1])
    return int((conf[tp] > cut).sum())


def at_fp_removal(conf: Sequence[float], tp: Sequence[bool], v: Sequence[float], rows: list[dict],
                  target: float = TARGET_FP_REMOVAL) -> dict | None:
    """FP 를 target 비율 이상 지우는 가장 낮은 τ 에서의 V0 TP 유지율과, 같은 수의 FP 를 신뢰도로 지울 때의 TP 유지율."""
    r = next((r for r in rows if r["fp_removal"] is not None and r["fp_removal"] >= target - 1e-9), None)
    if r is None or not r["n_tp"]:
        return None
    b = baseline_tp_kept(np.asarray(conf, float), np.asarray(tp, bool), r["fp_removed"])
    return {"tau": r["tau"], "fp_removal": r["fp_removal"], "tp_retention": r["tp_retention"],
            "baseline_tp_retention": b / r["n_tp"]}


def sweep(conf: Sequence[float], tp: Sequence[bool], v: Sequence[float], taus: Sequence[float]) -> list[dict]:
    """τ 별 V0 결과와 같은 TP 수를 남기는 신뢰도 기준 baseline."""
    conf, tp, v = np.asarray(conf, float), np.asarray(tp, bool), np.asarray(v, float)
    n_tp, n_fp = int(tp.sum()), int((~tp).sum())
    rows = []
    for t in taus:
        keep = v >= t - 1e-9
        k_tp, k_fp = int((keep & tp).sum()), int((keep & ~tp).sum())
        rm_fp = n_fp - k_fp
        b_rm, b_conf = baseline_fp_removed(conf, tp, k_tp) if n_tp else (0, None)
        rows.append({
            "tau": round(float(t), 4), "n_tp": n_tp, "n_fp": n_fp,
            "tp_kept": k_tp, "tp_retention": k_tp / n_tp if n_tp else None,
            "fp_removed": rm_fp, "fp_removal": rm_fp / n_fp if n_fp else None,
            "precision_before": n_tp / (n_tp + n_fp) if n_tp + n_fp else None,
            "precision_after": k_tp / (k_tp + k_fp) if k_tp + k_fp else None,
            "baseline_conf": b_conf, "baseline_fp_removed": b_rm,
            "baseline_fp_removal": b_rm / n_fp if n_fp else None,
            "gain_fp": rm_fp - b_rm,
        })
    return rows


def pick_tau(rows: list[dict], min_retention: float = TARGET_RETENTION) -> dict | None:
    """TP 유지율 ≥ min_retention 중 FP 를 가장 많이 지우는 τ (같으면 낮은 τ)."""
    ok = [r for r in rows if r["tp_retention"] is not None and r["tp_retention"] >= min_retention - 1e-9]
    return max(ok, key=lambda r: (r["fp_removed"], -r["tau"]), default=None)


def best_gain(rows: list[dict]) -> dict | None:
    """baseline 대비 FP 를 가장 많이 더 지우는 τ (TP 유지율 무관 — V0 가 어디서든 이기는지 보려고)."""
    return max(rows, key=lambda r: (r["gain_fp"], -r["tau"]), default=None)


# ---------- 기록 모으기 ----------

def collect(model: Path, parts_dir: Path, class_conf: dict[str, float], limit: int, threads: int,
            records_path: Path) -> list[dict]:
    import cv2

    from daboyeo_ai.inference import OnnxDetector
    from daboyeo_ai.inference.verifier import Verifier, central_score

    det = OnnxDetector(model, threads=threads, conf=class_conf)
    ver = Verifier(model, threads=1)
    by_source: dict[str, list[str]] = {}
    for cls, srcs in CLASS_SOURCES.items():
        for s in srcs:
            by_source.setdefault(s, []).append(cls)

    records_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    n_gt: dict[str, int] = {c: 0 for c in CLASS_SOURCES}
    with records_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=RECORD_FIELDS, lineterminator="\n")
        wr.writeheader()
        for source, classes in by_source.items():
            zp = parts_dir / f"{source}.zip"
            if not zp.exists():
                print(f"[skip] {zp} 없음 → {classes} 에서 {source} 제외", file=sys.stderr)
                continue
            ids = {CLASS_NAMES.index(c): c for c in classes}
            with tempfile.TemporaryDirectory(prefix=f"verify_{source}_") as tmp:
                root = extract_test(zp, Path(tmp))
                for n, (img_path, lab_path) in enumerate(iter_images(root)):
                    if limit and n >= limit:
                        break
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    gts = read_labels(lab_path, w, h)
                    raw = det.raw(img)
                    for cid, name in ids.items():
                        g = [b for c, b in gts if c == cid]
                        n_gt[name] += len(g)
                        p = sorted([(s, (x1, y1, x2, y2)) for c, s, x1, y1, x2, y2 in raw if c == cid],
                                   key=lambda t: -t[0])
                        for (s, box), (_, is_tp) in zip(p, match_image(p, g)):
                            jpg = crop_snapshot(img, pixel_box(*box, w, h))
                            t0 = time.perf_counter()
                            crop = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            cr = ver.raw(crop)
                            ch, cw = crop.shape[:2]
                            v = central_score(cr, cid, cw, ch, ver.center)
                            ms = (time.perf_counter() - t0) * 1000
                            rec = {"source": source, "image": img_path.name, "class": name, "conf": round(s, 5),
                                   "tp": int(is_tp), "v": round(v, 5),
                                   "v_any": round(central_score(cr, cid, cw, ch, 1.0), 5),
                                   "crop_w": cw, "crop_h": ch, "ms": round(ms, 2)}
                            wr.writerow(rec)
                            records.append(rec)
                    if (n + 1) % 200 == 0:
                        print(f"  {source}: {n + 1} images, {len(records)} detections", file=sys.stderr)
            print(f"[done] {source}", file=sys.stderr)
    (records_path.with_suffix(".gt.json")).write_text(json.dumps(n_gt), encoding="utf-8")
    return records


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("conf", "v", "v_any", "ms"):
            r[k] = float(r[k])
        r["tp"] = int(r["tp"])
    return rows


# ---------- 보고 ----------

def _arr(recs: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in recs], float)


def summarize(recs: list[dict], taus: Sequence[float], score_key: str = "v") -> tuple[list[dict], dict]:
    conf, tp, v = _arr(recs, "conf"), _arr(recs, "tp").astype(bool), _arr(recs, score_key)
    rows = sweep(conf, tp, v, taus)
    pick = pick_tau(rows)
    return rows, {"n": len(recs), "n_tp": rows[0]["n_tp"] if rows else 0, "n_fp": rows[0]["n_fp"] if rows else 0,
                  "v_zero_rate": {"tp": float((v[tp] == 0).mean()) if tp.any() else None,
                                  "fp": float((v[~tp] == 0).mean()) if (~tp).any() else None},
                  "pick_retention>=0.9": pick, "best_gain": best_gain(rows),
                  "at_fp_removal>=0.4": at_fp_removal(conf, tp, v, rows),
                  "at_tau": {f"{t:.2f}": next((r for r in rows if abs(r["tau"] - t) < 1e-6), None)
                             for t in (0.1, 0.2, 0.3)}}


def overall_at(recs_by_class: dict[str, list[dict]], tau_by_class: dict[str, float], score_key: str = "v") -> dict:
    """클래스별 τ 를 적용한 전체 합계와, 클래스별로 같은 TP 수를 남기는 신뢰도 baseline 의 합계."""
    tot = {"n_tp": 0, "n_fp": 0, "tp_kept": 0, "fp_removed": 0, "baseline_fp_removed": 0}
    for name, recs in recs_by_class.items():
        if not recs or name not in tau_by_class:
            continue
        r = sweep(_arr(recs, "conf"), _arr(recs, "tp").astype(bool), _arr(recs, score_key), [tau_by_class[name]])[0]
        for k in tot:
            tot[k] += r[k]
    n_tp, n_fp = tot["n_tp"], tot["n_fp"]
    k_fp = n_fp - tot["fp_removed"]
    return {**tot, "tau": tau_by_class,
            "tp_retention": tot["tp_kept"] / n_tp if n_tp else None,
            "fp_removal": tot["fp_removed"] / n_fp if n_fp else None,
            "baseline_fp_removal": tot["baseline_fp_removed"] / n_fp if n_fp else None,
            "precision_before": n_tp / (n_tp + n_fp) if n_tp + n_fp else None,
            "precision_after": tot["tp_kept"] / (tot["tp_kept"] + k_fp) if tot["tp_kept"] + k_fp else None}


def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def build_report(records: list[dict], class_conf: dict[str, float], model: str, n_gt: dict | None,
                 taus: Sequence[float]) -> tuple[dict, list[dict]]:
    report: dict = {
        "note": "V0 = 크롭(crop_snapshot 사본)에 같은 detector_v2 를 다시 돌린 점수. baseline = 같은 TP 수를 남기도록 "
                "탐지 신뢰도 기준만 올린 것. 판단 기준(제안): TP 유지 ≥ 90% 에서 FP 제거 ≥ 40%, 그리고 baseline 보다 "
                "나아야 함, ≤ 200 ms/크롭.",
        "model": model, "date": date.today().isoformat(), "detector_conf": class_conf,
        "low_band": f"class_conf_draft <= conf < {LOW_BAND_MAX}",
        "verifier_rule": "같은 클래스·중심이 크롭 가운데 60% 안인 탐지의 최고 신뢰도 (v). v_any = 위치 무관",
        "ms_per_crop": None, "classes": {}, "overall": {},
    }
    ms = _arr(records, "ms")
    if len(ms):
        report["ms_per_crop"] = {"mean": float(ms.mean()), "p50": float(np.median(ms)),
                                 "p95": float(np.percentile(ms, 95)), "max": float(ms.max()), "n": int(len(ms)),
                                 "threads": 1}
    csv_rows: list[dict] = []
    by_class = {c: [r for r in records if r["class"] == c] for c in CLASS_SOURCES}
    band = {c: [r for r in rs if r["conf"] < LOW_BAND_MAX] for c, rs in by_class.items()}
    picks: dict[str, dict[str, float]] = {"all": {}, "band": {}}
    for name in CLASS_SOURCES:
        entry = {"gt": (n_gt or {}).get(name)}
        for subset, recs in (("all", by_class[name]), ("band", band[name])):
            for key in ("v", "v_any"):
                if not recs:
                    continue
                rows, summ = summarize(recs, taus, key)
                label = subset if key == "v" else f"{subset}_v_any"
                entry[label] = summ
                if key == "v" and summ["pick_retention>=0.9"]:
                    picks[subset][name] = summ["pick_retention>=0.9"]["tau"]
                for r in rows:
                    csv_rows.append({"class": name, "subset": subset, "score": key, **r})
        report["classes"][name] = entry
    for subset, recs_by in (("all", by_class), ("band", band)):
        report["overall"][subset] = overall_at(recs_by, picks[subset])
    return report, csv_rows


CSV_FIELDS = ["class", "subset", "score", "tau", "n_tp", "n_fp", "tp_kept", "tp_retention", "fp_removed",
              "fp_removal", "precision_before", "precision_after", "baseline_conf", "baseline_fp_removed",
              "baseline_fp_removal", "gain_fp"]


def print_table(report: dict) -> None:
    print(f"\n{'class':<10} {'subset':<6} {'TP':>5} {'FP':>5} | {'τ':>5} {'TP유지':>6} {'FP제거':>6} {'base':>6} "
          f"{'P전':>6} {'P후':>6} | best gain τ / FP+ | FP40%: TP유지 V0 / conf")
    for name, c in report["classes"].items():
        for subset in ("all", "band"):
            s = c.get(subset)
            if not s:
                continue
            p, g = s["pick_retention>=0.9"] or {}, s["best_gain"] or {}
            print(f"{name:<10} {subset:<6} {s['n_tp']:>5} {s['n_fp']:>5} | {_f(p.get('tau'), 2):>5} "
                  f"{_f(p.get('tp_retention')):>6} {_f(p.get('fp_removal')):>6} {_f(p.get('baseline_fp_removal')):>6} "
                  f"{_f(p.get('precision_before')):>6} {_f(p.get('precision_after')):>6} | "
                  f"{_f(g.get('tau'), 2)} / {g.get('gain_fp')} | "
                  f"{_f((s['at_fp_removal>=0.4'] or {}).get('tp_retention'))} / "
                  f"{_f((s['at_fp_removal>=0.4'] or {}).get('baseline_tp_retention'))}")
    for subset, o in report["overall"].items():
        print(f"{'overall':<10} {subset:<6} {o['n_tp']:>5} {o['n_fp']:>5} | {'cls':>5} {_f(o['tp_retention']):>6} "
              f"{_f(o['fp_removal']):>6} {_f(o['baseline_fp_removal']):>6} {_f(o['precision_before']):>6} "
              f"{_f(o['precision_after']):>6}")
    if report["ms_per_crop"]:
        m = report["ms_per_crop"]
        print(f"\nms/crop (1 thread): mean {m['mean']:.1f}, p50 {m['p50']:.1f}, p95 {m['p95']:.1f}, n={m['n']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, default=Path("ai/models/detector_v2.onnx"))
    ap.add_argument("--card", type=Path, default=Path("ai/models/detector_v2.json"), help="class_conf_draft 를 읽을 모델 카드")
    ap.add_argument("--parts-dir", type=Path)
    ap.add_argument("--records", type=Path, default=Path("data/verify_v0/records.csv"))
    ap.add_argument("--from-records", type=Path, help="이미 모은 기록만 다시 분석")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--threads", type=int, default=4, help="전체 이미지 탐지용 (검증기는 항상 1 스레드)")
    ap.add_argument("--limit", type=int, default=0, help="출처당 이미지 수 제한 (빠른 확인용)")
    args = ap.parse_args(argv)

    from daboyeo_ai.inference import load_class_conf

    class_conf = load_class_conf(args.card)
    if args.from_records:
        records = load_records(args.from_records)
        gt_path = args.from_records.with_suffix(".gt.json")
    else:
        if args.parts_dir is None:
            ap.error("--parts-dir 또는 --from-records 가 필요하다")
        records = collect(args.model, args.parts_dir, class_conf, args.limit, args.threads, args.records)
        gt_path = args.records.with_suffix(".gt.json")
    n_gt = json.loads(gt_path.read_text()) if gt_path.exists() else None

    taus = [round(x, 2) for x in np.arange(0.0, 0.901, 0.01)]
    report, rows = build_report(records, class_conf, args.model.name, n_gt, taus)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "verify_v0.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (args.out_dir / "verify_v0.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=lambda v: round(v, 4)), encoding="utf-8")
    print_table(report)
    print(f"\n→ {args.out_dir / 'verify_v0.json'}, {args.out_dir / 'verify_v0.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
