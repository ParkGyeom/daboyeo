"""클래스별 탐지 신뢰도 기준 초안 — test 데이터의 정밀도·재현율 곡선에서 고른다 (오픈 이슈 #3 '노출 기준 θ').

    uv run python -m daboyeo_ai.eval.conf_thresholds \\
        --model /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \\
        --parts-dir /Users/parkgyeom/daboyeo/data/aihub/parts

`--parts-dir` 에서 `krid.zip`·`road.zip`·`rdd2022.zip` 중 있는 것만 쓴다 (없으면 그 출처를 빼고 진행).
각 zip 의 `det/images/test/` + `det/labels/test/` (YOLO 형식) 를 임시 폴더에 풀고, `OnnxDetector` (블랙박스와 같은
전처리) 를 낮은 기준(`--low-conf`, 0.05) 으로 한 번 돌린다. 클래스별 NMS 라서 낮은 기준으로 돌린 뒤 점수로
거른 결과는 그 기준으로 바로 돌린 결과와 같다 (`test_decode_low_conf_then_filter_equals_direct`).

클래스별로 어느 출처를 쓰나 (`CLASS_SOURCES`): pothole = RDD2022 + 도로장애물(179), crack = RDD2022 만
(179 의 크랙 라벨은 짧은 선분 박스 수백 개라 박스 매칭이 안 된다, mAP50 0.001), guardrail·sign = KRID.
출처가 하나도 없는 클래스는 기본값 0.25 를 그대로 두고 `rule = "default"` 로 적는다.

매칭: 이미지·클래스별로 점수 높은 탐지부터, 아직 안 짝지어진 GT 중 IoU 가 가장 큰 것과 IoU ≥ 0.5 이면 TP.
점수 순서대로 짝지으므로 기준 t 이상 탐지끼리의 매칭은 t 미만 탐지에 영향을 받지 않는다 → 곡선이 정확하다.

추천 규칙 (시민에게 헛것을 보여주지 않는 쪽): 정밀도 ≥ `--target-precision`(0.7) 이고 탐지가
`--min-support`(20) 건 이상인 가장 낮은 기준. 그런 기준이 없으면 F1 최대 기준. 둘 다, 그리고 0.25 일 때의
P/R 과 재현율 손해를 출력한다.

출력: `--out-dir`(기본 `experiments/`) 에 `conf_thresholds.csv` (클래스·기준별 곡선) 과
`conf_thresholds.json` (`class_conf` = {클래스: 기준} — `OnnxDetector(conf=load_class_conf(...))`·
`interval_sweep --conf-json` 이 읽는 키, 나머지는 근거).

한계: test 는 학습 데이터와 같은 출처의 사진이다. 시연 영상(다른 카메라·도로)에서는 정밀도가 달라진다 —
이 값은 초안이고 시연 영상으로 확인한다 (#3, 박겸·정현우).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from daboyeo_contracts import CLASS_NAMES

CLASS_SOURCES: dict[str, tuple[str, ...]] = {
    "pothole": ("rdd2022", "road"),
    "crack": ("rdd2022",),
    "guardrail": ("krid",),
    "sign": ("krid",),
}
SOURCE_NOTES = {
    "rdd2022": "RDD2022 test (해외 도로 사진)",
    "road": "AI Hub 도로장애물(179) test",
    "krid": "AI Hub KRID test",
}
DEFAULT_CONF = 0.25
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


# ---------- 매칭·곡선 (순수 함수, 테스트 대상) ----------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (N,4), b (M,4) xyxy → (N, M) IoU."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_image(
    preds: Sequence[tuple[float, Sequence[float]]], gts: Sequence[Sequence[float]], iou_thr: float = 0.5,
) -> list[tuple[float, bool]]:
    """한 이미지·한 클래스: [(score, xyxy)] 와 GT [xyxy] → [(score, is_tp)] (점수 내림차순, 탐욕 매칭)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][0])
    if not gts:
        return [(float(preds[i][0]), False) for i in order]
    ious = iou_matrix(np.array([preds[i][1] for i in order]).reshape(-1, 4), np.array(gts))
    used = np.zeros(len(gts), bool)
    out = []
    for k, i in enumerate(order):
        cand = np.where(used, -1.0, ious[k])
        j = int(cand.argmax())
        tp = bool(cand[j] >= iou_thr)
        if tp:
            used[j] = True
        out.append((float(preds[i][0]), tp))
    return out


def pr_curve(scored: Iterable[tuple[float, bool]], n_gt: int, grid: Sequence[float]) -> list[dict]:
    """[(score, is_tp)] + GT 수 → 기준별 {conf, tp, fp, P, R, F1}. 탐지가 없으면 P 는 None."""
    arr = np.array(list(scored), dtype=float).reshape(-1, 2)
    s, tp = arr[:, 0], arr[:, 1].astype(bool)
    rows = []
    for t in grid:
        m = s >= t - 1e-9
        ntp, nfp = int(tp[m].sum()), int((~tp[m]).sum())
        p = ntp / (ntp + nfp) if ntp + nfp else None
        r = ntp / n_gt if n_gt else None
        f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r > 0 else (0.0 if p is not None else None)
        rows.append({"conf": round(float(t), 4), "tp": ntp, "fp": nfp, "P": p, "R": r, "F1": f1})
    return rows


def _at(curve: list[dict], conf: float) -> dict | None:
    return min(curve, key=lambda r: abs(r["conf"] - conf)) if curve else None


def recommend(
    curve: list[dict], target_precision: float = 0.7, min_support: int = 20, default: float = DEFAULT_CONF,
) -> dict:
    """곡선 → 추천 기준. 정밀도 목표를 넘는 가장 낮은 기준, 없으면 F1 최대."""
    ok = [r for r in curve if r["P"] is not None and r["tp"] + r["fp"] >= min_support]
    hit = next((r for r in ok if r["P"] >= target_precision), None)
    best = max(ok, key=lambda r: (r["F1"] or 0.0, -r["conf"]), default=None)
    base = _at(curve, default)
    if hit is not None:
        pick, rule = hit, f"precision>={target_precision}"
    elif best is not None:
        pick, rule = best, "max_f1"
    else:
        return {"conf": default, "rule": "default", "pick": None, "precision_target": None, "max_f1": None,
                "at_default": base}
    return {"conf": pick["conf"], "rule": rule, "pick": pick, "precision_target": hit, "max_f1": best,
            "at_default": base}


# ---------- 데이터 ----------

@dataclass
class ClassStats:
    scored: list[tuple[float, bool]] = field(default_factory=list)
    n_gt: int = 0
    images: int = 0
    sources: dict[str, int] = field(default_factory=dict)  # 출처 → 이미지 수


def extract_test(zip_path: Path, out: Path) -> Path:
    """zip 의 det/images/test, det/labels/test 만 out 에 푼다 → out/det."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if n.startswith(("det/images/test/", "det/labels/test/")) and ".." not in n and not n.endswith("/")]
        if not names:
            raise ValueError(f"{zip_path}: det/images/test 가 없다")
        z.extractall(out, members=names)
    return out / "det"


def read_labels(path: Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    """YOLO 라벨 (cls cx cy bw bh, 0~1) → [(cls, [x1, y1, x2, y2] 픽셀)]."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().split("\n"):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        out.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def iter_images(det_root: Path) -> Iterator[tuple[Path, Path]]:
    for img in sorted((det_root / "images" / "test").iterdir()):
        if img.suffix.lower() in IMG_EXT:
            yield img, det_root / "labels" / "test" / f"{img.stem}.txt"


def evaluate_part(detector, det_root: Path, source: str, classes: Sequence[str],
                  stats: dict[str, ClassStats], iou_thr: float) -> int:
    import cv2

    ids = {CLASS_NAMES.index(c): c for c in classes}
    n = 0
    for img_path, lab_path in iter_images(det_root):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = read_labels(lab_path, w, h)
        raw = detector.raw(img)
        for cid, name in ids.items():
            st = stats[name]
            g = [b for c, b in gts if c == cid]
            p = [(s, (x1, y1, x2, y2)) for c, s, x1, y1, x2, y2 in raw if c == cid]
            st.scored.extend(match_image(p, g, iou_thr))
            st.n_gt += len(g)
            st.images += 1
            st.sources[source] = st.sources.get(source, 0) + 1
        n += 1
        if n % 200 == 0:
            print(f"  {source}: {n} images", file=sys.stderr)
    return n


# ---------- 출력 ----------

def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def build_report(stats: dict[str, ClassStats], grid: Sequence[float], target_precision: float,
                 min_support: int, iou_thr: float, model: str,
                 alt_targets: Sequence[float] = (0.8, 0.9)) -> tuple[dict, list[dict]]:
    classes, csv_rows, class_conf = {}, [], {}
    for name in CLASS_NAMES:
        if name not in CLASS_SOURCES:
            continue
        st = stats.get(name) or ClassStats()
        curve = pr_curve(st.scored, st.n_gt, grid) if st.images else []
        rec = recommend(curve, target_precision, min_support)
        class_conf[name] = rec["conf"]
        for r in curve:
            csv_rows.append({"class": name, **r})
        pick, base = rec["pick"], rec["at_default"]
        classes[name] = {
            "conf": rec["conf"],
            "rule": rec["rule"],
            "P": pick and pick["P"], "R": pick and pick["R"], "F1": pick and pick["F1"],
            "precision_target": rec["precision_target"],
            "max_f1": rec["max_f1"],
            "at_0.25": base,
            "recall_cost_vs_0.25": (base["R"] - pick["R"]) if pick and base and base["R"] is not None else None,
            "sources": {s: SOURCE_NOTES.get(s, s) for s in st.sources},
            "images": st.images, "gt": st.n_gt,
            "missing_sources": [s for s in CLASS_SOURCES[name] if s not in st.sources],
            # 더 엄격한 목표일 때 (같이 정할 때 비교용): 목표를 못 넘으면 None
            "alternatives": {f"precision>={t}": recommend(curve, t, min_support)["precision_target"]
                             for t in alt_targets},
        }
    report = {
        "class_conf": class_conf,
        "note": "초안 (제안 — 정현우 확인 필요). test 데이터 기준, 시연 영상으로 다시 확인 (오픈 이슈 #3). "
                "k(최소 차량 수)는 서버 쪽 값이라 여기서 정하지 않는다.",
        "model": model,
        "date": date.today().isoformat(),
        "rule": {"target_precision": target_precision, "min_support": min_support, "iou": iou_thr,
                 "fallback": "max_f1", "no_data": DEFAULT_CONF},
        "class_sources": {k: list(v) for k, v in CLASS_SOURCES.items()},
        "classes": classes,
    }
    return report, csv_rows


def print_alternatives(report: dict) -> None:
    alts = next(iter(report["classes"].values()))["alternatives"]
    print(f"\n{'class':<10} " + " | ".join(f"{k:>22}" for k in alts) + "   (conf / P / R)")
    for name, c in report["classes"].items():
        cells = [f"{r['conf']:.2f} / {_f(r['P'])} / {_f(r['R'])}" if r else "-"
                 for r in c["alternatives"].values()]
        print(f"{name:<10} " + " | ".join(f"{x:>22}" for x in cells))


def print_table(report: dict) -> None:
    print(f"\n{'class':<10} {'conf':>5} {'rule':<16} {'P':>6} {'R':>6} | {'P@.25':>6} {'R@.25':>6} {'R손해':>6}"
          f" | {'maxF1 conf':>10} {'P':>6} {'R':>6} | images  gt  sources")
    for name, c in report["classes"].items():
        b, m = c["at_0.25"] or {}, c["max_f1"] or {}
        print(f"{name:<10} {c['conf']:>5.2f} {c['rule']:<16} {_f(c['P']):>6} {_f(c['R']):>6} | "
              f"{_f(b.get('P')):>6} {_f(b.get('R')):>6} {_f(c['recall_cost_vs_0.25']):>6} | "
              f"{_f(m.get('conf'), 2):>10} {_f(m.get('P')):>6} {_f(m.get('R')):>6} | "
              f"{c['images']:>6} {c['gt']:>4}  {','.join(c['sources']) or '-'}"
              + (f"  (없음: {','.join(c['missing_sources'])})" if c["missing_sources"] else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--parts-dir", type=Path, required=True, help="krid.zip / road.zip / rdd2022.zip 이 있는 폴더")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments"))
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--target-precision", type=float, default=0.7)
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--alt-targets", default="0.8,0.9", help="비교용으로 같이 보여줄 정밀도 목표")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="출처당 이미지 수 제한 (빠른 확인용, 0 = 전부)")
    args = ap.parse_args(argv)

    from daboyeo_ai.inference import OnnxDetector

    det = OnnxDetector(args.model, threads=args.threads, conf=args.low_conf)
    by_source: dict[str, list[str]] = {}
    for cls, srcs in CLASS_SOURCES.items():
        for s in srcs:
            by_source.setdefault(s, []).append(cls)

    stats = {c: ClassStats() for c in CLASS_SOURCES}
    for source, classes in by_source.items():
        zp = args.parts_dir / f"{source}.zip"
        if not zp.exists():
            print(f"[skip] {zp} 없음 → {classes} 에서 {source} 제외", file=sys.stderr)
            continue
        with tempfile.TemporaryDirectory(prefix=f"conf_{source}_") as tmp:
            root = extract_test(zp, Path(tmp))
            if args.limit:
                imgs = sorted((root / "images" / "test").iterdir())
                for p in imgs[args.limit:]:
                    p.unlink()
            n = evaluate_part(det, root, source, classes, stats, args.iou)
            print(f"[done] {source}: {n} images → {classes}", file=sys.stderr)

    grid = [round(x, 2) for x in np.arange(max(args.low_conf, 0.01), 0.951, 0.01)]
    report, rows = build_report(stats, grid, args.target_precision, args.min_support, args.iou, args.model.name,
                                [float(x) for x in args.alt_targets.split(",") if x])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "conf_thresholds.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["class", "conf", "tp", "fp", "P", "R", "F1"], lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    (args.out_dir / "conf_thresholds.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=lambda v: round(v, 4)), encoding="utf-8")
    print_table(report)
    print_alternatives(report)
    print(f"\n→ {args.out_dir / 'conf_thresholds.json'}, {args.out_dir / 'conf_thresholds.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
