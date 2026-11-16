"""발표 슬라이드용 RDD2022 테스트 이미지 예시 — 탐지 모델이 포트홀·균열을 잡는 장면 (handoff 19장).

    uv run python -m daboyeo_ai.eval.rdd_figs \\
        --zip /Users/parkgyeom/daboyeo/data/aihub/parts/rdd2022.zip \\
        --model /Users/parkgyeom/daboyeo/ai/models/detector_v3.onnx \\
        --out /Users/parkgyeom/daboyeo/data/presentation/rdd_pothole

1) zip 의 `det/images/test` (학습에 쓰지 않은 split) 를 임시 폴더에 풀고 `OnnxDetector` 를 낮은 기준(0.05)으로 한 번
   돌린다. 결과는 `<out>/preds.json` 에 캐시 → 다시 돌리면 추론을 건너뛴다 (`--redo` 로 다시).
2) 모델 카드의 `class_conf_draft` (pothole 0.31, crack 0.42) 로 거른 뒤 RDD test 전체의 클래스별 정밀도·재현율
   (IoU 0.5, `conf_thresholds.match_image` 와 같은 탐욕 매칭) 을 계산한다.
3) 포트홀 정답이 있는 이미지마다 결과를 나눈다: hit (정답 전부 잡고 헛것 없음) / miss (하나도 못 잡음) /
   false (포트홀이 아닌 곳을 포트홀로). 낮(밝기)·박스 크기·국가 우선순위로 후보를 고른다.
4) 후보를 원본 해상도 PNG 로 그린다 — 정답 = 얇은 초록 선, 모델 = 굵은 선 + `class conf` 글씨
   (포트홀 빨강, 균열 노랑). `contact_sheet.png` 와 `summary.json` (숫자·후보 목록) 을 같이 쓴다.

이미지는 커밋하지 않는다 (`data/` 는 git 밖). 라이선스·출처 표기는 docs/ai-handoff.md 19장.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .conf_thresholds import extract_test, iter_images, match_image, read_labels

CLASSES = ("pothole", "crack")  # RDD2022 에 정답이 있는 클래스만 그린다 (가드레일·표지판은 라벨이 없다)
# 블랙박스(차량 앞 유리) 와 화각이 비슷한 순서. China_Drone 은 변환 때 이미 빠졌다
COUNTRY_PRIORITY = ("Japan", "Czech", "Norway", "India", "United_States", "China_MotorBike")
GT_COLOR = (60, 220, 60)  # BGR 초록
PRED_COLOR = {"pothole": (40, 40, 255), "crack": (0, 215, 255)}  # 빨강, 노랑


# ---------- 순수 함수 (테스트 대상) ----------

def country_of(stem: str) -> str:
    """'rdd_United_States_001808' → 'United_States'."""
    s = stem[4:] if stem.startswith("rdd_") else stem
    return s.rsplit("_", 1)[0]


def filter_preds(preds: Sequence[dict], class_conf: dict[str, float]) -> list[dict]:
    return [p for p in preds if p["cls"] in class_conf and p["score"] >= class_conf[p["cls"]] - 1e-9]


def match_marks(preds: Sequence[dict], gts: Sequence[dict], cls: str, iou_thr: float = 0.5) -> tuple[list[bool], list[bool]]:
    """한 이미지·한 클래스 → (pred 별 TP 여부 (preds 순서), gt 별 잡힘 여부). match_image 와 같은 점수순 탐욕 매칭."""
    from .conf_thresholds import iou_matrix

    p = [x for x in preds if x["cls"] == cls]
    g = [x for x in gts if x["cls"] == cls]
    tp_p, hit_g = [False] * len(p), [False] * len(g)
    if p and g:
        order = sorted(range(len(p)), key=lambda i: -p[i]["score"])
        ious = iou_matrix(np.array([p[i]["box"] for i in order]), np.array([x["box"] for x in g]))
        for k, i in enumerate(order):
            cand = np.where(hit_g, -1.0, ious[k])
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                hit_g[j], tp_p[i] = True, True
    return tp_p, hit_g


def pr_at(images: Sequence[dict], class_conf: dict[str, float], iou_thr: float = 0.5) -> dict[str, dict]:
    """images = [{gts, preds}] → 클래스별 {tp, fp, n_gt, P, R} (class_conf 로 거른 뒤)."""
    out = {}
    for cls in class_conf:
        tp = fp = n_gt = 0
        for im in images:
            g = [x["box"] for x in im["gts"] if x["cls"] == cls]
            p = [(x["score"], x["box"]) for x in filter_preds(im["preds"], class_conf) if x["cls"] == cls]
            m = match_image(p, g, iou_thr)
            tp += sum(t for _, t in m)
            fp += sum(not t for _, t in m)
            n_gt += len(g)
        out[cls] = {"conf": class_conf[cls], "tp": tp, "fp": fp, "n_gt": n_gt,
                    "P": round(tp / (tp + fp), 3) if tp + fp else None,
                    "R": round(tp / n_gt, 3) if n_gt else None}
    return out


@dataclass
class Outcome:
    name: str
    country: str
    w: int
    h: int
    brightness: float
    n_gt: int  # 포트홀 정답 수
    tp: int
    fp: int
    fn: int
    gt_area: float  # 가장 큰 포트홀 정답 박스 넓이 / 이미지 넓이
    best_score: float  # 포트홀 예측 중 가장 높은 점수 (없으면 0)
    kind: str  # hit / miss / false / partial / none


def outcome_of(im: dict, class_conf: dict[str, float], iou_thr: float = 0.5) -> Outcome:
    preds = filter_preds(im["preds"], class_conf)
    tp_p, hit_g = match_marks(preds, im["gts"], "pothole", iou_thr)
    n_gt, tp, fp = len(hit_g), sum(tp_p), len(tp_p) - sum(tp_p)
    fn = n_gt - sum(hit_g)
    w, h = im["w"], im["h"]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) / (w * h) for b in (x["box"] for x in im["gts"] if x["cls"] == "pothole")]
    scores = [x["score"] for x in preds if x["cls"] == "pothole"]
    if n_gt and fn == 0 and fp == 0:
        kind = "hit"
    elif n_gt and tp == 0 and fp == 0:
        kind = "miss"
    elif fp and tp == 0:
        kind = "false"
    elif n_gt or fp:
        kind = "partial"
    else:
        kind = "none"
    return Outcome(im["name"], country_of(im["name"]), w, h, float(im.get("brightness", 0.0)), n_gt, tp, fp, fn,
                   round(max(areas, default=0.0), 5), round(max(scores, default=0.0), 3), kind)


def select_candidates(outs: Sequence[Outcome], n_hit: int = 14, n_miss: int = 3, n_false: int = 3,
                      per_country: int = 4, min_brightness: float = 70.0, min_area: float = 0.004) -> list[Outcome]:
    """슬라이드 후보. hit 는 낮·큰 포트홀·높은 점수·화각 우선 국가 순, 국가당 per_country 장까지.
    miss 는 잘 보이는(큰) 포트홀을 놓친 것, false 는 헛것 점수가 높은 것 (한계를 정직하게 보여 주기 위해)."""
    pri = {c: i for i, c in enumerate(COUNTRY_PRIORITY)}
    day = [o for o in outs if o.brightness >= min_brightness]

    def cap(items: list[Outcome], n: int) -> list[Outcome]:
        got, per = [], {}
        if n <= 0:
            return got
        for o in items:
            if per.get(o.country, 0) >= per_country:
                continue
            got.append(o)
            per[o.country] = per.get(o.country, 0) + 1
            if len(got) >= n:
                break
        return got

    hits = sorted((o for o in day if o.kind == "hit" and o.gt_area >= min_area),
                  key=lambda o: (pri.get(o.country, 99), -round(o.best_score, 1), -o.gt_area, o.name))
    misses = sorted((o for o in day if o.kind == "miss" and o.gt_area >= min_area),
                    key=lambda o: (-o.gt_area, o.name))
    falses = sorted((o for o in day if o.kind == "false"), key=lambda o: (-o.best_score, o.name))
    return cap(hits, n_hit) + cap(misses, n_miss) + cap(falses, n_false)


# ---------- 그림 ----------

def draw(img: np.ndarray, im: dict, class_conf: dict[str, float]) -> np.ndarray:
    """정답 = 얇은 초록 선, 예측 = 굵은 클래스 색 선 + 'class 0.78' 라벨. img 는 바꾸지 않는다."""
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    s = min(w, h)
    thick = max(3, round(s / 140))
    thin = max(1, round(s / 450))
    fscale = max(0.5, s / 900)
    for g in im["gts"]:
        x1, y1, x2, y2 = map(round, g["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), GT_COLOR, thin, cv2.LINE_AA)
    for p in sorted(filter_preds(im["preds"], class_conf), key=lambda p: p["score"]):
        x1, y1, x2, y2 = map(round, p["box"])
        color = PRED_COLOR.get(p["cls"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
        label = f"{p['cls']} {p['score']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fscale, max(1, thick // 2))
        ty = y1 - 2 if y1 - th - base - 4 >= 0 else y2 + th + base + 2
        cv2.rectangle(out, (x1, ty - th - base), (min(w, x1 + tw + 4), ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 2, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX, fscale, (0, 0, 0),
                    max(1, thick // 2), cv2.LINE_AA)
    return out


def contact_sheet(tiles: Sequence[tuple[np.ndarray, str]], cols: int = 5, tw: int = 480, th: int = 360) -> np.ndarray:
    """[(이미지, 제목)] → 한 장. 각 칸은 비율을 지켜 tw×th 안에 맞추고 아래에 제목."""
    import cv2

    cap = 28
    rows = max(1, -(-len(tiles) // cols))
    sheet = np.full((rows * (th + cap), cols * tw, 3), 255, np.uint8)
    for i, (img, title) in enumerate(tiles):
        r, c = divmod(i, cols)
        h, w = img.shape[:2]
        k = min(tw / w, th / h)
        small = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
        y0, x0 = r * (th + cap), c * tw
        oy, ox = (th - small.shape[0]) // 2, (tw - small.shape[1]) // 2
        sheet[y0 + oy:y0 + oy + small.shape[0], x0 + ox:x0 + ox + small.shape[1]] = small
        cv2.putText(sheet, title, (x0 + 6, y0 + th + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


# ---------- 추론 ----------

def run_detector(detector, det_root: Path) -> list[dict]:
    import cv2

    ids = {CLASS_NAMES.index(c): c for c in CLASSES}
    images = []
    for n, (img_path, lab_path) in enumerate(iter_images(det_root), 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = [{"cls": ids[c], "box": [round(v, 1) for v in b]} for c, b in read_labels(lab_path, w, h) if c in ids]
        preds = [{"cls": ids[c], "score": round(float(s), 4), "box": [round(float(v), 1) for v in (x1, y1, x2, y2)]}
                 for c, s, x1, y1, x2, y2 in detector.raw(img) if c in ids]
        images.append({"name": img_path.stem, "file": img_path.name, "w": w, "h": h,
                       "brightness": round(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()), 1),
                       "gts": gts, "preds": preds})
        if n % 500 == 0:
            print(f"  {n} images", file=sys.stderr)
    return images


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", type=Path, required=True, help="rdd2022.zip (det/images/test 포함)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--card", type=Path, help="class_conf_draft 를 읽을 모델 카드 (기본: 모델 옆 .json)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--redo", action="store_true", help="preds.json 이 있어도 추론을 다시 한다")
    ap.add_argument("--pick", default="", help="이 이름들만 그린다 (쉼표, 자동 후보 대신)")
    args = ap.parse_args(argv)

    from daboyeo_ai.inference import load_class_conf

    card = args.card or args.model.with_suffix(".json")
    class_conf = {c: v for c, v in load_class_conf(card).items() if c in CLASSES}
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "preds.json"

    with tempfile.TemporaryDirectory(prefix="rdd_figs_") as tmp:
        root = extract_test(args.zip, Path(tmp))
        if args.redo or not cache.exists():
            from daboyeo_ai.inference import OnnxDetector

            det = OnnxDetector(args.model, threads=args.threads, conf=args.low_conf)
            images = run_detector(det, root)
            cache.write_text(json.dumps({"model": args.model.name, "low_conf": args.low_conf, "images": images}))
        else:
            images = json.loads(cache.read_text())["images"]
            print(f"[cache] {cache} ({len(images)} images)", file=sys.stderr)

        pr = pr_at(images, class_conf)
        outs = [outcome_of(im, class_conf) for im in images]
        by_name = {im["name"]: im for im in images}
        if args.pick:
            picked = [o for o in outs if o.name in set(args.pick.split(","))]
        else:
            picked = select_candidates(outs)

        import cv2

        tiles = []
        for i, o in enumerate(picked, 1):
            im = by_name[o.name]
            img = cv2.imread(str(root / "images" / "test" / im["file"]))
            fig = draw(img, im, class_conf)
            cv2.imwrite(str(args.out / f"{i:02d}_{o.kind}_{o.name}.png"), fig)
            tiles.append((fig, f"{i:02d} {o.kind} {o.name[4:]}"))
        cv2.imwrite(str(args.out / "contact_sheet.png"), contact_sheet(tiles))

    kinds = {}
    for o in outs:
        if o.n_gt or o.fp:
            kinds[o.kind] = kinds.get(o.kind, 0) + 1
    summary = {"model": args.model.name, "class_conf": class_conf, "images": len(images), "iou": 0.5,
               "pr_at_class_conf": pr, "pothole_image_outcomes": kinds,
               "candidates": [{"idx": i, **asdict(o)} for i, o in enumerate(picked, 1)]}
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for cls, r in pr.items():
        print(f"{cls:8s} conf {r['conf']:.2f}: P {r['P']} R {r['R']} (tp {r['tp']}, fp {r['fp']}, gt {r['n_gt']})")
    print(f"pothole images: {kinds}")
    print(f"→ {len(picked)} figures, {args.out / 'contact_sheet.png'}, {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
