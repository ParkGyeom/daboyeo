"""차선 마모 판정을 실제 마모 등급 라벨(RMDD)로 재 본다 (handoff 24장).

데이터: Roboflow Universe `rmdd/road-marking-damage-detection` v7 "original" (1,000장, MIT 표기, 일본, 차량 앞 카메라).
YOLO 박스 클래스 AD0-2 화살표 · BD0-2 횡단보도 · LD0-2 차선 · WND0-2 글자, 끝자리 0 = 멀쩡 … 2 = 많이 닳음.
`/Users/…/daboyeo/data/rmdd/` 에 YOLOv8 내보내기 그대로 풀어 둔다 (git 제외).

    uv run python -m daboyeo_ai.eval.lane_wear_rmdd --root data/rmdd --json experiments/lane_wear_rmdd.json \
        [--seg-model ai/models/lane_seg_v1.onnx]

방법 (결정과 이유):
- **사진 되돌리기**: Roboflow 가 원본(16:9)을 640×640 으로 늘려 저장했다. 원근·각도 조건이 있는 측정기에 그대로 넣으면
  기울기가 바뀌므로 640×360 으로 되돌린다 (`--aspect`). 박스는 정규화 좌표라 그대로 쓴다.
- **박스 1개 = 표본 1개.** 박스만 잘라서 재면 ROI·소실점 투표(화면 전체 기하)가 깨지므로, **사진 전체를 시연과 똑같이 재고**
  채택된 차선 선분 중 **가운데 점이 박스 안에 있는 선분만** 모아 그 박스의 값(contrast·contrast_p25·paint_len)을 낸다
  (`LaneWearMeter._segments` → 선분 고르기 → `_finish`). 박스 안 선분이 없으면 "박스에서 차선 못 찾음".
  박스 값은 길이 하한(min_total_len) 없이 낸다 — 박스 하나에는 차선 한 줄 토막만 들어 있는 경우가 많다.
  박스 단위로 시연 규칙(사진 전체에서 차선을 찾았나)도 함께 센다 (`frame_found`).
- **★ 점수 = contrast_p25 ÷ 기준 × paint_len.** 기준(주행 중앙값)은 한 주행의 여러 프레임이 있어야 하는데 RMDD 는 낱장이다.
  세 가지로 근사해 나란히 낸다:
  1. `dataset` — 데이터 전체의 사진 단위 contrast_p25 중앙값 (상수라 순위·AUC 는 절대값 contrast_p25 × paint_len 과 같다)
  2. `tracker` — 사진을 파일 번호 순(같은 주행에서 뽑은 프레임으로 가정)으로 `LaneWearTracker` 에 넣었을 때의 그 시점 기준
     (최근 100장 중앙값, 20장 전에는 판정 없음). 시연 판정과 가장 가깝다.
  3. `image` — 같은 사진 안에서 차선을 찾은 박스들의 contrast_p25 중앙값 (박스가 1개면 1 → paint_len 만 남는다)
- **사진 단위**: 차선 박스가 있는 사진 중 LD2 가 하나라도 있으면 "닳음", 차선 박스가 모두 LD0 이면 "멀쩡".
  사진 전체 측정값(시연과 같은 값)으로 AUC 와 "마모로 판정한 비율"을 낸다.
- AUC 는 점수가 낮을수록 마모 (`lane_wear_eval.auc_low_is_worn`), 95% 구간은 **사진 단위** 부트스트랩
  (한 사진의 박스들은 서로 닮았으므로).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.lane_wear import PROVISIONAL_WEAR_SCORE, LaneWearMeter, LaneWearTracker

from .lane_wear_eval import auc_low_is_worn

NAMES = ("AD0", "AD1", "AD2", "BD0", "BD1", "BD2", "LD0", "LD1", "LD2", "WND0", "WND1", "WND2")
GROUPS = {"LD": "차선", "BD": "횡단보도", "AD": "화살표", "WND": "글자"}
SPLITS = ("train", "valid", "test")
METRICS = ("star", "contrast", "contrast_p25", "paint_len")  # star 는 기준별로 star_dataset 등으로 펼친다
BASELINES = ("dataset", "tracker", "image")
# 비교: 이름 → (멀쩡 쪽 등급, 닳음 쪽 등급)
COMPARISONS = {"0_vs_2": ((0,), (2,)), "0_vs_12": ((0,), (1, 2)), "0_vs_1": ((0,), (1,)), "1_vs_2": ((1,), (2,))}
SOURCE = {
    "name": "RMDD — Road Marking Damage Detection",
    "url": "https://universe.roboflow.com/rmdd/road-marking-damage-detection",
    "version": 7,
    "version_name": "original 1000",
    "license": "MIT (Roboflow 표기. 원 출처·촬영자·등급 기준은 공개 안 됨)",
    "country": "일본",
    "preprocessing": "Roboflow 가 640×640 으로 늘림 (Stretch). 평가 때 16:9 로 되돌림",
}


@dataclass
class Box:
    cls: str  # "LD2" 등
    xyxy: tuple[float, float, float, float]  # 정규화 좌표

    @property
    def group(self) -> str:
        return self.cls[:-1]

    @property
    def grade(self) -> int:
        return int(self.cls[-1])


@dataclass
class ImageItem:
    path: Path
    split: str
    num: int  # 파일 이름 앞 번호 (1000_jpg.rf.… → 1000). 정렬 순서 = 주행 순서로 가정
    boxes: list[Box] = field(default_factory=list)


def parse_label(text: str) -> list[Box]:
    """YOLO 박스 라벨 (class cx cy w h, 정규화) → Box 목록. 다각형 줄(값 5개 초과)은 외접 박스로."""
    out = []
    for line in text.splitlines():
        p = line.split()
        if not p:
            continue
        c, vals = int(p[0]), [float(v) for v in p[1:]]
        if len(vals) == 4:
            cx, cy, w, h = vals
            xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        else:
            xs, ys = vals[0::2], vals[1::2]
            xyxy = (min(xs), min(ys), max(xs), max(ys))
        out.append(Box(NAMES[c], tuple(float(np.clip(v, 0, 1)) for v in xyxy)))
    return out


def image_num(path: Path) -> int:
    m = re.match(r"(\d+)", path.name)
    return int(m.group(1)) if m else -1


def load_items(root: Path) -> list[ImageItem]:
    items = []
    for split in SPLITS:
        for img in sorted((root / split / "images").glob("*.jpg")):
            lab = root / split / "labels" / (img.stem + ".txt")
            boxes = parse_label(lab.read_text()) if lab.exists() else []
            items.append(ImageItem(img, split, image_num(img), boxes))
    items.sort(key=lambda it: (it.num, it.path.name))
    return items


def class_counts(items: list[ImageItem]) -> dict[str, dict[str, dict[str, int]]]:
    """split → 클래스 → {"boxes": 박스 수, "images": 그 클래스가 있는 사진 수}."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for sp in (*SPLITS, "all"):
        rows = {n: {"boxes": 0, "images": 0} for n in NAMES}
        for it in items:
            if sp != "all" and it.split != sp:
                continue
            for b in it.boxes:
                rows[b.cls]["boxes"] += 1
            for n in {b.cls for b in it.boxes}:
                rows[n]["images"] += 1
        out[sp] = rows
    return out


def unstretch(img: np.ndarray, aspect: float) -> np.ndarray:
    h, w = img.shape[:2]
    if aspect <= 0:
        return img
    return cv2.resize(img, (w, round(w / aspect)), interpolation=cv2.INTER_AREA)


_RELAXED = LaneWearMeter(min_lane_pixels=0, min_total_len=0.0)  # 박스 값은 길이 하한 없이 (_finish 만 쓴다)


def _subset(p: dict, keep: list[int]) -> dict:
    q = dict(p)
    for k in ("segs", "lane_vals", "road_vals", "row_vals"):
        q[k] = [p[k][i] for i in keep]
    return q


def seg_midpoints_norm(p: dict) -> np.ndarray:
    """채택 선분 가운데 점 (정규화 좌표, N×2)."""
    (x0, y0), (sw, sh) = p["origin"], p["size"]
    if not p["segs"]:
        return np.zeros((0, 2))
    s = np.array(p["segs"], np.float64)
    return np.stack([((s[:, 0] + s[:, 2]) / 2 + x0) / sw, ((s[:, 1] + s[:, 3]) / 2 + y0) / sh], 1)


def measure_image(meter: LaneWearMeter, frame: np.ndarray, boxes: list[Box],
                  lane_mask: np.ndarray | None = None) -> tuple[dict, list[dict]]:
    """사진 전체를 한 번 재고, 박스마다 가운데 점이 박스 안인 선분만으로 값을 낸다."""
    p = meter._segments(frame, lane_mask)
    fr, _ = meter._finish(p)
    frame_rec = {"found": fr.lane_found, "contrast": fr.contrast, "contrast_p25": fr.contrast_p25,
                 "paint_len": fr.paint_len if fr.lane_found else None, "n_segments": fr.n_segments}
    mids = seg_midpoints_norm(p)
    roi_top = min(y for _, y in meter.roi)
    box_recs = []
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy
        keep = [i for i, (mx, my) in enumerate(mids) if x1 <= mx <= x2 and y1 <= my <= y2]
        rec = {"cls": b.cls, "group": b.group, "grade": b.grade, "box": [round(v, 4) for v in b.xyxy],
               "in_roi": y2 > roi_top, "n_seg": len(keep), "found": bool(keep), "frame_found": fr.lane_found,
               "contrast": None, "contrast_p25": None, "paint_len": None}
        if keep:
            r, _ = _RELAXED._finish(_subset(p, keep))
            if r.lane_found:
                rec.update(contrast=r.contrast, contrast_p25=r.contrast_p25, paint_len=r.paint_len)
            else:
                rec["found"] = False
        box_recs.append(rec)
    return frame_rec, box_recs


def add_baselines(frames: list[dict], boxes: list[dict], window: int = 100, warmup: int = 20) -> None:
    """frames(번호 순) 에 tracker 기준·점수, boxes 에 ★ 점수 3종을 채운다 (제자리)."""
    vals = [f["contrast_p25"] for f in frames if f["found"]]
    ds_med = float(np.median(vals)) if vals else None
    tracker = LaneWearTracker(window=window, warmup=warmup)
    for f in frames:
        f["baseline_tracker"] = tracker.baseline
        f["baseline_dataset"] = ds_med
        if f["found"]:
            j = tracker.update(_FrameResult(f["contrast_p25"], f["paint_len"]))
            f["score_tracker"] = j.score
            f["worn_tracker"] = j.is_worn if j.ready else None
        else:
            f["score_tracker"], f["worn_tracker"] = None, None
        f["score_dataset"] = (f["contrast_p25"] / ds_med * f["paint_len"]) if f["found"] and ds_med else None
    by_img: dict[int, list[dict]] = {}
    for b in boxes:
        by_img.setdefault(b["img"], []).append(b)
    fmap = {f["img"]: f for f in frames}
    for img, bs in by_img.items():
        found = [b["contrast_p25"] for b in bs if b["found"] and b["group"] == "LD"]
        img_med = float(np.median(found)) if found else None
        for b in bs:
            bases = {"dataset": ds_med, "tracker": fmap[img]["baseline_tracker"],
                     "image": img_med if b["group"] == "LD" else None}
            for k, base in bases.items():
                b[f"star_{k}"] = (b["contrast_p25"] / max(base, 1e-3) * b["paint_len"]) \
                    if b["found"] and base is not None else None


@dataclass(frozen=True)
class _FrameResult:  # LaneWearTracker.update 가 읽는 필드만
    contrast_p25: float
    paint_len: float
    lane_found: bool = True


def bootstrap_auc(score: np.ndarray, worn: np.ndarray, cluster: np.ndarray, reps: int = 500,
                  seed: int = 0) -> tuple[float, float, float]:
    """AUC 와 95% 구간 (cluster = 사진 번호 단위로 다시 뽑기)."""
    auc = auc_low_is_worn(score, worn)
    if np.isnan(auc):
        return auc, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    ids = np.unique(cluster)
    idx_of = {c: np.flatnonzero(cluster == c) for c in ids}
    out = []
    for _ in range(reps):
        pick = np.concatenate([idx_of[c] for c in rng.choice(ids, len(ids))])
        a = auc_low_is_worn(score[pick], worn[pick])
        if not np.isnan(a):
            out.append(a)
    lo, hi = (np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan))
    return float(auc), float(lo), float(hi)


def _auc_rows(recs: list[dict], metrics: list[str], grade_key: str, reps: int) -> dict:
    rows: dict[str, dict] = {}
    for m in metrics:
        rows[m] = {}
        for cname, (neg, pos) in COMPARISONS.items():
            sel = [r for r in recs if r.get(m) is not None and r[grade_key] in (*neg, *pos)]
            if not sel:
                continue
            s = np.array([r[m] for r in sel], float)
            w = np.array([r[grade_key] in pos for r in sel])
            c = np.array([r["img"] for r in sel])
            auc, lo, hi = bootstrap_auc(s, w, c, reps=reps)
            rows[m][cname] = {"auc": _r(auc), "ci95": [_r(lo), _r(hi)], "n_intact": int((~w).sum()),
                              "n_worn": int(w.sum())}
    return rows


def operating_points(recs: list[dict], metric: str, grade_key: str = "grade",
                     false_rates: tuple[float, ...] = (0.1, 0.2)) -> list[dict]:
    """기준값을 train 의 등급 0 값 하위 false_rate 분위로 정하고 (= train 에서 멀쩡한 것의 그 비율만 마모로 오판),
    valid+test 에서 등급별로 몇 개를 마모(값 < 기준)로 판정하는지 센다. 못 찾은 표본은 마모로 판정 안 함(시연과 같음)."""
    train0 = [r[metric] for r in recs if r["split"] == "train" and r[grade_key] == 0 and r.get(metric) is not None]
    held = [r for r in recs if r["split"] != "train" and r[grade_key] is not None]
    out = []
    for fr in false_rates:
        if not train0:
            continue
        thr = float(np.percentile(train0, 100 * fr))
        row: dict = {"train_false_rate": fr, "threshold": _r(thr, 4)}
        for g in (0, 1, 2):
            rs = [r for r in held if r[grade_key] == g]
            found = [r for r in rs if r.get(metric) is not None]
            hit = sum(r[metric] < thr for r in found)
            row[str(g)] = {"flagged": hit, "n_found": len(found), "n_all": len(rs)}
        out.append(row)
    return out


def _r(x: float | None, nd: int = 3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def summarize_boxes(boxes: list[dict], group: str, reps: int = 500) -> dict:
    recs = [b for b in boxes if b["group"] == group]
    per_grade = {}
    for g in (0, 1, 2):
        rs = [b for b in recs if b["grade"] == g]
        n = len(rs)
        flagged = [b for b in rs if b.get("star_tracker") is not None]
        per_grade[str(g)] = {
            "n_boxes": n,
            "n_images": len({b["img"] for b in rs}),
            "in_roi": _r(sum(b["in_roi"] for b in rs) / n) if n else None,
            "found_in_box": _r(sum(b["found"] for b in rs) / n) if n else None,
            "found_in_box_and_frame": _r(sum(b["found"] and b["frame_found"] for b in rs) / n) if n else None,
            "median": {m: _r(_median([b.get(m) for b in rs])) for m in ("contrast", "contrast_p25", "paint_len",
                                                                         "star_tracker")},
            # 시연 기준(★ < 1.0, tracker 기준)으로 마모라고 볼 박스 비율 — 차선을 찾고 기준이 준비된 박스 중
            "flagged_worn": _r(sum(b["star_tracker"] < PROVISIONAL_WEAR_SCORE for b in flagged) / len(flagged))
            if flagged else None,
            "n_judged": len(flagged),
        }
    metrics = [f"star_{k}" for k in BASELINES if group == "LD" or k != "image"] + \
        ["contrast", "contrast_p25", "paint_len"]
    return {"per_grade": per_grade, "auc": _auc_rows(recs, metrics, "grade", reps),
            "held_out": {m: operating_points(recs, m) for m in ("contrast", "contrast_p25", "star_tracker")}}


def _median(vals):
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def frame_labels(frames: list[dict], boxes: list[dict]) -> None:
    """사진 단위 등급: LD 박스 최고 등급 (LD 없으면 None). 사진 '닳음' = 2, '멀쩡' = 0."""
    worst: dict[int, int] = {}
    for b in boxes:
        if b["group"] == "LD":
            worst[b["img"]] = max(worst.get(b["img"], 0), b["grade"])
    for f in frames:
        f["grade"] = worst.get(f["img"])


def summarize_frames(frames: list[dict], reps: int = 500) -> dict:
    lab = [f for f in frames if f["grade"] is not None]
    per = {}
    for g in (0, 1, 2):
        rs = [f for f in lab if f["grade"] == g]
        judged = [f for f in rs if f["worn_tracker"] is not None]
        per[str(g)] = {
            "n_images": len(rs),
            "found": _r(sum(f["found"] for f in rs) / len(rs)) if rs else None,
            "flagged_worn": _r(sum(f["worn_tracker"] for f in judged) / len(judged)) if judged else None,
            "n_judged": len(judged),
            "median_score_tracker": _r(_median([f["score_tracker"] for f in rs])),
        }
    metrics = ["score_tracker", "score_dataset", "contrast", "contrast_p25", "paint_len"]
    return {"per_worst_grade": per, "auc": _auc_rows(lab, metrics, "grade", reps),
            "held_out": {m: operating_points(lab, m) for m in ("contrast", "contrast_p25", "score_tracker")},
            "n_no_lane_box": sum(f["grade"] is None for f in frames)}


def run(items: list[ImageItem], meter: LaneWearMeter, aspect: float, segmenter=None,
        progress: bool = False) -> tuple[list[dict], list[dict]]:
    frames, boxes = [], []
    for k, it in enumerate(items):
        img = cv2.imread(str(it.path))
        if img is None:
            continue
        frame = unstretch(img, aspect)
        mask = segmenter.lane_mask(frame) if segmenter is not None else None
        fr, brs = measure_image(meter, frame, it.boxes, mask)
        fr.update(img=it.num, split=it.split, name=it.path.name)
        frames.append(fr)
        for b in brs:
            b.update(img=it.num, split=it.split)
        boxes.extend(brs)
        if progress and k % 100 == 0:
            print(f"  {k}/{len(items)}", file=sys.stderr, flush=True)
    add_baselines(frames, boxes)
    frame_labels(frames, boxes)
    return frames, boxes


def summarize(frames: list[dict], boxes: list[dict], reps: int = 500) -> dict:
    return {"boxes": {g: summarize_boxes(boxes, g, reps) for g in GROUPS},
            "frames": summarize_frames(frames, reps)}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("data/rmdd"))
    ap.add_argument("--json", type=Path, default=Path("experiments/lane_wear_rmdd.json"))
    ap.add_argument("--csv-dir", type=Path, default=Path("data/rmdd_eval"), help="박스·사진별 값 (git 제외)")
    ap.add_argument("--aspect", type=float, default=16 / 9, help="원본 가로세로비 (0 이면 640×640 그대로)")
    ap.add_argument("--seg-model", type=Path, default=None, help="lane_seg_v1.onnx 를 주면 마스크 측정도 함께")
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args(argv)

    items = load_items(a.root)
    if not items:
        print(f"{a.root} 에 사진이 없다", file=sys.stderr)
        return 1
    out = {"source": SOURCE, "n_images": len(items), "class_counts": class_counts(items),
           "method": {"aspect": round(a.aspect, 4), "box_rule": "사진 전체 측정 → 가운데 점이 박스 안인 선분만",
                      "wear_score_threshold": PROVISIONAL_WEAR_SCORE, "tracker_window": 100, "tracker_warmup": 20,
                      "order": "파일 이름 번호 순 (train·valid·test 합침)"},
           "variants": {}}
    variants = {"base": None}
    if a.seg_model is not None:
        from daboyeo_ai.inference.lane_seg import LaneSegmenter
        variants["seg"] = LaneSegmenter(a.seg_model)
    for name, seg in variants.items():
        print(f"[{name}] {len(items)}장 측정", file=sys.stderr)
        frames, boxes = run(items, LaneWearMeter(), a.aspect, seg, progress=True)
        out["variants"][name] = summarize(frames, boxes, a.reps)
        write_csv(a.csv_dir / f"{name}_boxes.csv", boxes)
        write_csv(a.csv_dir / f"{name}_frames.csv", frames)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
