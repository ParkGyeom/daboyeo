"""도로장애물(179) 국내 test 사진 단위 지표 — RDD2022(handoff 22장)와 같은 정의·기준·코드로 재고, 새는 값·기저 비율을 같이 낸다.

    uv run python -m daboyeo_ai.eval.kr179_image \\
        --zip /Users/parkgyeom/daboyeo/data/aihub/parts/road_v3.zip \\
        --base-zip /Users/parkgyeom/daboyeo/data/aihub/parts/road.zip \\
        --model /Users/parkgyeom/daboyeo/ai/models/detector_v3.onnx --card ai/models/detector_v3.json \\
        --cache /Users/parkgyeom/daboyeo/data/presentation/kr_road179 \\
        --rdd-preds /Users/parkgyeom/daboyeo/data/presentation/rdd_pothole/preds.json \\
        --out experiments/image_level_kr179_v3.json

추론은 `rdd_figs.run_detector` (0.05 로 한 번, `<cache>/preds.json` 에 캐시), 지표는 `model_gate.image_report` 를 그대로 쓴다.
여기서 더하는 것은 ① 95% 구간(Wilson) ② 기저 비율·결함 없는 사진에서 헛것이 뜬 비율 ③ 학습(train·val)과 겹치는 사진을 뺀 값
④ 포트홀 크기별 재현율뿐이다. 겹침 규칙은 결과를 보기 전에 정했다:

- `no_shared_clip`: 파일 이름의 촬영 날짜+시각(영상 하나, 카메라 채널 무시)이 train·val 에 없는 test 사진만
- `no_near_dup`: 64×36 회색조 정규화 상관이 train·val 사진(v2 학습 묶음 `--base-zip` 포함) 중 하나와 0.90 이상이면 뺀다
  (같은 장소를 다른 영상에서 찍은 사진은 다 못 거른다 — 여전히 새는 값)

결과 JSON 에는 집계 숫자만 들어간다 (사진 이름·이미지 없음).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .conf_thresholds import extract_test
from .model_gate import IMAGE_DEFINITIONS, RDD_CLASSES, image_report
from .rdd_figs import filter_preds, match_marks, run_detector

NEAR_DUP = 0.90
# 긴 변 640 으로 맞춘 사진에서 포트홀 정답 박스 √넓이 (px). COCO small 경계 32 px 와 그 절반
SIZE_BINS = (("<16px", 0.0, 16.0), ("16~32px", 16.0, 32.0), (">=32px", 32.0, math.inf))
_CLIP_RE = re.compile(r"_HY_\d+_(\d{8})_(\d{6})_")


# ---------- 순수 함수 (테스트 대상) ----------

def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    """k/n 의 Wilson 95% 구간 [아래, 위]. n = 0 이면 None."""
    if not n:
        return None
    p, z2 = k / n, z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [round(max(0.0, mid - half), 3), round(min(1.0, mid + half), 3)]


def clip_of(name: str) -> str:
    """'ro_V3F_HY_4751_20201217_101520_N_CH1_…' → '20201217_101520' (촬영 영상의 날짜·시각)."""
    m = _CLIP_RE.search(name)
    if not m:
        raise ValueError(f"179 파일 이름이 아니다: {name}")
    return f"{m[1]}_{m[2]}"


def descriptor(gray: np.ndarray) -> np.ndarray:
    """비슷한 사진 찾기용: 64×36 회색조를 평균 0·길이 1 로. 두 벡터의 내적 = 정규화 상관."""
    import cv2

    d = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    d -= d.mean()
    return d / (np.linalg.norm(d) + 1e-6)


def composition(images: Sequence[dict]) -> dict:
    has = [{g["cls"] for g in im["gts"]} for im in images]
    n = len(images)
    out = {"images": n, "pothole_images": sum("pothole" in s for s in has), "crack_images": sum("crack" in s for s in has),
           "both": sum({"pothole", "crack"} <= s for s in has), "neither": sum(not s for s in has)}
    out["base_rate"] = {c: round(out[f"{c}_images"] / n, 3) if n else None for c in RDD_CLASSES}
    return out


def add_intervals(rep: dict) -> dict:
    """image_report 결과에 Wilson 구간·기저 비율·'그 결함이 없는 사진에서 경보가 뜬 비율'을 붙인다."""
    for cls in RDD_CLASSES:
        b, i = rep[cls]["box_iou50"], rep[cls]["image"]
        b["P_ci95"], b["R_ci95"] = wilson(b["tp"], b["tp"] + b["fp"]), wilson(b["tp"], b["n_gt"])
        i["recall_ci95"] = wilson(i["hit_images"], i["gt_images"])
        i["precision_ci95"] = wilson(i["hit_images"], i["alert_images"])
        i["recall_overlap_ci95"] = wilson(i["hit_images_overlap"], i["gt_images"])
        i["precision_overlap_ci95"] = wilson(i["hit_images_overlap"], i["alert_images"])
        i["base_rate"] = round(i["gt_images"] / i["images"], 3) if i["images"] else None
        neg, fa = i["images"] - i["gt_images"], i["alert_images"] - i["hit_images"]
        i["negative_images"], i["false_alarm_images"] = neg, fa
        i["false_alarm_rate"] = round(fa / neg, 3) if neg else None  # 기저 비율의 영향을 받지 않는다
        i["false_alarm_rate_ci95"] = wilson(fa, neg)
    return rep


def precision_at(recall: float, false_alarm_rate: float, base_rate: float) -> float | None:
    """같은 재현율·헛것 비율에서 결함 사진 비율이 base_rate 였다면 나왔을 사진 단위 정밀도 (베이즈)."""
    tp, fp = recall * base_rate, false_alarm_rate * (1 - base_rate)
    return round(tp / (tp + fp), 3) if tp + fp else None


def add_precision_at_base(rep: dict, base: dict[str, float]) -> dict:
    for cls in RDD_CLASSES:
        i = rep[cls]["image"]
        if i["gt_images"] and i["negative_images"]:
            i["precision_at_rdd_base_rate"] = precision_at(i["hit_images"] / i["gt_images"],
                                                           i["false_alarm_images"] / i["negative_images"], base[cls])
    return rep


def size_px(im: dict, box: Sequence[float]) -> float:
    s = 640 / max(im["w"], im["h"])
    return math.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])) * s


def recall_by_size(images: Sequence[dict], class_conf: dict[str, float], cls: str = "pothole") -> dict:
    """크기별 ① 박스 재현율 (IoU 0.5) ② 사진 재현율 (위치 무관 / 그 크기 박스와 겹침; 사진은 가장 큰 정답 박스로 나눔)."""
    out = {name: {"boxes": 0, "found": 0, "images": 0, "hit_images": 0, "hit_images_overlap_largest": 0}
           for name, _, _ in SIZE_BINS}

    def bin_of(px: float) -> dict:
        return out[next(n for n, lo, hi in SIZE_BINS if lo <= px < hi)]

    for im in images:
        gts = [g for g in im["gts"] if g["cls"] == cls]
        if not gts:
            continue
        preds = filter_preds(im["preds"], {cls: class_conf[cls]})
        _, hit = match_marks(preds, im["gts"], cls)
        for g, h in zip(gts, hit):
            r = bin_of(size_px(im, g["box"]))
            r["boxes"] += 1
            r["found"] += h
        big = max(gts, key=lambda g: size_px(im, g["box"]))
        r = bin_of(size_px(im, big["box"]))
        a = big["box"]
        r["images"] += 1
        r["hit_images"] += bool(preds)
        r["hit_images_overlap_largest"] += any(
            min(a[2], p["box"][2]) > max(a[0], p["box"][0]) and min(a[3], p["box"][3]) > max(a[1], p["box"][1]) for p in preds)
    for r in out.values():
        r["box_recall"] = round(r["found"] / r["boxes"], 3) if r["boxes"] else None
        r["box_recall_ci95"] = wilson(r["found"], r["boxes"])
        r["image_recall"] = round(r["hit_images"] / r["images"], 3) if r["images"] else None
        r["image_recall_ci95"] = wilson(r["hit_images"], r["images"])
    return out


def report(images: Sequence[dict], class_conf: dict[str, float]) -> dict:
    return {"composition": composition(images), **add_intervals(image_report(images, class_conf)),
            "pothole_by_size": recall_by_size(images, class_conf)}


def subsets(images: Sequence[dict], learned_clips: set[str], best_sim: dict[str, float]) -> dict[str, list[dict]]:
    """겹침을 뺀 부분집합과, 반대로 겹치는 쪽."""
    return {"all": list(images),
            "no_shared_clip": [im for im in images if clip_of(im["name"]) not in learned_clips],
            "no_near_dup": [im for im in images if best_sim[im["name"]] < NEAR_DUP],
            "near_dup_only": [im for im in images if best_sim[im["name"]] >= NEAR_DUP]}


# ---------- 파일 ----------

def zip_images(path: Path):
    """det/images/<split>/<이름>.jpg → (이름, split, jpg 바이트)."""
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            p = n.split("/")
            if len(p) == 4 and p[:2] == ["det", "images"] and p[3].endswith(".jpg"):
                yield p[3][:-4], p[2], z.read(n)


def overlap_info(zips: Sequence[Path]) -> tuple[set[str], dict[str, float], dict]:
    """(train·val 의 촬영 영상, test 사진별 train·val 사진과의 최대 상관, 집계). zips[0] 의 test 가 대상."""
    import cv2

    test, learned, clips, meta = {}, {}, set(), []
    for k, zp in enumerate(zips):
        for name, split, data in zip_images(zp):
            d = descriptor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE))
            if split != "test":
                learned[f"{k}:{name}"] = d
                clips.add(clip_of(name))
            elif k == 0:
                test[name] = d
        meta.append(zp.name)
    best = (np.stack(list(test.values())) @ np.stack(list(learned.values())).T).max(1)
    sim = {n: float(b) for n, b in zip(test, best)}
    days = {c[:8] for c in clips}
    info = {"learned_zips": meta, "learned_images": len(learned),
            "test_images": len(test), "test_clips": len({clip_of(n) for n in test}),
            "test_images_sharing_clip": sum(clip_of(n) in clips for n in test),
            "test_images_sharing_date": sum(clip_of(n)[:8] in days for n in test),
            "near_dup_threshold": NEAR_DUP,
            "test_images_by_best_similarity": {f">={t}": int((best >= t).sum()) for t in (0.8, 0.9, 0.95, 0.97, 0.99)}}
    return clips, sim, info


def print_report(tag: str, rep: dict) -> None:
    print(f"[{tag}] {rep['composition']}")
    for cls in RDD_CLASSES:
        b, i = rep[cls]["box_iou50"], rep[cls]["image"]
        print(f"  {cls:8s} 사진 P {i['precision']} {i['precision_ci95']} (기저 {i['base_rate']}) R {i['recall']} {i['recall_ci95']} | "
              f"겹침 P {i['precision_overlap']} R {i['recall_overlap']} | 박스 P {b['P']} {b['P_ci95']} R {b['R']} {b['R_ci95']} | "
              f"정답 {i['gt_images']} 경보 {i['alert_images']} 맞음 {i['hit_images']} | 없는 사진 헛것 {i['false_alarm_images']}/{i['negative_images']}")
    for k, r in rep["pothole_by_size"].items():
        print(f"    {k:8s} 박스 {r['found']}/{r['boxes']} = {r['box_recall']} · 사진 {r['hit_images']}/{r['images']} = {r['image_recall']}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", type=Path, required=True, help="road_v3.zip")
    ap.add_argument("--base-zip", type=Path, action="append", default=[], help="겹침을 볼 때 같이 볼 학습 묶음 (v2 의 road.zip)")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--card", type=Path, required=True, help="class_conf_draft 를 읽을 모델 카드")
    ap.add_argument("--cache", type=Path, required=True, help="preds.json 을 둘 git 밖 폴더")
    ap.add_argument("--rdd-preds", type=Path, help="rdd_figs preds.json (나란히 놓을 RDD2022)")
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    from daboyeo_ai.inference import OnnxDetector, load_class_conf

    conf = {c: v for c, v in load_class_conf(a.card).items() if c in RDD_CLASSES}
    a.cache.mkdir(parents=True, exist_ok=True)
    cache = a.cache / "preds.json"
    if a.redo or not cache.exists():
        with tempfile.TemporaryDirectory(prefix="kr179_") as tmp:
            images = run_detector(OnnxDetector(a.model, threads=a.threads, conf=a.low_conf), extract_test(a.zip, Path(tmp)))
        cache.write_text(json.dumps({"model": a.model.name, "low_conf": a.low_conf, "images": images}))
    else:
        images = json.loads(cache.read_text())["images"]
        print(f"[cache] {cache} ({len(images)} images)", file=sys.stderr)

    clips, sim, info = overlap_info([a.zip, *a.base_zip])
    rep = {"model": a.model.name, "class_conf": conf, "definitions": IMAGE_DEFINITIONS, "interval": "Wilson 95%",
           "size": "긴 변 640 px 기준 정답 박스 √넓이. 사진은 가장 큰 포트홀 정답 박스로 나눔", "overlap": info,
           "kr179": {k: report(v, conf) for k, v in subsets(images, clips, sim).items()}}
    if a.rdd_preds:
        rep["rdd2022"] = report(json.loads(a.rdd_preds.read_text())["images"], conf)
        base = {c: r["gt_images"] / r["images"] for c in RDD_CLASSES for r in [rep["rdd2022"][c]["image"]]}
        for r in rep["kr179"].values():
            add_precision_at_base(r, base)
    print(json.dumps(info, ensure_ascii=False))
    for k, r in rep["kr179"].items():
        print_report(f"179 {k}", r)
    if a.rdd_preds:
        print_report("RDD2022", rep["rdd2022"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
