"""표지판 파손 분류 v2 — Mac 로컬 비교 학습 (handoff 25장).

Colab 노트북 03 과 같은 설정(YOLO11n-cls 224, 60 epoch, 조기 종료 없음, last.pt, 같은 증강)으로 여러 데이터 조합을
학습하고, **KRID val 최대 F1 임계값 → KRID test** 로 v1b 와 같은 지표를 낸다. 조합 고르기는 val 로만 한다.

    uv run --with ultralytics --with onnx --with onnxslim python experiments/sign_damage_v2_local.py \
        --krid-cls data/sign_damage_v2/krid/cls --rf-root data/sign_damage_v2 --work data/sign_damage_v2/runs \
        --json experiments/sign_damage_v2_local.json

variants (이름: facility, Roboflow 사용, KRID train normal 배수):
  krid_all  : all,  no,  5   ← v1b 와 같은 데이터 (로컬 학습 기준선)
  krid_sign : sign, no,  5
  rf_all    : all,  yes, 5
  rf_sign   : sign, yes, 5
  rf_sign20 : sign, yes, 20  (KRID 정상을 더 넣어 오탐을 줄여 보기)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from daboyeo_ai.data.damage_cls import crop_type, read_type_table
from daboyeo_ai.data.sign_damage import build_cls_v2, rf_holdout_files
from daboyeo_ai.eval.damage_metrics import binary_report, pick_threshold, report_by_group, score_folder

VARIANTS = {
    "krid_all": ("all", False, 5),
    "krid_sign": ("sign", False, 5),
    "rf_all": ("all", True, 5),
    "rf_sign": ("sign", True, 5),
    "rf_sign20": ("sign", True, 20),
}
TRAIN = dict(epochs=60, patience=0, imgsz=224, batch=64, seed=0, workers=2, fliplr=0.5, flipud=0.0,
             scale=0.3, auto_augment="randaugment", erasing=0.0, hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, plots=False)
CROP = {"pad": 0.1, "resize_max": 224, "min_side_px": 24}


def evaluate(onnx_path: Path, cls_root: Path, krid_cls: Path, rf_root: Path, facility: str, table: dict) -> dict:
    from daboyeo_ai.inference import DamageClassifier

    import cv2

    clf = DamageClassifier(onnx_path, threads=4)
    scores = {s: score_folder(cls_root, s, clf.prob_damaged_crop) for s in ("val", "test")}
    _, y_v, p_v = scores["val"]
    thr, val_f1 = pick_threshold(y_v, p_v)
    out = {"threshold": thr, "val_max_f1": round(val_f1, 4)}
    for split in ("val", "test"):
        names, y, p = scores[split]
        out[split] = report_by_group(y, p, thr, [crop_type(n, table) for n in names])
    # test 정상 크롭 전부 (v1b 의 4,848장 오탐률과 같은 잣대) — 표지판만 모델이어도 가드레일 값은 참고로 낸다
    names_all, y_all, p_all = score_folder(krid_cls, "test", clf.prob_damaged_crop, exts=(".jpg",))
    types = np.array([crop_type(n, table) or "unknown" for n in names_all])
    full = {}
    for k in ("all", "guardrail", "sign"):
        m = (y_all == 0) & (np.ones(len(types), bool) if k == "all" else types == k)
        full[k] = {"n": int(m.sum()), "fpr": round(float((p_all[m] >= thr).mean()), 4)}
    out["test_full_normal_fpr"] = full
    # test 전체(damaged 전부 + normal 전부) 표지판만 — 발표용 'N개 중 M개'
    ms = types == "sign"
    out["test_full_sign"] = binary_report(y_all[ms], p_all[ms], thr)
    # Roboflow holdout (같은 도메인 — 모델이 '파손'을 배우긴 했는지)
    hy, hp = [], []
    for f, yy in rf_holdout_files(rf_root):
        hy.append(yy)
        hp.append(clf.prob_damaged_crop(cv2.imread(str(f))))
    if hy:
        out["rf_holdout"] = binary_report(hy, hp, thr)
        out["rf_holdout_own_thr"] = binary_report(hy, hp, pick_threshold(hy, hp)[0])
    out["_scores"] = {"test_names": list(names_all), "test_y": y_all.tolist(), "test_p": [round(float(x), 5) for x in p_all]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--krid-cls", type=Path, required=True)
    ap.add_argument("--rf-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    args.work = args.work.resolve()  # Ultralytics 는 상대 project 를 runs/classify/ 아래로 보낸다
    from ultralytics import YOLO

    table = read_type_table()
    results = json.loads(args.json.read_text()) if args.json.exists() else {}
    for name in args.variants.split(","):
        facility, use_rf, ratio = VARIANTS[name]
        cls_root = args.work / f"data_{name}"
        counts = build_cls_v2(args.krid_cls, args.rf_root if use_rf else None, cls_root, facility,
                              krid_normal_ratio={"train": ratio}, types_table=table)
        print(name, counts, flush=True)
        run = args.work / name
        last = run / "weights" / "last.pt"
        t = time.time()
        if not last.exists():
            YOLO("yolo11n-cls.pt").train(data=str(cls_root), project=str(args.work), name=name, exist_ok=True,
                                         device=args.device, **{**TRAIN, "epochs": args.epochs})
        train_min = round((time.time() - t) / 60, 1)
        m = YOLO(str(last))
        classes = [m.names[i] for i in range(len(m.names))]
        onnx = Path(m.export(format="onnx", imgsz=224, opset=17, simplify=True, dynamic=False))
        onnx.with_suffix(".json").write_text(json.dumps({"classes": classes, "threshold": 0.5, "crop": CROP}))
        r = evaluate(onnx, cls_root, args.krid_cls, args.rf_root, facility, table)
        (run / "test_scores.json").write_text(json.dumps(r.pop("_scores")))  # 예시 그림용 (커밋 안 함)
        results[name] = {"facility": facility, "roboflow": use_rf, "krid_train_normal_ratio": ratio,
                         "epochs": args.epochs, "train_min": train_min, "counts": counts, "onnx": str(onnx), **r}
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        s, f = r["test"].get("sign", {}), r["test_full_normal_fpr"]
        print(f"[{name}] thr={r['threshold']:.3f} val_f1={r['val_max_f1']} | test all F1={r['test']['all']['f1']} "
              f"sign F1={s.get('f1')} P={s.get('precision')} R={s.get('recall')} | full FPR all={f['all']['fpr']} "
              f"sign={f['sign']['fpr']} grd={f['guardrail']['fpr']} | rf_holdout F1={r.get('rf_holdout', {}).get('f1')}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
