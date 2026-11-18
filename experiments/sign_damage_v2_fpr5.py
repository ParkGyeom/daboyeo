"""표지판 파손 v2 보조 지표 (handoff 25-3): 임계값에 덜 흔들리는 비교.

- AUC: KRID test 표지판 전부 (파손 9 · 정상 3,303) — 임계값 없이 '파손이 정상보다 높은 점수일 확률'
- 오탐 5% 운전점: KRID **val** 정상 표지판 전부(4,356)에서 5% 만 넘는 점수를 임계값으로 → test 에서 몇 개 잡나
  (val 파손 9개로 F1 최대 임계값을 고르면 0.000~0.882 로 크게 흔들려서)

    uv run --with onnxruntime python experiments/sign_damage_v2_fpr5.py --krid-cls data/sign_damage_v2/krid/cls \
        --work data/sign_damage_v2/runs --json experiments/sign_damage_v2_local.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from daboyeo_ai.data.damage_cls import crop_type, read_type_table
from daboyeo_ai.eval.damage_metrics import binary_report, score_folder
from daboyeo_ai.inference import DamageClassifier


def auc(y, p) -> float:
    y, p = np.asarray(y, bool), np.asarray(p, float)
    pos, neg = p[y], p[~y]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--krid-cls", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--fpr", type=float, default=0.05)
    args = ap.parse_args()
    table = read_type_table()
    res = json.loads(args.json.read_text())
    for name, r in res.items():
        clf = DamageClassifier(r["onnx"], threads=4)
        out = {}
        for split in ("val", "test"):
            names, y, p = score_folder(args.krid_cls, split, clf.prob_damaged_crop, exts=(".jpg",))
            m = np.array([crop_type(n, table) == "sign" for n in names])
            out[split] = (y[m], p[m])
        yv, pv = out["val"]
        thr = float(np.quantile(pv[yv == 0], 1 - args.fpr))
        yt, pt = out["test"]
        r["sign_aux"] = {"auc_test": round(auc(yt, pt), 4), "auc_val": round(auc(yv, pv), 4),
                         "fpr5_threshold_from_val_normal": round(thr, 5),
                         "test_at_fpr5": binary_report(yt, pt, thr), "val_at_fpr5": binary_report(yv, pv, thr)}
        t = r["sign_aux"]["test_at_fpr5"]
        print(f"{name:10} AUC test {r['sign_aux']['auc_test']} val {r['sign_aux']['auc_val']} | @val-FPR5 thr={thr:.4f}: "
              f"test caught {t['tp']}/{t['n_damaged']} false {t['fp']}/{t['n_normal']} ({t['fpr_normal']}) P={t['precision']}")
    args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
