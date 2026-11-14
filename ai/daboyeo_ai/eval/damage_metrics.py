"""시설물 파손 분류 평가: val 에서 임계값 고르기, test 지표 (계획 2.7).

양성(positive) = damaged. 점수 p = 모델이 낸 P(damaged). `p >= threshold` 이면 '파손 의심'.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np


def pick_threshold(y_true: Sequence[int], p: Sequence[float]) -> tuple[float, float]:
    """F1 이 가장 큰 임계값 → (threshold, f1). 같은 F1 이면 더 높은(오탐이 적은) 값을 고른다.

    후보는 관측된 점수 값들이다. 양성이 없으면 (0.5, 0.0).
    """
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(p, dtype=float)
    if not y.any():
        return 0.5, 0.0
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(s):  # 오름차순 → '>=' 로 갱신하면 같은 F1 중 큰 t 가 남는다
        pred = s >= t
        tp = int((pred & y).sum())
        f1 = 2 * tp / (int(pred.sum()) + int(y.sum()))
        if f1 >= best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def binary_report(y_true: Sequence[int], p: Sequence[float], threshold: float) -> dict:
    """혼동 행렬과 precision / recall / F1 / normal 오탐률(FPR)."""
    y = np.asarray(y_true, dtype=bool)
    pred = np.asarray(p, dtype=float) >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = int((~pred & ~y).sum())

    def div(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    return {
        "n_damaged": tp + fn, "n_normal": fp + tn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr_normal": div(fp, fp + tn),
    }


def report_by_group(
    y_true: Sequence[int], p: Sequence[float], threshold: float, groups: Sequence[str | None],
) -> dict[str, dict]:
    """전체("all") + 그룹(시설물 종류 등)별 `binary_report`. 그룹이 None 인 샘플은 "unknown"."""
    y, s = np.asarray(y_true), np.asarray(p, dtype=float)
    g = np.asarray([x if x is not None else "unknown" for x in groups], dtype=object)
    out = {"all": binary_report(y, s, threshold)}
    for name in sorted(set(g.tolist())):
        m = g == name
        out[name] = binary_report(y[m], s[m], threshold)
    return out


def score_folder(
    root: Path, split: str, prob_fn: Callable[[np.ndarray], float], exts: tuple[str, ...] = (".png", ".jpg"),
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """`root/<split>/{normal,damaged}/*` 이미지마다 prob_fn(BGR) = P(damaged) → (이름들, y(1=damaged), p)."""
    names, y, p = [], [], []
    for label in ("damaged", "normal"):
        for f in sorted((Path(root) / split / label).glob("*")):
            if f.suffix.lower() not in exts:
                continue
            img = cv2.imread(str(f))
            if img is None:
                raise ValueError(f"cannot read {f}")
            names.append(f.stem)
            y.append(1 if label == "damaged" else 0)
            p.append(float(prob_fn(img)))
    return names, np.asarray(y, dtype=int), np.asarray(p, dtype=float)


def compare_probs(ours: Sequence[float], ref: Sequence[float], threshold: float) -> dict:
    """같은 ONNX 를 우리 전처리와 Ultralytics 로 돌린 P(damaged) 비교 (일치 확인)."""
    a, b = np.asarray(ours, dtype=float), np.asarray(ref, dtype=float)
    d = np.abs(a - b)
    return {
        "n": int(len(a)),
        "max_abs_diff": round(float(d.max()), 5) if len(a) else None,
        "mean_abs_diff": round(float(d.mean()), 5) if len(a) else None,
        "label_agree": round(float(((a >= threshold) == (b >= threshold)).mean()), 4) if len(a) else None,
    }
