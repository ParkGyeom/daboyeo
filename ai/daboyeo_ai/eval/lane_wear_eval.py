"""차선 마모 판정 점수 비교 — 손 라벨에서 선명/마모가 얼마나 갈리나 (오픈 이슈 #6).

    uv run python -m daboyeo_ai.eval.lane_wear_eval data/lane_calib/labels_claude.json data/lane_calib/frames

라벨 JSON: `{"<drive>/<file>": "선명"|"마모"|"제외"}` 또는 값이 `["선명", ...]` 같은 리스트(첫 원소가 라벨).
경로 첫 폴더(`krid/`, `road/`)를 **한 주행**으로 보고 상대 점수의 기준(baseline)을 그 안에서 만든다.
기준은 라벨을 쓰지 않으므로 `제외` 프레임도 포함한다. 점수 비교에는 선명·마모만 쓰고, 차선을 못 찾은
프레임은 뺀다 (판정 대상이 아님).

점수(낮을수록 마모)마다 출력: AUC, 전체에서 고른 best-F1 과 그 기준값, 2-fold F1 (절반에서 기준값을 고르고
나머지 절반에서 잰 F1 의 평균 — 기준값을 맞춘 데이터에서 잰 F1 은 낙관적이다), 잠정 기준값에서의 F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.lane_wear import (
    PROVISIONAL_THRESHOLD,
    PROVISIONAL_WEAR_SCORE,
    LaneWearMeter,
    LaneWearResult,
    LaneWearTracker,
)

CLEAR, WORN, SKIP = "선명", "마모", "제외"


@dataclass(frozen=True)
class Item:
    key: str
    drive: str
    label: str
    result: LaneWearResult


def load_labels(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        lab = v[0] if isinstance(v, list | tuple) else v
        if lab not in (CLEAR, WORN, SKIP):
            raise ValueError(f"모르는 라벨 {lab!r} ({k})")
        out[k] = lab
    return out


def measure_all(labels: dict[str, str], frames: Path, meter: LaneWearMeter) -> tuple[list[Item], int]:
    items, missing = [], 0
    for key in sorted(labels):
        img = cv2.imread(str(frames / key), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            print(f"건너뜀 (읽기 실패): {frames / key}", file=sys.stderr)
            continue
        items.append(Item(key, key.split("/")[0], labels[key], meter.measure(img)))
    return items, missing


# ---- 지표 ------------------------------------------------------------------------------


def auc_low_is_worn(score: np.ndarray, worn: np.ndarray) -> float:
    """AUC (점수가 낮을수록 마모). 0.5 = 무작위, 1 = 완벽."""
    pos, neg = -score[worn], -score[~worn]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def f1_at(score: np.ndarray, worn: np.ndarray, thr: float) -> float:
    pred = score < thr
    tp = int(np.sum(pred & worn))
    return 2 * tp / max(1, int(pred.sum()) + int(worn.sum()))


def best_f1(score: np.ndarray, worn: np.ndarray) -> tuple[float, float]:
    """(best F1, 기준값). 기준값은 `score < thr` 이 마모가 되도록 이웃한 두 값의 가운데로 둔다."""
    u = np.unique(score)
    cands = np.concatenate([(u[:-1] + u[1:]) / 2, u[-1:] + 1e-9]) if len(u) else u
    best = (0.0, float("nan"))
    for t in cands:
        f = f1_at(score, worn, float(t))
        if f > best[0]:
            best = (f, float(t))
    return best


def two_fold_f1(score: np.ndarray, worn: np.ndarray, fold: np.ndarray) -> float:
    fs = []
    for k in (0, 1):
        fit, test = fold == k, fold != k
        _, t = best_f1(score[fit], worn[fit])
        fs.append(f1_at(score[test], worn[test], t))
    return float(np.mean(fs))


def make_folds(drives: list[str], worn: np.ndarray) -> np.ndarray:
    """주행·라벨마다 번갈아 0/1 (정렬된 키 순서). 두 fold 의 주행·마모 비율이 같아진다."""
    fold = np.zeros(len(drives), int)
    seen: dict[tuple[str, bool], int] = {}
    for i, (d, w) in enumerate(zip(drives, worn)):
        n = seen.get((d, bool(w)), 0)
        fold[i] = n % 2
        seen[(d, bool(w))] = n + 1
    return fold


# ---- 점수 ------------------------------------------------------------------------------


def drive_medians(items: list[Item], field: str) -> dict[str, float]:
    out = {}
    for d in {it.drive for it in items}:
        v = [getattr(it.result, field) for it in items if it.drive == d and it.result.lane_found]
        out[d] = float(np.median(v)) if v else float("nan")
    return out


def online_scores(items: list[Item], tracker_factory: Callable[[], LaneWearTracker]) -> dict[str, float | None]:
    """주행마다 tracker 하나를 정렬된 키 순서로 돌린 점수 (warmup 동안은 None). 실제 사용에 가장 가깝다."""
    out: dict[str, float | None] = {}
    trackers: dict[str, LaneWearTracker] = {}
    for it in items:
        tr = trackers.setdefault(it.drive, tracker_factory())
        out[it.key] = tr.update(it.result).score
    return out


def candidate_scores(items: list[Item], window: int, warmup: int) -> dict[str, tuple[np.ndarray, float | None]]:
    """점수 이름 → (lane_found 프레임별 점수, 잠정 기준값 또는 None). NaN = 판정 안 함 (tracker warmup)."""
    found = [it for it in items if it.result.lane_found]
    r = [it.result for it in found]
    med_c = drive_medians(items, "contrast")
    med_p = drive_medians(items, "contrast_p25")
    tr = LaneWearTracker(window=window, warmup=warmup)
    on = online_scores(items, lambda: LaneWearTracker(window=window, warmup=warmup))

    def a(xs) -> np.ndarray:
        return np.array(xs, float)

    return {
        "contrast (절대)": (a([x.contrast for x in r]), PROVISIONAL_THRESHOLD),
        "contrast ÷ 주행 중앙값": (a([x.contrast / med_c[it.drive] for x, it in zip(r, found)]), None),
        "contrast_p25 ÷ 주행 중앙값": (a([x.contrast_p25 / med_p[it.drive] for x, it in zip(r, found)]), None),
        "n_segments": (a([x.n_segments for x in r]), None),
        "lane_pixel_ratio": (a([x.lane_pixel_ratio for x in r]), None),
        "paint_len": (a([x.paint_len for x in r]), None),
        "contrast ÷ 중앙값 × paint_len": (a([x.contrast / med_c[it.drive] * x.paint_len for x, it in zip(r, found)]), None),
        "★ score (p25 ÷ 중앙값 × paint_len)": (
            a([tr.score(x, med_p[it.drive]) for x, it in zip(r, found)]), PROVISIONAL_WEAR_SCORE),
        "★ score, tracker 순차 (판정한 프레임만)": (
            a([np.nan if on[it.key] is None else on[it.key] for it in found]), PROVISIONAL_WEAR_SCORE),
    }


def report(items: list[Item], window: int = 100, warmup: int = 20) -> str:
    found = [it for it in items if it.result.lane_found]
    use = np.array([it.label != SKIP for it in found])
    worn = np.array([it.label == WORN for it in found])[use]
    drives = [it.drive for it, u in zip(found, use) if u]
    fold = make_folds(drives, worn)
    lines = [(f"프레임 {len(items)}개, 차선 찾음 {len(found)}개, 비교에 쓴 것(선명/마모) {int(use.sum())}개 "
              f"(마모 {int(worn.sum())})")]
    for d in sorted(set(drives)):
        m = np.array([x == d for x in drives])
        lines.append(f"  {d}: {int(m.sum())}개 (마모 {int(worn[m].sum())})")
    lines.append("")
    lines.append("| 점수 (낮을수록 마모) | n (마모) | AUC | best F1 | 기준값 | 2-fold F1 | 잠정 기준 F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, (score, prov) in candidate_scores(items, window, warmup).items():
        ok = ~np.isnan(score[use])
        s, w, fd = score[use][ok], worn[ok], fold[ok]
        f, t = best_f1(s, w)
        prov_f = f"{f1_at(s, w, prov):.3f} (<{prov})" if prov is not None else "-"
        lines.append(f"| {name} | {len(s)} ({int(w.sum())}) | {auc_low_is_worn(s, w):.3f} | {f:.3f} | {t:.3f} | "
                     f"{two_fold_f1(s, w, fd):.3f} | {prov_f} |")
    lines.append("")
    lines.append(f"주행 중앙값 = 그 주행의 차선 프레임 전체(제외 포함) 중앙값. tracker 순차: window={window}, "
                 f"warmup={warmup}, 정렬된 키 순서, warmup 동안의 프레임은 빼고 셌다.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="차선 마모 점수 비교 (손 라벨 대비 AUC / F1)")
    ap.add_argument("labels", type=Path, help="라벨 JSON ({\"drive/file\": \"선명\"|\"마모\"|\"제외\"})")
    ap.add_argument("frames", type=Path, help="프레임 폴더 (라벨 키의 기준 폴더)")
    ap.add_argument("--window", type=int, default=100, help="tracker 기준 창 크기 (기본 100)")
    ap.add_argument("--warmup", type=int, default=20, help="tracker 판정 시작 전 차선 프레임 수 (기본 20)")
    a = ap.parse_args(argv)
    labels = load_labels(a.labels)
    items, missing = measure_all(labels, a.frames, LaneWearMeter())
    if missing:
        print(f"읽지 못한 프레임 {missing}개", file=sys.stderr)
    print(report(items, a.window, a.warmup))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
