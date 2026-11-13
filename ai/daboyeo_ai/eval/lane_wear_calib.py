"""차선 마모 기준값 보정 도구 — 실제 프레임에서 contrast 분포를 본다 (오픈 이슈 #6).

    uv run python -m daboyeo_ai.eval.lane_wear_calib data/frames/ --out experiments/lane_wear_calib.csv
    uv run python -m daboyeo_ai.eval.lane_wear_calib data/video1.mp4 --every-s 1.0 --overlay-dir out/overlay

입력은 이미지 폴더(.jpg/.png, 하위 폴더 포함) 또는 영상 파일 하나. 프레임마다
`source, frame, t_s, lane_found, contrast, is_worn, confidence, n_segments, lane_pixel_ratio, ms`
를 CSV 로 쓰고, lane_found 프레임의 contrast 백분위수를 출력한다.

기준값 정하는 법(제안): 선명한 차선 / 마모 차선 프레임을 손으로 몇십 장씩 골라 각각 돌려 보고, 두 분포가
갈리는 지점을 고른다. `--overlay-dir` 로 ROI·검출 선분을 그린 이미지를 보면 잘못 잡힌 프레임을 걸러낼 수 있다.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.lane_wear import PROVISIONAL_THRESHOLD, LaneWearMeter, LaneWearResult

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
FIELDS = ["source", "frame", "t_s", "lane_found", "contrast", "is_worn", "confidence", "n_segments",
          "lane_pixel_ratio", "ms"]


def iter_frames(path: Path, every_s: float = 1.0) -> Iterator[tuple[str, int, float | None, np.ndarray]]:
    """(source, frame 번호, t_s, BGR) — 폴더면 이미지마다, 영상이면 every_s 초마다."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        for i, p in enumerate(files):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"건너뜀 (읽기 실패): {p}", file=sys.stderr)
                continue
            yield str(p.relative_to(path)), i, None, img
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(every_s * fps))
    i = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, img = cap.retrieve()
                if ok:
                    yield path.name, i, i / fps, img
            i += 1
    finally:
        cap.release()


def draw_overlay(img: np.ndarray, meter: LaneWearMeter, r: LaneWearResult, segs) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.polylines(out, [meter.roi_polygon_px(w, h)], True, (255, 200, 0), 2)
    color = (0, 0, 255) if r.is_worn else (0, 255, 0)
    for x1, y1, x2, y2 in segs:
        cv2.line(out, (x1, y1), (x2, y2), color, 2)
    text = (f"contrast={r.contrast:.3f} worn={r.is_worn} conf={r.confidence:.2f} seg={r.n_segments}"
            if r.lane_found else f"lane not found (seg={r.n_segments})")
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def summarize(rows: list[dict], threshold: float) -> str:
    n = len(rows)
    found = [r["contrast"] for r in rows if r["lane_found"]]
    lines = [f"프레임 {n}개, 차선 찾음 {len(found)}개 ({len(found) / max(1, n):.0%})"]
    if found:
        c = np.array(found)
        pct = np.percentile(c, PERCENTILES)
        lines.append("contrast 백분위수: " + ", ".join(f"p{p}={v:.3f}" for p, v in zip(PERCENTILES, pct)))
        lines.append(f"평균 {c.mean():.3f}, 최소 {c.min():.3f}, 최대 {c.max():.3f}")
        lines.append(f"잠정 기준 {threshold} 미만(마모): {(c < threshold).sum()}개 ({(c < threshold).mean():.0%})")
    ms = [r["ms"] for r in rows]
    if ms:
        lines.append(f"measure 시간: 중앙값 {np.median(ms):.2f} ms, p95 {np.percentile(ms, 95):.2f} ms")
    return "\n".join(lines)


def run(src: Path, out: Path, every_s: float, threshold: float, overlay_dir: Path | None) -> list[dict]:
    meter = LaneWearMeter(threshold=threshold)
    out.parent.mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for source, i, t_s, img in iter_frames(src, every_s):
            t = time.perf_counter()
            r, segs = meter.measure_debug(img)
            ms = (time.perf_counter() - t) * 1000
            row = dict(source=source, frame=i, t_s=None if t_s is None else round(t_s, 3),
                       lane_found=r.lane_found, contrast=r.contrast, is_worn=r.is_worn,
                       confidence=round(r.confidence, 4), n_segments=r.n_segments,
                       lane_pixel_ratio=round(r.lane_pixel_ratio, 5), ms=round(ms, 3))
            wr.writerow(row)
            rows.append(row)
            if overlay_dir:
                name = Path(source).stem + (f"_{i:06d}" if t_s is not None else "") + ".jpg"
                cv2.imwrite(str(overlay_dir / name), draw_overlay(img, meter, r, segs))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="차선 마모 contrast 분포 측정 (기준값 보정용)")
    ap.add_argument("src", type=Path, help="이미지 폴더 또는 영상 파일")
    ap.add_argument("--out", type=Path, default=Path("experiments/lane_wear_calib.csv"))
    ap.add_argument("--every-s", type=float, default=1.0, help="영상일 때 몇 초마다 한 프레임 (기본 1.0)")
    ap.add_argument("--threshold", type=float, default=PROVISIONAL_THRESHOLD, help="잠정 기준값 (TBD, 이슈 #6)")
    ap.add_argument("--overlay-dir", type=Path, default=None, help="ROI·검출 선분을 그린 이미지 저장 폴더")
    a = ap.parse_args(argv)
    if not a.src.exists():
        ap.error(f"없는 경로: {a.src}")
    rows = run(a.src, a.out, a.every_s, a.threshold, a.overlay_dir)
    print(summarize(rows, a.threshold))
    print(f"CSV: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
