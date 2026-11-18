"""차선 마모 발표 자료 — 실시간 시연 대신 보여 줄 측정 화면 (handoff 24-8).

    # 1) 주행 영상 오버레이: 측정한 차선 선을 대비(이 주행 보통 = 1.0)에 따라 색칠하고 아래에 시간 그래프
    uv run python -m daboyeo_ai.eval.lane_wear_figs video --video data/videos/gn_b_3320-3620.mp4 \
        --start 128 --end 148 --out data/presentation/lane_wear/gn_b_128-148.mp4
    # 2) RMDD 실제 마모 등급 예시 + 등급별 대비 그림 (lane_wear_rmdd 를 먼저 돌려 data/rmdd_eval/base_boxes.csv 가 있어야 함)
    uv run python -m daboyeo_ai.eval.lane_wear_figs rmdd --root data/rmdd --boxes data/rmdd_eval/base_boxes.csv \
        --out data/presentation/lane_wear/rmdd_examples.jpg

결과물은 `data/presentation/` (git 제외). 영상은 YouTube 표준 라이선스, RMDD 는 MIT 표기 · 원 출처 불명 → 슬라이드에 출처를 적는다.

색 기준 (화면 설명용, 판정 규칙이 아님): 선분의 대비 ÷ 이 주행 보통 대비 ≥ 0.8 선명 · 0.5~0.8 흐림 · < 0.5 많이 흐림.
"이 주행 보통" = 영상 전체를 3 프레임마다 잰 contrast_p25 중앙값 (`LaneWearTracker` 기준과 같은 뜻, 주행 전체로).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.lane_wear import LaneWearMeter

from .lane_wear_rmdd import load_items, measure_image, unstretch

CLEAR, FAINT = 0.8, 0.5  # 색 구간 (비율)
GREEN, YELLOW, RED = (80, 200, 60), (0, 200, 255), (40, 40, 230)  # BGR
FONT_PATHS = ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf")


def tier_color(ratio: float) -> tuple[int, int, int]:
    return GREEN if ratio >= CLEAR else YELLOW if ratio >= FAINT else RED


def _font(size: int):
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return None


def put_text(img: np.ndarray, xy: tuple[int, int], text: str, size: int = 24,
             color: tuple[int, int, int] = (255, 255, 255), bg: bool = True) -> np.ndarray:
    """한글 글자 (PIL + 시스템 글꼴). 글꼴이 없으면 cv2 (한글은 ? 로 나옴)."""
    font = _font(size)
    if font is None:
        cv2.putText(img, text.encode("ascii", "replace").decode(), (xy[0], xy[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 32, color, 2)
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    if bg:
        x0, y0, x1, y1 = d.textbbox(xy, text, font=font)
        d.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
    d.text(xy, text, font=font, fill=color[::-1])
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def segment_ratios(p: dict, baseline: float) -> list[float]:
    """선분마다 (행별 대비 중앙값 ÷ 기준)."""
    return [float(np.median(rv)) / max(baseline, 1e-3) for rv in p["row_vals"]]


def drive_baseline(video: Path, meter: LaneWearMeter, every: int = 3, seg=None) -> float:
    cap = cv2.VideoCapture(str(video))
    vals, i = [], 0
    while cap.grab():
        if i % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            r = meter.measure(frame, seg.lane_mask(frame) if seg is not None else None)
            if r.lane_found:
                vals.append(r.contrast_p25)
        i += 1
    cap.release()
    if not vals:
        raise SystemExit(f"{video}: 차선을 찾은 프레임이 없다")
    return float(np.median(vals))


def draw_graph(w: int, h: int, hist: list[tuple[float, float | None]], t0: float, t1: float,
               source: str = "") -> np.ndarray:
    """시간 그래프: x = 시간, y = 대비 비율 (0~ymax). 빨강 띠 < 0.5 · 노랑 띠 0.5~0.8 · 1.0 가로선 = 보통."""
    g = np.full((h, w, 3), 24, np.uint8)
    top, bot, left, right = 64, h - 10, 70, w - 10
    ymax = 2.4

    def ypx(v: float) -> int:
        return int(bot - (min(v, ymax) / ymax) * (bot - top))

    cv2.rectangle(g, (left, ypx(FAINT)), (right, bot), (40, 40, 110), -1)
    cv2.rectangle(g, (left, ypx(CLEAR)), (right, ypx(FAINT)), (20, 90, 110), -1)
    cv2.line(g, (left, ypx(1.0)), (right, ypx(1.0)), (150, 150, 150), 1)
    put_text(g, (6, ypx(1.0) - 11), "보통", 16, (220, 220, 220), bg=False)
    put_text(g, (6, ypx(FAINT / 2) - 11), "흐림", 16, (120, 120, 255), bg=False)
    pts = [(int(left + (t - t0) / max(t1 - t0, 1e-6) * (right - left)), ypx(v)) for t, v in hist if v is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(g, a, b, (255, 255, 255), 2, cv2.LINE_AA)
    put_text(g, (10, 6), "차선 대비 (이 주행 보통 = 1.0, 최근 0.25초 중앙값)", 18, (230, 230, 230), bg=False)
    put_text(g, (10, 32), "선 색: 초록 선명 · 노랑 흐림 · 빨강 많이 흐림", 16, (200, 200, 200), bg=False)
    if source:
        put_text(g, (w - 12 - 10 * len(source), 6), source, 16, (200, 200, 200), bg=False)
    return g


def render_video(video: Path, start: float, end: float, out: Path, width: int = 1280, graph_h: int = 220,
                 baseline: float | None = None, smooth: int = 15, source: str = "", seg=None) -> dict:
    """seg: `LaneSegmenter` 를 주면 차선 분할 마스크 안의 선만 잰다 (23장, 노면 글자·차체 광고 선 제거). 기준도 같은 방식으로 잰다."""
    meter = LaneWearMeter()
    if baseline is None:
        baseline = drive_baseline(video, meter, seg=seg)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    writer = None
    recent: deque[float] = deque(maxlen=smooth)
    hist: list[tuple[float, float | None]] = []
    stats = {"frames": 0, "found": 0, "faint": 0, "very_faint": 0}
    i = round(start * fps)
    while i <= round(end * fps):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        p = meter._segments(frame, seg.lane_mask(frame) if seg is not None else None)
        r, segs_px = meter._finish(p)
        h0, w0 = frame.shape[:2]
        s = width / w0
        view = cv2.resize(frame, (width, round(h0 * s)), interpolation=cv2.INTER_AREA)
        cur = None
        if r.lane_found:
            for (x1, y1, x2, y2), ratio in zip(segs_px, segment_ratios(p, baseline)):
                cv2.line(view, (round(x1 * s), round(y1 * s)), (round(x2 * s), round(y2 * s)), tier_color(ratio), 6,
                         cv2.LINE_AA)
            recent.append(r.contrast_p25 / baseline)
            cur = float(np.median(recent))
            stats["found"] += 1
            stats["faint"] += cur < CLEAR
            stats["very_faint"] += cur < FAINT
        stats["frames"] += 1
        hist.append((t, cur))
        if cur is None:
            put_text(view, (16, 14), "차선 못 찾음", 28)
        else:
            label = "선명" if cur >= CLEAR else "흐림" if cur >= FAINT else "많이 흐림"
            put_text(view, (16, 14), f"차선 대비 {cur:.2f}  ·  {label}", 30, tier_color(cur))
        g = draw_graph(width, graph_h, hist, start, end, source)
        frame_out = np.vstack([view, g])
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame_out.shape[1], frame_out.shape[0]))
        writer.write(frame_out)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    try:  # 슬라이드에서 바로 재생되게 H.264 로
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", str(out)], check=True)
        tmp.unlink()
    except (OSError, subprocess.CalledProcessError):
        tmp.replace(out)
    return {"baseline": round(baseline, 4), "fps": fps, **stats}


def pick_examples(rows: list[dict], cls: str, n: int = 3, min_size: float = 0.25) -> list[dict]:
    """valid+test 에서 차선을 찾은 cls 박스(가로·세로 min_size 이상) 중, 그 등급 대비 중앙값에 가까운 n 개 (사진 중복 없이). 극단값을 고르지 않는다."""
    def big(r: dict) -> bool:  # 너무 작은 박스는 확대하면 뭉개져 예시로 안 쓴다
        x1, y1, x2, y2 = json.loads(r["box"])
        return (x2 - x1) >= min_size and (y2 - y1) >= min_size

    cand = [r for r in rows if r["cls"] == cls and r["split"] != "train" and r["contrast"] not in ("", None) and big(r)]
    if not cand:
        return []
    med = float(np.median([float(r["contrast"]) for r in cand]))
    cand.sort(key=lambda r: abs(float(r["contrast"]) - med))
    out, seen = [], set()
    for r in cand:
        if r["img"] in seen:
            continue
        seen.add(r["img"])
        out.append(r)
        if len(out) == n:
            break
    return out


def grade_bars(medians: dict[str, float], w: int, h: int) -> np.ndarray:
    """등급별 대비 중앙값 막대."""
    g = np.full((h, w, 3), 255, np.uint8)
    names = [("LD0", "멀쩡", GREEN), ("LD1", "조금 닳음", YELLOW), ("LD2", "많이 닳음", RED)]
    vmax = max(medians.values()) * 1.15
    bw = w // (len(names) * 2)
    for k, (cls, lab, col) in enumerate(names):
        v = medians[cls]
        x = bw // 2 + k * 2 * bw
        top = int(h - 50 - (v / vmax) * (h - 110))
        cv2.rectangle(g, (x, top), (x + bw, h - 50), col, -1)
        put_text(g, (x, top - 34), f"{v:.2f}", 24, (0, 0, 0), bg=False)
        put_text(g, (x - 10, h - 44), f"{lab} ({cls})", 20, (0, 0, 0), bg=False)
    put_text(g, (10, 6), "차선 대비 중앙값 (높을수록 페인트가 선명)", 20, (0, 0, 0), bg=False)
    return g


def render_rmdd(root: Path, boxes_csv: Path, out: Path, aspect: float = 16 / 9, tile: tuple[int, int] = (420, 300)) -> dict:
    rows = list(csv.DictReader(boxes_csv.open()))
    items = {it.num: it for it in load_items(root)}
    meter = LaneWearMeter()
    tiles_rows = []
    picked = {}
    for cls, lab in (("LD2", "많이 닳음"), ("LD0", "멀쩡")):
        ex = pick_examples(rows, cls)
        picked[cls] = [(r["img"], round(float(r["contrast"]), 3)) for r in ex]
        tiles = []
        for r in ex:
            it = items[int(r["img"])]
            frame = unstretch(cv2.imread(str(it.path)), aspect)
            box = next(b for b in it.boxes if b.cls == cls and [round(v, 4) for v in b.xyxy] == json.loads(r["box"]))
            p = meter._segments(frame)
            _, segs_px = meter._finish(p)
            fr, (br,) = measure_image(meter, frame, [box])
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = box.xyxy
            for (a, b_, c, d) in segs_px:
                mx, my = (a + c) / 2 / w, (b_ + d) / 2 / h
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    cv2.line(frame, (a, b_), (c, d), (255, 0, 255), 2, cv2.LINE_AA)
            col = RED if cls == "LD2" else GREEN
            cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, 2)
            mx, my = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
            cx1, cy1 = int(max(0, x1 - mx) * w), int(max(0, y1 - my) * h)
            cx2, cy2 = int(min(1, x2 + mx) * w), int(min(1, y2 + my) * h)
            crop = cv2.resize(frame[cy1:cy2, cx1:cx2], tile, interpolation=cv2.INTER_AREA)
            put_text(crop, (8, 6), f"라벨 {lab} · 측정 대비 {br['contrast']:.2f}", 22, col)
            tiles.append(crop)
        while len(tiles) < 3:
            tiles.append(np.full((tile[1], tile[0], 3), 255, np.uint8))
        tiles_rows.append(np.hstack(tiles))
    grid = np.vstack(tiles_rows)
    meds = {c: float(np.median([float(r["contrast"]) for r in rows if r["cls"] == c and r["contrast"] not in ("", None)]))
            for c in ("LD0", "LD1", "LD2")}
    bars = grade_bars(meds, 520, grid.shape[0])
    sheet = np.hstack([grid, bars])
    foot = np.full((44, sheet.shape[1], 3), 255, np.uint8)
    put_text(foot, (10, 10), "자료: RMDD (Roboflow Universe, MIT 표기, 일본) · 보라 선 = 우리 측정기가 찾은 차선 · 예시는 각 등급 대비 중앙값에 가까운 사진",
             18, (0, 0, 0), bg=False)
    sheet = np.vstack([sheet, foot])
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"examples": picked, "medians": {k: round(v, 3) for k, v in meds.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("video")
    v.add_argument("--video", type=Path, required=True)
    v.add_argument("--start", type=float, required=True)
    v.add_argument("--end", type=float, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--baseline", type=float, default=None, help="주지 않으면 영상 전체에서 잰다")
    v.add_argument("--source", default="영상: Seoul Walker (YouTube)", help="그래프 오른쪽 위 출처 표기")
    v.add_argument("--seg-model", type=Path, default=None, help="lane_seg_v1.onnx — 주면 분할 마스크 안만 잰다")
    r = sub.add_parser("rmdd")
    r.add_argument("--root", type=Path, default=Path("data/rmdd"))
    r.add_argument("--boxes", type=Path, default=Path("data/rmdd_eval/base_boxes.csv"))
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "video":
        seg = None
        if a.seg_model is not None:
            from daboyeo_ai.inference.lane_seg import LaneSegmenter
            seg = LaneSegmenter(a.seg_model)
        res = render_video(a.video, a.start, a.end, a.out, baseline=a.baseline, source=a.source, seg=seg)
    else:
        res = render_rmdd(a.root, a.boxes, a.out)
    print(res, file=sys.stderr)
    print(f"→ {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
