"""새 후보 영상(GPS 없음)에서 시연 모델 v3 탐지가 쓸 만한지 보기 — 프레임별 탐지 + 아래 OSD 띠 잘라 내기 비교 (Mac, ONNX).

    # 1) 돌리기: --every-s 간격 프레임에서 전체 프레임 / 아래 --crop-bottom px 을 잘라 낸 프레임 두 번 탐지 → dets.json
    uv run python experiments/bbcsi_v3_check.py run --video data/videos/bbcsi/NSUEkDQl3m4_1080p30.mp4 \\
        --out-dir data/presentation/bbcsi --crop-bottom 30
    # 2) 박스 그린 프레임: 원하는 시각(초)을 골라 그린다
    uv run python experiments/bbcsi_v3_check.py draw --out-dir data/presentation/bbcsi --t 12.5 40.0

- 낮은 기준(--low-conf 0.1)으로 모두 저장하고, 집계·그림은 모델 카드 class_conf(기본 detector_v3.json) 이상만 쓴다.
- 전체 프레임 vs 잘라 낸 프레임: 같은 클래스 IoU >= 0.5 면 같은 탐지. 한쪽에만 있는 것을 따로 센다.
- 영상·프레임은 YouTube 표준 라이선스라 레포 밖(--out-dir)에만 둔다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

COLORS = {"pothole": (0, 0, 255), "crack": (0, 200, 255), "guardrail": (255, 160, 0), "sign": (200, 0, 200)}


def read_frames(video: Path, every_s: float):
    """every_s 간격 프레임 (t_s, frame). daboyeo_ai.eval.demo_clip.read_frames 와 같은 방식."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, nxt = 0, 0.0
    while cap.grab():
        t = i / fps
        if t + 1e-6 >= nxt:
            ok, f = cap.retrieve()
            if ok:
                yield round(t, 3), f
            nxt += every_s
        i += 1
    cap.release()


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def above(dets: list, thr: list[float]) -> list:
    return [d for d in dets if d[1] >= thr[d[0]]]


def only_in(a: list, b: list, min_iou: float = 0.5) -> list:
    """a 에만 있는 탐지 (b 의 같은 클래스와 IoU < min_iou)."""
    return [d for d in a if not any(e[0] == d[0] and iou(d[2:], e[2:]) >= min_iou for e in b)]


def segments(times: list[float], gap: float) -> list[tuple[float, float, int]]:
    """정렬된 시각들을 gap 이하 간격으로 이어 (시작, 끝, 개수) 구간으로."""
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap + 1e-6:
            s, _, n = out[-1]
            out[-1] = (s, t, n + 1)
        else:
            out.append((t, t, 1))
    return out


def window_counts(frames: dict[str, list], thr: list[float], win_s: float, step_s: float, dur: float) -> list[dict]:
    """win_s 길이 창을 step_s 씩 옮기며 클래스별 탐지 프레임 수."""
    rows, t0 = [], 0.0
    ts = {float(k): above(v, thr) for k, v in frames.items()}
    while t0 + win_s <= dur + 1e-6:
        c = Counter()
        for t, ds in ts.items():
            if t0 <= t < t0 + win_s:
                for name in {CLASS_NAMES[d[0]] for d in ds}:
                    c[name] += 1
        rows.append({"start": round(t0, 1), "end": round(t0 + win_s, 1)} | {k: c[k] for k in CLASS_NAMES})
        t0 += step_s
    return rows


def draw(img: np.ndarray, dets: list, crop_bottom: int = 0) -> np.ndarray:
    out = img.copy()
    if crop_bottom:
        h = out.shape[0]
        cv2.line(out, (0, h - crop_bottom), (out.shape[1], h - crop_bottom), (80, 80, 80), 1)
    for c, s, x1, y1, x2, y2 in dets:
        name = CLASS_NAMES[c]
        col = COLORS[name]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
        lab = f"{name} {s:.2f}"
        cv2.putText(out, lab, (int(x1), max(18, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, lab, (int(x1), max(18, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out


def _thr(card: Path) -> list[float]:
    from daboyeo_ai.inference import load_class_conf
    from daboyeo_ai.inference.onnx_detector import class_conf_list

    return class_conf_list(load_class_conf(card))


def cmd_run(args) -> int:
    from daboyeo_ai.inference import OnnxDetector, road_tiles

    windows = road_tiles(args.road_tiles, bottom=args.tiles_bottom) if args.road_tiles else None
    det = OnnxDetector(args.model, threads=args.threads, conf=args.low_conf, windows=windows, tile_classes=args.tile_classes)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full, crop = {}, {}
    h = w = 0
    for t, f in read_frames(Path(args.video), args.every_s):
        h, w = f.shape[:2]
        full[str(t)] = [list(map(float, d)) for d in det.raw(f)]
        if args.crop_bottom:
            crop[str(t)] = [list(map(float, d)) for d in det.raw(f[: h - args.crop_bottom])]
    for ds in (*full.values(), *crop.values()):
        for d in ds:
            d[0] = int(d[0])
    data = {"video": str(args.video), "model": str(args.model), "every_s": args.every_s, "low_conf": args.low_conf,
            "frame_wh": [w, h], "crop_bottom": args.crop_bottom, "road_tiles": args.road_tiles, "tiles_bottom": args.tiles_bottom,
            "tile_classes": args.tile_classes, "full": full, "crop": crop}
    (out / "dets.json").write_text(json.dumps(data), encoding="utf-8")
    print(f"{len(full)} frames → {out / 'dets.json'}")
    return 0


def summarize(data: dict, thr: list[float]) -> dict:
    full, crop = data["full"], data["crop"]
    every = data["every_s"]
    dur = max(map(float, full)) + every
    s = {"frames": len(full), "duration_s": round(dur, 1), "class_conf": dict(zip(CLASS_NAMES, thr))}
    for key, fr in (("full", full), ("crop", crop)):
        if not fr:
            continue
        per = {}
        for c, name in enumerate(CLASS_NAMES):
            ts = [float(t) for t, ds in fr.items() if any(d[0] == c for d in above(ds, thr))]
            n = sum(1 for ds in fr.values() for d in above(ds, thr) if d[0] == c)
            per[name] = {"frames": len(ts), "boxes": n,
                         "segments": [list(x) for x in segments(ts, gap=2 * every) if x[2] >= 2]}
        s[key] = per
    if crop:
        h = data["frame_wh"][1]
        band = h - data["crop_bottom"]
        lost = [(t, d) for t in full for d in only_in(above(full[t], thr), above(crop[t], thr))]
        gained = [(t, d) for t in crop for d in only_in(above(crop[t], thr), above(full[t], thr))]
        s["crop_vs_full"] = {
            "only_full": dict(Counter(CLASS_NAMES[d[0]] for _, d in lost)),
            "only_full_touching_band": dict(Counter(CLASS_NAMES[d[0]] for _, d in lost if d[5] > band)),
            "only_crop": dict(Counter(CLASS_NAMES[d[0]] for _, d in gained)),
            "only_full_list": [[t, CLASS_NAMES[d[0]], round(d[1], 3), [round(v) for v in d[2:]]] for t, d in lost],
        }
    s["windows_20s"] = window_counts(full, thr, 20.0, 5.0, dur)
    return s


def cmd_summary(args) -> int:
    out = Path(args.out_dir)
    data = json.loads((out / "dets.json").read_text(encoding="utf-8"))
    s = summarize(data, _thr(Path(args.card)))
    (out / "summary_auto.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    for key in ("full", "crop"):
        if key in s:
            print(key, {k: (v["frames"], v["boxes"]) for k, v in s[key].items()})
    if "crop_vs_full" in s:
        print("only_full", s["crop_vs_full"]["only_full"], "band", s["crop_vs_full"]["only_full_touching_band"],
              "only_crop", s["crop_vs_full"]["only_crop"])
    return 0


def cmd_draw(args) -> int:
    out = Path(args.out_dir)
    data = json.loads((out / "dets.json").read_text(encoding="utf-8"))
    thr = _thr(Path(args.card))
    want = {str(round(t, 3)) for t in args.t}
    key = "crop" if args.use_crop else "full"
    (out / args.sub).mkdir(parents=True, exist_ok=True)
    for t, f in read_frames(Path(data["video"]), data["every_s"]):
        if str(t) in want:
            img = draw(f, above(data[key][str(t)], thr), data["crop_bottom"])
            p = out / args.sub / f"t{t:07.1f}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(p)
    return 0


def _events(frames: dict[str, list], thr: list[float], gt: list, video_id: str, tol_s: float, merge_gap_s: float,
            minutes: float) -> dict:
    """유형별 예측 이벤트 (video_events 규칙 그대로) + 정답이 있으면 맞음/헛것과 찾은 정답."""
    from daboyeo_contracts import DETECTOR_TO_DEFECT

    from daboyeo_ai.eval.video_events import group_events, in_window, match

    out = {}
    for c, name in enumerate(CLASS_NAMES):
        dtype = DETECTOR_TO_DEFECT[name].value
        hits = [(float(t), d[1]) for t, ds in frames.items() for d in above(ds, thr) if d[0] == c]
        preds = group_events(video_id, dtype, hits, merge_gap_s)
        row = {"frames": len({t for t, _ in hits}), "events": len(preds)}
        if gt is not None:
            g = [e for e in gt if e.defect_type == dtype]
            tc = match(gt, preds, dtype, tol_s, minutes)
            row |= {"n_gt": tc.n_gt, "found": tc.n_gt_found, "pred_ok": tc.n_pred_ok, "false_events": tc.n_pred - tc.n_pred_ok,
                    "precision": None if tc.precision is None else round(tc.precision, 3), "missed": tc.missed,
                    "false_event_times": [[round(e.t_start, 1), round(e.t_end, 1)] for e in preds
                                          if not any(in_window(t, x, tol_s) for t in e.hits for x in g)]}
        out[name] = row
    return out


def cmd_compare(args) -> int:
    """base 와 tiles dets.json 비교: 유형별 탐지 프레임·이벤트(정답이 있으면 맞음/헛것), tiles 에만 있는 결함 크롭 모음."""
    from daboyeo_ai.eval.demo_clip import crop_tile, sheet
    from daboyeo_ai.eval.recall_res import new_vs_base
    from daboyeo_ai.eval.video_events import load_gt

    thr = _thr(Path(args.card))
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    tiles = json.loads(Path(args.tiles).read_text(encoding="utf-8"))
    vid = Path(base["video"]).stem
    gt = load_gt(args.gt, vid) if args.gt else None
    dur = max(map(float, base["full"])) + base["every_s"]
    res = {"video": vid, "class_conf": dict(zip(CLASS_NAMES, thr)), "gt": args.gt, "tol_s": args.tol, "merge_gap_s": args.merge_gap}
    for key, data in (("base", base), ("tiles", tiles)):
        res[key] = _events(data["full"], thr, gt, vid, args.tol, args.merge_gap, dur / 60)
    want = {CLASS_NAMES.index(n) for n in args.classes}
    new = []
    for t, ds in tiles["full"].items():
        for d in new_vs_base(above(ds, thr), above(base["full"].get(t, []), thr)):
            if d[0] in want:
                new.append((float(t), d))
    res["tiles_only"] = [{"id": i, "t": t, "class": CLASS_NAMES[d[0]], "conf": round(d[1], 3), "box": [round(v) for v in d[2:]]}
                         for i, (t, d) in enumerate(sorted(new, key=lambda x: x[0]))]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_t: dict[float, list] = {}
    for r in res["tiles_only"]:
        by_t.setdefault(r["t"], []).append(r)
    crops = []
    for t, f in read_frames(Path(base["video"]), base["every_s"]):
        for r in by_t.get(t, []):
            d = (CLASS_NAMES.index(r["class"]), r["conf"], *r["box"])
            crops.append(crop_tile(f, d, f"{r['id']} t{t} {r['class'][:4]} {r['conf']:.2f}"))
    for k in range(0, len(crops), 24):
        cv2.imwrite(str(out / f"tiles_only_{k // 24}.jpg"), sheet(crops[k:k + 24], 6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for key in ("base", "tiles"):
        print(key, {n: {k: v for k, v in r.items() if k not in ("false_event_times", "missed")} for n, r in res[key].items()})
    print("tiles_only", len(res["tiles_only"]), "→", out)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--model", default="ai/models/detector_v3.onnx")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--every-s", type=float, default=0.5)
    r.add_argument("--low-conf", type=float, default=0.1)
    r.add_argument("--crop-bottom", type=int, default=0, help="아래에서 잘라 낼 px (0 이면 비교 안 함)")
    r.add_argument("--road-tiles", type=int, default=0, help="N 이면 전체 + 도로 N 조각 (OnnxDetector windows=road_tiles(N))")
    r.add_argument("--tiles-bottom", type=float, default=1.0, help="조각 아래 끝 (0.8 = 보닛·반사 띠 빼기)")
    r.add_argument("--tile-classes", nargs="+", help="조각에서 받을 클래스 (기본 전부)")
    r.add_argument("--threads", type=int, default=4)
    for name in ("summary", "draw"):
        q = sub.add_parser(name)
        q.add_argument("--out-dir", required=True)
        q.add_argument("--card", default="ai/models/detector_v3.json")
    d = sub.choices["draw"]
    d.add_argument("--t", type=float, nargs="+", required=True)
    d.add_argument("--sub", default="frames")
    d.add_argument("--use-crop", action="store_true")
    c = sub.add_parser("compare", help="base vs tiles dets.json (+ 정답 CSV 가 있으면 이벤트 단위)")
    c.add_argument("--base", required=True)
    c.add_argument("--tiles", required=True)
    c.add_argument("--gt", help="12장 형식 정답 CSV (strict 면 amb 행을 뺀 것)")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--card", default="ai/models/detector_v3.json")
    c.add_argument("--classes", nargs="+", default=["crack", "pothole"], help="tiles 에만 있는 탐지 크롭을 모을 클래스")
    c.add_argument("--tol", type=float, default=0.5)
    c.add_argument("--merge-gap", type=float, default=1.0)
    args = p.parse_args(argv)
    return {"run": cmd_run, "summary": cmd_summary, "draw": cmd_draw, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
