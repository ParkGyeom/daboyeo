"""차선 분할 마스크(`LaneSegmenter`)를 켠 차선 마모 판정을 시연 클립 gn_a·gn_b 에서 17장과 같은 방법으로 다시 잰다 (handoff 23장).

    # 1) 재기: 3 프레임마다 품질·노면 움직임·마스크 없는 측정·마스크 측정(설정 여러 개) 캐시 (레포 밖)
    uv run python -m daboyeo_ai.eval.lane_wear_seg measure --model ai/models/lane_seg_v1.onnx \\
        --clips data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4
    # 2) 마스크 설정에서 확정이 나온 2초 칸마다 오버레이 1장 (마스크 + 선분) → sheet
    uv run python -m daboyeo_ai.eval.lane_wear_seg sheets
    # 3) sheet 를 보고 data/lane_seg_eval/review.json ({"<clip>/<칸 시작 초>": "real"|"amb"|"fp"}) 을 쓴 뒤 집계
    uv run python -m daboyeo_ai.eval.lane_wear_seg summarize --stopped gn_b_3320-3620:30-96,152-172 --out experiments/lane_wear_seg.json

- 판정은 17장과 같다: 품질 통과 프레임만, `LaneWearTracker` 기준 1.0 · 5 중 3, 주기 3 프레임 (60fps 초당 20번).
- 마스크 칸 라벨은 **마스크 오버레이를 보고 새로** 붙인다 (17장 칸 라벨은 마스크 없는 선분을 보고 붙인 것이라 쓰지 않는다).
- `seg_m15`: 마스크를 15 프레임(0.25초)마다만 새로 구하고 그 사이 측정은 직전 마스크를 쓴다 (CPU 절약안).
영상·프레임·sheet 는 레포 밖(`--out-dir`)에만 둔다. 레포에는 집계 숫자만.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.lane_wear import CONFIRM_K, CONFIRM_N, LaneWearMeter

from .lane_wear_demo import LABELS, BIN_S, clip_stats, confirmed_times, grid, parse_stopped, read_frames, record, replay

OUT_DIR = Path("data/lane_seg_eval")
# 마스크 측정 설정: 이름 → (LaneWearMeter 인자, 마스크 갱신 주기 프레임)
SEG_CONFIGS = {
    "seg": ({}, 3),
    "seg_rd05": ({"mask_rel_delta": 0.05}, 3),   # 마스크 안에서는 5% 만 밝아도 후보 (젖은 노면)
    "seg_m15": ({}, 15),
}
# 비교 행: 이름 → replay variant (lane_wear_demo.replay). "seg" 키는 rec["xw"] 측정 이름
VARIANTS = {
    "base (17장 지금)": {},
    "base+motion": {"min_motion": 3.0},
    "seg": {"seg": "seg"},
    "seg+motion": {"seg": "seg", "min_motion": 3.0},
    "seg_rd05": {"seg": "seg_rd05"},
    "seg_rd05+motion": {"seg": "seg_rd05", "min_motion": 3.0},
    "seg_m15": {"seg": "seg_m15"},
    "seg_m15+motion": {"seg": "seg_m15", "min_motion": 3.0},
    "seg+motion+found_guard": {"seg": "seg", "min_motion": 3.0, "mode": "p25_x_len", "min_found": 0.3},
    # 17-6 의 점수 식 후보 (paint_len 빼기·줄이기) + 정지·찾은 비율 보호. 마스크 없이는 fp 84% 가 최선이었다
    "p25_only+motion+guard (마스크 없음)": {"min_motion": 3.0, "mode": "p25_only", "min_found": 0.3},
    "seg+p25_only+motion+guard": {"seg": "seg", "min_motion": 3.0, "mode": "p25_only", "min_found": 0.3},
    "seg+sqrt_rel_len+motion+guard": {"seg": "seg", "min_motion": 3.0, "mode": "p25_x_sqrt_rel_len", "min_found": 0.3},
    "seg_rd05+p25_only+motion+guard": {"seg": "seg_rd05", "min_motion": 3.0, "mode": "p25_only", "min_found": 0.3},
    "seg_m15+p25_only+motion+guard": {"seg": "seg_m15", "min_motion": 3.0, "mode": "p25_only", "min_found": 0.3},
}
REVIEW_VARIANTS = ("seg", "seg_rd05", "seg_m15", "seg+motion", "seg_rd05+motion", "seg_m15+motion",
                   "seg+p25_only+motion+guard", "seg+sqrt_rel_len+motion+guard", "seg_rd05+p25_only+motion+guard",
                   "seg_m15+p25_only+motion+guard")
THRESHOLDS = (0.4, 0.5, 0.6, 0.8, 1.0)
WINDOW_VARIANTS = ("base (17장 지금)", "seg", "seg+p25_only+motion+guard")


def measure_clip(video: Path, model: Path, every: int = 3, threads: int = 1) -> list[dict]:
    from edge.blackbox.quality import measure as quality  # 정현우 코드 — 읽기 전용으로 부른다

    from daboyeo_ai.inference.lane_seg import LaneSegmenter

    from .lane_wear_variants import MotionProbe

    seg = LaneSegmenter(model, threads=threads)
    base = LaneWearMeter()
    meters = {name: LaneWearMeter(**kw) for name, (kw, _) in SEG_CONFIGS.items()}
    probe = MotionProbe()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recs, i = [], 0
    masks: dict[int, np.ndarray] = {}  # 갱신 주기 → 마지막 마스크
    while cap.grab():
        if i % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            q = quality(frame)
            if q.ok:
                r, segs = base.measure_debug(frame)
                rec = record(i, i / fps, True, q.reason, r, segs)
                cur = None
                for _, m_every in SEG_CONFIGS.values():
                    if i % m_every == 0 or m_every not in masks:
                        if cur is None:
                            cur = seg.lane_mask(frame)
                        masks[m_every] = cur
                rec["mask_px"] = round(float(np.count_nonzero(masks[every])) / masks[every].size, 4)
                rec["xw"] = {}
                for name, (_, m_every) in SEG_CONFIGS.items():
                    rx, sx = meters[name].measure_debug(frame, lane_mask=masks[m_every])
                    rec["xw"][name] = {k: v for k, v in record(i, 0, True, "", rx, sx).items()
                                       if k not in ("i", "t", "q_ok", "q_reason")}
            else:
                rec = record(i, i / fps, False, q.reason, None, [])
            rec["motion"] = probe.update(frame, i / fps)
            recs.append(rec)
        i += 1
    cap.release()
    return recs


def bin_key(clip: str, t: float, bin_s: float = BIN_S) -> str:
    return f"{clip}/{int(t // bin_s * bin_s)}"


def tally(ts: list[float], clip: str, review: dict[str, str]) -> dict:
    c = {lab: 0 for lab in (*LABELS, "unlabeled")}
    for t in ts:
        c[review.get(bin_key(clip, t), "unlabeled")] += 1
    done = sum(c[x] for x in LABELS)
    return c | {"fp_rate": round(c["fp"] / done, 3) if done else None}


def seg_view(rec: dict, name: str) -> dict:
    return rec | rec["xw"][name] if rec.get("xw") else rec


def overlay(frame: np.ndarray, rec: dict, j: dict | None, mask: np.ndarray | None, label: str, width: int = 640) -> np.ndarray:
    img = frame.copy()
    if mask is not None:
        on = mask > 0
        img[on] = (0.5 * img[on] + np.array((0, 110, 0))).astype(np.uint8)
    worn = bool(j and j.get("is_worn"))
    for x1, y1, x2, y2 in rec.get("segs", []):
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255) if worn else (255, 255, 0), 4)
    h = img.shape[0]
    s = img.shape[1] / width
    img = cv2.resize(img, (width, round(h / s)), interpolation=cv2.INTER_AREA)
    sc = j.get("score") if j else None
    txt = f"{label} t={rec['t']:.1f}s score={'-' if sc is None else f'{sc:.2f}'}"
    cv2.rectangle(img, (0, 0), (width, 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_sheets(clip_dir: Path, video: Path, recs: list[dict], model: Path, per_sheet: int = 12) -> list[str]:
    """REVIEW_VARIANTS × THRESHOLDS 의 확정이 든 2초 칸마다, 그 칸을 확정한 설정에서 점수가 가장 낮은 프레임 1장."""
    from daboyeo_ai.inference.lane_seg import LaneSegmenter

    clip = clip_dir.name
    by_i = {r["i"]: r for r in recs}
    best: dict[str, tuple[float, int, str, dict]] = {}
    for vn in REVIEW_VARIANTS:
        v = VARIANTS[vn]
        for thr in THRESHOLDS:
            judged = replay(recs, thr, CONFIRM_K, CONFIRM_N, v)
            keys = {bin_key(clip, t) for t in confirmed_times(judged)}
            for j in judged:
                k = bin_key(clip, j["t"])
                if k in keys and j["score"] is not None and (k not in best or j["score"] < best[k][0]):
                    best[k] = (j["score"], j["i"], v["seg"], j)
    keys = sorted(best, key=lambda k: int(k.split("/")[1]))
    frames = read_frames(video, [best[k][1] for k in keys])
    seg = LaneSegmenter(model)
    out = clip_dir / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    tiles = []
    for k in keys:
        _, i, name, j = best[k]
        f = frames[i]
        tiles.append(overlay(f, seg_view(by_i[i], name), j, seg.lane_mask(f), f"[{k.split('/')[1]}s {name}]"))
    for s0 in range(0, len(tiles), per_sheet):
        a, b = keys[s0].split("/")[1], keys[min(s0 + per_sheet, len(keys)) - 1].split("/")[1]
        cv2.imwrite(str(out / f"bins_{int(a):03d}-{int(b):03d}.jpg"), grid(tiles[s0:s0 + per_sheet]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    return keys


def summarize_clip(clip: str, recs: list[dict], review: dict[str, str], stops: list[tuple[float, float]]) -> dict:
    stats = {"base": clip_stats(recs, replay(recs))}
    for name in SEG_CONFIGS:
        sv = [seg_view(r, name) for r in recs]
        stats[name] = clip_stats(sv, replay(recs, variant={"seg": name}))
    rows = []
    for vn, v in VARIANTS.items():
        for thr in THRESHOLDS:
            ts = confirmed_times(replay(recs, thr, CONFIRM_K, CONFIRM_N, v))
            in_stop = [t for t in ts if any(a <= t <= b for a, b in stops)]
            rows.append({"variant": vn, "threshold": thr, "confirmed": len(ts), **tally(ts, clip, review),
                         "confirmed_while_stopped": len(in_stop),
                         "real_135_140": sum(135 <= t <= 141 and review.get(bin_key(clip, t)) == "real" for t in ts),
                         "times": [round(t, 2) for t in ts]})
    mp = [r["mask_px"] for r in recs if "mask_px" in r]
    return {"stats": stats, "mask_px_mean": round(float(np.mean(mp)), 4) if mp else None, "rows": rows}


def window_rows(clip: str, recs: list[dict], review: dict[str, str], start: float, end: float,
                variants: tuple[str, ...] = WINDOW_VARIANTS, thresholds=THRESHOLDS) -> list[dict]:
    """클립을 [start, end] 초로 잘라 돌린 것처럼 (tracker 가 start 에서 새로 시작) 확정·눈 분류를 센다.
    시연에서 한 구간만 잘라 보여 줄 때 그 구간 안 헛것을 미리 보려는 것 (handoff 23-6)."""
    w = [r for r in recs if start <= r["t"] <= end]
    rows = []
    for vn in variants:
        for thr in thresholds:
            ts = confirmed_times(replay(w, thr, CONFIRM_K, CONFIRM_N, VARIANTS[vn]))
            rows.append({"window": [start, end], "variant": vn, "threshold": thr, "confirmed": len(ts),
                         **tally(ts, clip, review), "times": [round(t, 2) for t in ts]})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--clips", nargs="+", type=Path, required=True)
    m.add_argument("--every", type=int, default=3)
    m.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("sheets")
    u = sub.add_parser("summarize")
    u.add_argument("--out", type=Path, default=Path("experiments/lane_wear_seg.json"))
    u.add_argument("--stopped", nargs="*", default=[], metavar="CLIP:A-B")
    wd = sub.add_parser("window", help="한 구간만 잘라 돌린 것처럼 (시연 구간 고르기, 23-6)")
    wd.add_argument("--clip", required=True)
    wd.add_argument("--windows", nargs="+", required=True, metavar="A-B")
    wd.add_argument("--out", type=Path, default=None)
    for p in (m, s):
        p.add_argument("--model", type=Path, default=Path("ai/models/lane_seg_v1.onnx"))
    for p in (m, s, u, wd):
        p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)

    if a.cmd == "window":
        recs = json.loads((a.out_dir / a.clip / "frames.json").read_text())["frames"]
        rp = a.out_dir / "review.json"
        review = json.loads(rp.read_text()) if rp.exists() else {}
        rows = []
        for spec in a.windows:
            s0, e0 = (float(x) for x in spec.split("-"))
            rows += window_rows(a.clip, recs, review, s0, e0)
        for r in rows:
            print(f"{r['window'][0]:g}-{r['window'][1]:g} {r['variant']:28s} thr={r['threshold']} n={r['confirmed']} "
                  f"real={r['real']} amb={r['amb']} fp={r['fp']} unl={r['unlabeled']} {r['times']}")
        if a.out:
            a.out.write_text(json.dumps({"clip": a.clip, "rows": rows}, ensure_ascii=False, indent=1) + "\n")
        return

    if a.cmd == "measure":
        for v in a.clips:
            d = a.out_dir / v.stem
            d.mkdir(parents=True, exist_ok=True)
            recs = measure_clip(v, a.model, a.every, a.threads)
            (d / "frames.json").write_text(json.dumps({"video": str(v), "every": a.every, "model": str(a.model),
                                                       "frames": recs}))
            print(v.name, len(recs), "frames")
        return

    clips = sorted(p.parent for p in a.out_dir.glob("*/frames.json"))
    if a.cmd == "sheets":
        for d in clips:
            c = json.loads((d / "frames.json").read_text())
            keys = make_sheets(d, Path(c["video"]), c["frames"], a.model)
            print(d.name, len(keys), "bins:", " ".join(k.split("/")[1] for k in keys))
        return

    review_p = a.out_dir / "review.json"
    review = json.loads(review_p.read_text()) if review_p.exists() else {}
    stops = parse_stopped(a.stopped)
    out = {"setup": {"every_frames": 3, "vote": f"{CONFIRM_K}/{CONFIRM_N}", "bin_s": BIN_S, "seg_configs":
                     {k: {"meter": kw, "mask_every_frames": me} for k, (kw, me) in SEG_CONFIGS.items()},
                     "review": "Claude 눈 분류 (마스크 설정 확정이 든 2초 칸, 마스크·선분 오버레이 1장, real/amb/fp)"},
           "clips": {}}
    for d in clips:
        c = json.loads((d / "frames.json").read_text())
        out["setup"]["model"] = c.get("model")
        out["clips"][d.name] = summarize_clip(d.name, c["frames"], review, stops.get(d.name, []))
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for clip, v in out["clips"].items():
        print(clip, {k: s["lane_found_of_pass"] for k, s in v["stats"].items()})
        for r in v["rows"]:
            print(f"  {r['variant']:28s} thr={r['threshold']} n={r['confirmed']:3d} real={r['real']} amb={r['amb']} "
                  f"fp={r['fp']} unl={r['unlabeled']} fp%={r['fp_rate']} stopped={r['confirmed_while_stopped']}")


if __name__ == "__main__":
    main()
