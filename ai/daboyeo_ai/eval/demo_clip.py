"""시연 클립(gn_a·gn_b)에서 탐지 v2 vs v3 — 같은 프레임, 분당 탐지·conf 분포·한쪽에만 있는 탐지·자막 띠·품질 게이트 (Mac, ONNX).

    # 1) 돌리기: 프레임별 탐지 캐시 + 한쪽에만 있는 탐지 크롭 모음 (data/demo_clip_eval/, 레포 밖)
    uv run python -m daboyeo_ai.eval.demo_clip run \\
        --clips /Users/parkgyeom/daboyeo/data/videos/gn_a_2050-2510.mp4 /Users/parkgyeom/daboyeo/data/videos/gn_b_3320-3620.mp4
    # 2) 크롭 모음을 눈으로 보고 data/demo_clip_eval/review.json ({"크롭 id": "real"|"amb"|"fp"}) 을 쓴 뒤 집계
    uv run python -m daboyeo_ai.eval.demo_clip summarize

- 같은 프레임: `--every-s`(기본 0.5초) 간격. 블랙박스 `--detect-every 30`(60fps) 과 같은 간격.
- 기준: v2 = `experiments/conf_thresholds.json`, v3 = `experiments/conf_thresholds_v3.json` (각자 카드 기준).
- 한쪽에만 있는 탐지 = 같은 프레임에서 상대 모델의 같은 클래스(상대 기준 이상) 와 IoU < 0.3 (`recall_res.new_vs_base`).
- 자막 띠: 박스 중심이 프레임 아래 `BANDS` 비율 안에 있는 탐지 수. 품질: `edge/blackbox/quality.measure` 를 읽기 전용으로 부른다.
- (선택) `--tiles`: v2 + `road_tiles(2)` 로 포트홀이 더 잡히는지 (v2 base 에 없던 탐지 크롭).
영상·프레임·크롭은 YouTube 표준 라이선스라 레포 밖(`--out-dir`)에만 둔다. 레포에는 집계 숫자만 (`--experiments`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .recall_res import new_vs_base

BANDS = (0.06, 0.08, 0.10, 0.12, 0.15)  # 프레임 아래 띠 높이 (비율)
LABELS = ("real", "amb", "fp")
TOP_CAPTION = (0.30, 0.0, 0.70, 0.20)  # gn_a 교차로 이름 자막 자리 (가운데 위). 같은 자리의 건물·안내 표지판도 들어간다
# 채택 기준 (제안): v3 로 바꾸려면 균열 real 증가 >= 균열 clear FP 증가 이고, 나머지 3종 real 순손실 <= 0.03/프레임
MAX_OTHER_REAL_LOSS_PER_FRAME = 0.03
OTHER = ("pothole", "guardrail", "sign")


def above(dets: list, thr: list[float]) -> list:
    return [d for d in dets if d[1] >= thr[d[0]]]


def per_minute(frames: dict[str, list], thr: list[float], minutes: float) -> dict[str, float]:
    c = Counter(CLASS_NAMES[d[0]] for ds in frames.values() for d in above(ds, thr))
    return {k: round(c[k] / minutes, 2) for k in CLASS_NAMES}


def conf_quantiles(frames: dict[str, list], thr: list[float]) -> dict[str, dict]:
    """기준 이상 탐지의 클래스별 conf 분포 (개수, p10/p50/p90, 0.5 이상 비율)."""
    out = {}
    for c, name in enumerate(CLASS_NAMES):
        s = np.array([d[1] for ds in frames.values() for d in above(ds, thr) if d[0] == c])
        out[name] = {"n": int(s.size)} | ({} if not s.size else {
            "p10": round(float(np.quantile(s, 0.1)), 3), "p50": round(float(np.median(s)), 3),
            "p90": round(float(np.quantile(s, 0.9)), 3), "ge_0.5": round(float((s >= 0.5).mean()), 3)})
    return out


def band_counts(frames: dict[str, list], thr: list[float], frame_h: int, bands=BANDS) -> dict[str, dict]:
    """박스 중심이 프레임 아래 band 비율 안에 있는 기준 이상 탐지 수 (클래스별)."""
    out = {}
    for b in bands:
        c = Counter(CLASS_NAMES[d[0]] for ds in frames.values() for d in above(ds, thr)
                    if (d[3] + d[5]) / 2 >= (1 - b) * frame_h)
        out[f"{round(b * 100)}%"] = {k: c[k] for k in CLASS_NAMES}
    return out


def region_counts(frames: dict[str, list], thr: list[float], frame_w: int, frame_h: int,
                  region: tuple[float, float, float, float]) -> dict[str, int]:
    """박스 중심이 region (x1, y1, x2, y2, 0~1 비율) 안에 있는 기준 이상 탐지 수 (클래스별)."""
    x1, y1, x2, y2 = region
    c = Counter(CLASS_NAMES[d[0]] for ds in frames.values() for d in above(ds, thr)
                if x1 * frame_w <= (d[2] + d[4]) / 2 <= x2 * frame_w and y1 * frame_h <= (d[3] + d[5]) / 2 <= y2 * frame_h)
    return {k: c[k] for k in CLASS_NAMES}


def one_sided(a: dict[str, list], b: dict[str, list], thr_a: list[float], thr_b: list[float]) -> dict[str, list]:
    """프레임마다 a 에만 있는 탐지 (a 기준 이상, b 의 기준 이상 탐지와 IoU < 0.3)."""
    return {t: new_vs_base(above(a[t], thr_a), above(b[t], thr_b)) for t in a}


def review_counts(crops: dict[str, dict], review: dict[str, str]) -> dict:
    """크롭 id → 라벨을 (clip, kind, class, label) 개수로. 라벨이 없으면 'unreviewed'."""
    out: dict = {}
    for cid, m in crops.items():
        lab = review.get(cid, "unreviewed")
        out.setdefault(m["clip"], {}).setdefault(m["kind"], {}).setdefault(m["cls"], Counter())[lab] += 1
    return {c: {k: {cls: dict(cn) for cls, cn in v.items()} for k, v in kk.items()} for c, kk in out.items()}


def review_at(crops: dict[str, dict], review: dict[str, str], kind: str, cls: str, thr: float) -> dict[str, int]:
    """kind·cls 크롭 중 conf >= thr 인 것의 라벨 개수 (기준을 올리면 real/fp 가 어떻게 남는지)."""
    return dict(Counter(review.get(k, "unreviewed") for k, m in crops.items()
                        if m["kind"] == kind and m["cls"] == cls and m["conf"] >= thr))


def decide(counts: dict, frames: int) -> dict:
    """채택 기준 (모듈 상단). counts = review_counts 결과 (모든 클립 합산), frames = 전체 분석 프레임 수."""
    def tot(kind: str, cls: str, lab: str) -> int:
        return sum(c.get(kind, {}).get(cls, {}).get(lab, 0) for c in counts.values())

    crack_real_gain = tot("v3_only", "crack", "real") - tot("v2_only", "crack", "real")
    crack_fp_gain = tot("v3_only", "crack", "fp") - tot("v2_only", "crack", "fp")
    other_loss = sum(tot("v2_only", c, "real") - tot("v3_only", c, "real") for c in OTHER) / max(1, frames)
    checks = {
        "crack_real_gain_ge_fp_gain": {"real_gain": crack_real_gain, "fp_gain": crack_fp_gain,
                                       "ok": crack_real_gain >= crack_fp_gain},
        "crack_real_gain_positive": {"value": crack_real_gain, "ok": crack_real_gain > 0},
        "other_real_net_loss_per_frame": {"value": round(other_loss, 3), "need": f"<= {MAX_OTHER_REAL_LOSS_PER_FRAME}",
                                          "ok": other_loss <= MAX_OTHER_REAL_LOSS_PER_FRAME},
    }
    return {"checks": checks, "adopt_v3": all(c["ok"] for c in checks.values())}


def read_frames(video: Path, every_s: float):
    """every_s 간격 프레임 (t_s, frame). 순서대로 grab 하고 필요한 프레임만 retrieve."""
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


def crop_tile(img: np.ndarray, d: tuple, label: str, h: int = 220) -> np.ndarray:
    """박스 주변(박스 2배, 최소 200px)을 잘라 높이 h 로, 박스와 라벨을 그린다."""
    H, W = img.shape[:2]
    _, _, x1, y1, x2, y2 = d
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(100, x2 - x1, y2 - y1)
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, e = int(min(W, cx + half)), int(min(H, cy + half))
    crop = img[b:e, a:c].copy()
    cv2.rectangle(crop, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (0, 255, 0), 2)
    s = h / crop.shape[0]
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), h))
    bar = np.zeros((26, crop.shape[1], 3), np.uint8)
    cv2.putText(bar, label, (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([bar, crop])


def sheet(tiles: list[np.ndarray], cols: int = 6) -> np.ndarray:
    w = max(t.shape[1] for t in tiles)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(40, 40, 40))
             for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 40))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def _thr(path: Path) -> list[float]:
    from daboyeo_ai.inference import load_class_conf

    cc = load_class_conf(path)
    return [cc[n] for n in CLASS_NAMES]


def cmd_run(args) -> int:
    from daboyeo_ai.inference import OnnxDetector, road_tiles
    from edge.blackbox.quality import measure  # 읽기 전용 (정현우 담당 코드, 고치지 않는다)

    thr2, thr3 = _thr(args.v2_conf), _thr(args.v3_conf)
    dets = {"v2": OnnxDetector(args.v2, threads=args.threads, conf=args.low_conf),
            "v3": OnnxDetector(args.v3, threads=args.threads, conf=args.low_conf)}
    if args.tiles:
        dets["v2_t2"] = OnnxDetector(args.v2, threads=args.threads, conf=args.low_conf, windows=road_tiles(2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops: dict[str, dict] = {}
    for clip in args.clips:
        name = "_".join(clip.stem.split("_")[:2])  # gn_a_2050-2510 → gn_a
        cdir = args.out_dir / name
        cdir.mkdir(exist_ok=True)
        raw: dict = {k: {} for k in dets}
        quality: dict = {}
        tiles: dict[str, list] = {}
        h = 0
        for t, f in read_frames(clip, args.every_s):
            key = f"{t:.3f}"
            h = f.shape[0]
            q = measure(f)
            quality[key] = {"ok": q.ok, "reason": q.reason}
            for k, d in dets.items():
                raw[k][key] = [tuple(round(float(v), 4) if i else int(v) for i, v in enumerate(x)) for x in d.raw(f)]
            pairs = [("v3_only", one_sided({key: raw["v3"][key]}, {key: raw["v2"][key]}, thr3, thr2)[key]),
                     ("v2_only", one_sided({key: raw["v2"][key]}, {key: raw["v3"][key]}, thr2, thr3)[key])]
            if args.tiles:
                pairs.append(("t2_gain", one_sided({key: raw["v2_t2"][key]}, {key: raw["v2"][key]}, thr2, thr2)[key]))
            for kind, ds in pairs:
                for d in ds:
                    cls = CLASS_NAMES[d[0]]
                    if kind == "t2_gain" and cls != "pothole":
                        continue
                    cid = f"{name}-{kind}-{cls}-{sum(1 for m in crops.values() if m['clip'] == name and m['kind'] == kind and m['cls'] == cls):03d}"
                    crops[cid] = {"clip": name, "kind": kind, "cls": cls, "t": t, "conf": d[1], "box": d[2:]}
                    tiles.setdefault(f"{kind}-{cls}", []).append(
                        crop_tile(f, d, f"{cid.split('-')[-1]} t{t:.1f} {d[1]:.2f}"))
        (cdir / "raw.json").write_text(json.dumps({"frame_h": h, "every_s": args.every_s, "quality": quality,
                                                   "raw": raw}), encoding="utf-8")
        for k, ts in tiles.items():
            for i in range(0, len(ts), args.per_sheet):
                cv2.imwrite(str(cdir / f"{k}_{i // args.per_sheet}.jpg"), sheet(ts[i:i + args.per_sheet]),
                            [cv2.IMWRITE_JPEG_QUALITY, 82])
        print(f"{name}: {len(quality)} frames, crops " + ", ".join(f"{k}={len(v)}" for k, v in sorted(tiles.items())))
    (args.out_dir / "crops.json").write_text(json.dumps(crops, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"→ {args.out_dir} (review.json 을 쓰고 summarize)")
    return 0


def cmd_summarize(args) -> int:
    thr2, thr3 = _thr(args.v2_conf), _thr(args.v3_conf)
    card3 = json.loads(args.v3_card.read_text(encoding="utf-8")) if args.v3_card.exists() else {}
    thr3_p08 = [card3["class_conf_p08"][n] for n in CLASS_NAMES] if "class_conf_p08" in card3 else None
    crops = json.loads((args.out_dir / "crops.json").read_text(encoding="utf-8"))
    rp = args.out_dir / "review.json"
    review = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    out: dict = {"every_s": None, "v2_conf": dict(zip(CLASS_NAMES, thr2)), "v3_conf": dict(zip(CLASS_NAMES, thr3)),
                 "clips": {}}
    total_frames = 0
    for cdir in sorted(p for p in args.out_dir.iterdir() if (p / "raw.json").exists()):
        r = json.loads((cdir / "raw.json").read_text(encoding="utf-8"))
        raw, q, h = r["raw"], r["quality"], r["frame_h"]
        out["every_s"] = r["every_s"]
        n = len(q)
        total_frames += n
        minutes = n * r["every_s"] / 60
        ok_keys = [k for k in q if q[k]["ok"]]
        c: dict = {"frames": n, "minutes": round(minutes, 2),
                   "quality": {"ok_ratio": round(len(ok_keys) / max(1, n), 3),
                               "skip_reasons": dict(Counter(v["reason"] for v in q.values() if not v["ok"]))}}
        for m, thr in (("v2", thr2), ("v3", thr3)):
            c[m] = {"per_min": per_minute(raw[m], thr, minutes),
                    "per_min_quality_ok": per_minute({k: raw[m][k] for k in ok_keys}, thr,
                                                     max(1e-9, len(ok_keys) * r["every_s"] / 60)),
                    "conf": conf_quantiles(raw[m], thr), "bottom_band": band_counts(raw[m], thr, h),
                    "top_center": region_counts(raw[m], thr, round(h * 16 / 9), h, TOP_CAPTION)}
        if thr3_p08:
            c["v3_p08"] = {"conf": dict(zip(CLASS_NAMES, thr3_p08)), "per_min": per_minute(raw["v3"], thr3_p08, minutes),
                           "v3_only_per_frame": {}}
            os_ = one_sided(raw["v3"], raw["v2"], thr3_p08, thr2)
            cnt = Counter(CLASS_NAMES[d[0]] for ds in os_.values() for d in ds)
            c["v3_p08"]["v3_only_per_frame"] = {k: round(cnt[k] / n, 3) for k in CLASS_NAMES}
        for kind, a, b, ta, tb in (("v3_only", "v3", "v2", thr3, thr2), ("v2_only", "v2", "v3", thr2, thr3)):
            os_ = one_sided(raw[a], raw[b], ta, tb)
            cnt = Counter(CLASS_NAMES[d[0]] for ds in os_.values() for d in ds)
            c[f"{kind}_per_frame"] = {k: round(cnt[k] / n, 3) for k in CLASS_NAMES}
        if "v2_t2" in raw:
            c["v2_t2"] = {"per_min": per_minute(raw["v2_t2"], thr2, minutes)}
            g = one_sided(raw["v2_t2"], raw["v2"], thr2, thr2)
            c["v2_t2"]["pothole_gain_per_frame"] = round(sum(d[0] == 0 for ds in g.values() for d in ds) / n, 3)
        out["clips"][cdir.name] = c
    counts = review_counts(crops, review)
    for name, cc in counts.items():
        out["clips"].setdefault(name, {})["review"] = cc
    ci = CLASS_NAMES.index("crack")
    out["v3_only_crack_review_at"] = {f"{t:.2f}": review_at(crops, review, "v3_only", "crack", t)
                                      for t in sorted({thr3[ci]} | ({thr3_p08[ci]} if thr3_p08 else set()))}
    out["unreviewed"] = sum(1 for k in crops if k not in review)
    out["decision"] = decide(counts, total_frames)
    args.experiments.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["decision"], ensure_ascii=False, indent=1), f"\nunreviewed {out['unreviewed']}\n→ {args.experiments}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize"):
        p = sub.add_parser(name)
        p.add_argument("--v2-conf", type=Path, default=Path("experiments/conf_thresholds.json"))
        p.add_argument("--v3-conf", type=Path, default=Path("experiments/conf_thresholds_v3.json"))
        p.add_argument("--out-dir", type=Path, default=Path("data/demo_clip_eval"))
        if name == "run":
            p.add_argument("--clips", type=Path, nargs="+", required=True)
            p.add_argument("--v2", type=Path, default=Path("ai/models/detector_v2.onnx"))
            p.add_argument("--v3", type=Path, default=Path("ai/models/detector_v3.onnx"))
            p.add_argument("--every-s", type=float, default=0.5)
            p.add_argument("--low-conf", type=float, default=0.05)
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--per-sheet", type=int, default=36)
            p.add_argument("--tiles", action="store_true", help="v2 + road_tiles(2) 도 돌린다 (포트홀 확인)")
        else:
            p.add_argument("--experiments", type=Path, default=Path("experiments/demo_clip_compare.json"))
            p.add_argument("--v3-card", type=Path, default=Path("ai/models/detector_v3.json"),
                           help="class_conf_p08 (정밀도 >= 0.8 기준) 도 같이 집계")
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
