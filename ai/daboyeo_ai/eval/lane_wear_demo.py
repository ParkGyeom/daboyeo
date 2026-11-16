"""시연 클립(gn_a·gn_b)에서 차선 마모 판정(`LaneWearMeter` + `LaneWearTracker`)을 블랙박스와 같은 주기로 돌려 본다.

    # 1) 재기: 프레임별 측정 캐시 (data/lane_wear_demo/<clip>/frames.json, 레포 밖)
    uv run python -m daboyeo_ai.eval.lane_wear_demo measure --clips data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4
    # 2) 확정 칸 모음: 주기·기준·투표를 바꿔 나온 확정이 든 2초 칸마다 오버레이 1장 → contact sheet
    uv run python -m daboyeo_ai.eval.lane_wear_demo sheets
    # 3) sheet 를 눈으로 보고 data/lane_wear_demo/review.json ({"<clip>/b<칸 id 3자리>": "real"|"amb"|"fp"}) 을 쓴 뒤 집계
    uv run python -m daboyeo_ai.eval.lane_wear_demo summarize --out experiments/lane_wear_demo.json

- 주기: 정현우 블랙박스(`edge/blackbox/run.py`)는 `--lane-every 3` **프레임**마다 (탐지 주기와 따로) 재고,
  품질 판별(`edge/blackbox/quality.measure`)에 떨어진 프레임은 tracker 에 넣지 않는다. 같게 한다 (`--every 3`).
  60fps 클립이면 초당 20번, 투표 창 5 프레임 = 0.25초.
- 측정은 한 번만 하고(캐시), 기준·투표는 캐시를 `LaneWearTracker` 로 다시 돌려 바꿔 본다 (`replay`).
영상·프레임·sheet 는 YouTube 표준 라이선스라 레포 밖(`--out-dir`)에만 둔다. 레포에는 집계 숫자만.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.lane_wear import (
    CONFIRM_K,
    CONFIRM_N,
    PROVISIONAL_WEAR_SCORE,
    LaneWearMeter,
    LaneWearResult,
    LaneWearTracker,
)

OUT_DIR = Path("data/lane_wear_demo")
LABELS = ("real", "amb", "fp")
THRESHOLDS = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
VOTES = ((2, 5), (3, 5), (4, 5), (5, 5))
CLUSTER_GAP_S = 1.5  # 확정 시각이 이만큼 안에 있으면 같은 구간으로 묶는다 (시연 구간 후보)
BIN_S = 2.0  # 눈 분류 단위: 확정이 든 2초 칸 (60 km/h 면 약 33 m). 칸 안 가장 낮은 점수 프레임 1장을 본다
EVERY_MULT = (1, 2, 5, 10)  # 캐시(3프레임마다)를 다시 솎아 주기 3·6·15·30 프레임 (60fps: 0.05·0.1·0.25·0.5초)


def record(i: int, t: float, q_ok: bool, q_reason: str, r: LaneWearResult | None, segs: list) -> dict:
    """프레임 하나의 측정값 (캐시 한 줄). 품질 탈락이면 r=None."""
    d = {"i": i, "t": round(t, 3), "q_ok": q_ok, "q_reason": q_reason}
    if r is not None:
        d |= {"lane_found": r.lane_found, "contrast": r.contrast, "contrast_p25": r.contrast_p25,
              "paint_len": r.paint_len, "n_segments": r.n_segments, "lane_pixel_ratio": r.lane_pixel_ratio,
              "segs": [list(map(int, s)) for s in segs]}
    return d


def to_result(rec: dict) -> LaneWearResult:
    return LaneWearResult(lane_found=bool(rec.get("lane_found")), contrast=rec.get("contrast"), is_worn=False,
                          confidence=0.0, lane_pixel_ratio=rec.get("lane_pixel_ratio", 0.0),
                          n_segments=rec.get("n_segments", 0), contrast_p25=rec.get("contrast_p25"),
                          paint_len=rec.get("paint_len", 0.0))


def replay(recs: list[dict], threshold: float = PROVISIONAL_WEAR_SCORE, k: int = CONFIRM_K,
           n: int = CONFIRM_N, variant: dict | None = None) -> list[dict]:
    """블랙박스와 같게: 품질 통과 프레임만 tracker 에 넣는다. 프레임마다 {t, score, is_worn, confirmed}.

    variant (§17-6 개선 후보, `lane_wear_variants`): {"seg": 횡단보도 거른 측정 이름 (rec["xw"] 키),
    "min_motion": 노면 차이가 이보다 작으면 (정지) 넣지 않음, "mode": 점수 식, "min_found": 차선 찾은 비율 보호}."""
    v = variant or {}
    if v.get("mode") or v.get("min_found"):
        from .lane_wear_variants import VariantTracker
        tr = VariantTracker(mode=v.get("mode") or "p25_x_len", min_found=v.get("min_found", 0.0),
                            threshold=threshold, confirm_k=k, confirm_n=n)
    else:
        tr = LaneWearTracker(threshold=threshold, confirm_k=k, confirm_n=n)
    out = []
    for rec in recs:
        if not rec["q_ok"]:
            continue
        if v.get("min_motion") is not None and rec.get("motion") is not None and rec["motion"] < v["min_motion"]:
            continue
        if v.get("seg"):
            rec = rec | rec["xw"][v["seg"]]
        j = tr.update(to_result(rec), t=rec["t"])
        out.append({"i": rec["i"], "t": rec["t"], "lane_found": j.lane_found, "score": j.score,
                    "is_worn": j.is_worn, "confirmed": j.confirmed, "votes": j.votes})
    return out


def confirmed_times(judged: list[dict]) -> list[float]:
    return [j["t"] for j in judged if j["confirmed"]]


def worn_runs(judged: list[dict], k: int = CONFIRM_K, n: int = CONFIRM_N) -> list[tuple[float, float]]:
    """투표 창 안 마모가 k 개 이상인 판정 프레임이 이어진 구간 (시작, 끝 초). 확정 1건이 덮는 시간 길이를 보려고."""
    runs: list[list[float]] = []
    prev_in = False
    for j in judged:
        if j["score"] is None:
            continue
        inside = j["votes"] >= k
        if inside and prev_in:
            runs[-1][1] = j["t"]
        elif inside:
            runs.append([j["t"], j["t"]])
        prev_in = inside
    return [(a, b) for a, b in runs]


def cluster(times: list[float], gap: float = CLUSTER_GAP_S) -> list[tuple[float, float]]:
    """정렬한 시각을 gap 안에서 이어 붙인 구간 [(시작, 끝)]."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def label_of(t: float, clusters: list[tuple[float, float]], labels: dict[int, str]) -> str | None:
    for ci, (a, b) in enumerate(clusters):
        if a - 1e-6 <= t <= b + 1e-6:
            return labels.get(ci)
    return None


def clip_stats(recs: list[dict], judged: list[dict]) -> dict:
    """차선 찾은 비율, 점수 분포, 한 프레임 마모 비율, 확정 수 (기본 기준·투표)."""
    n_all = len(recs)
    q_ok = [r for r in recs if r["q_ok"]]
    found = [r for r in q_ok if r.get("lane_found")]
    scores = np.array([j["score"] for j in judged if j["score"] is not None])
    p25 = np.array([r["contrast_p25"] for r in found])
    pl = np.array([r["paint_len"] for r in found])
    q = (lambda a: {f"p{p}": round(float(np.percentile(a, p)), 3) for p in (5, 10, 25, 50, 75, 90, 95)}
         if a.size else {})
    return {
        "frames": n_all, "quality_pass": round(len(q_ok) / max(n_all, 1), 3),
        "lane_found_of_pass": round(len(found) / max(len(q_ok), 1), 3),
        "judged": int(scores.size),
        "score": q(scores), "contrast_p25": q(p25), "paint_len": q(pl),
        "frame_worn_rate": round(float((scores < PROVISIONAL_WEAR_SCORE).mean()), 3) if scores.size else None,
        "near_threshold_rate": round(float((np.abs(scores - PROVISIONAL_WEAR_SCORE) < 0.1).mean()), 3)
        if scores.size else None,
    }


def sweep(recs: list[dict], clusters: list[tuple[float, float]], labels: dict[int, str], duration_s: float,
          thresholds=THRESHOLDS, votes=VOTES, every_mult=(1,), every: int = 3) -> list[dict]:
    """주기 × 기준 × 투표별 확정 수, 분당, 눈 분류(real/amb/fp/미분류) 수, 헛것 비율 (fp ÷ 분류된 것)."""
    rows = []
    for m in every_mult:
        for thr in thresholds:
            for k, n in votes:
                ts = confirmed_times(replay(recs[::m], thr, k, n))
                c = {lab: 0 for lab in (*LABELS, "unlabeled")}
                for t in ts:
                    c[label_of(t, clusters, labels) or "unlabeled"] += 1
                done = sum(c[x] for x in LABELS)
                rows.append({"every": every * m, "threshold": thr, "vote": f"{k}/{n}", "confirmed": len(ts),
                             "per_min": round(len(ts) / (duration_s / 60), 2), **c,
                             "fp_rate": round(c["fp"] / done, 3) if done else None})
    return rows


# ---------- 영상 (Mac) ----------

XW_CONFIGS = {"xw_k4_n3": {"k": 4.0, "min_lines": 3}, "xw_k6_n4": {"k": 6.0, "min_lines": 4}}


def measure_clip(video: Path, every: int = 3, variants: bool = False) -> list[dict]:
    """variants=True 면 같은 프레임에서 횡단보도 거른 측정(XW_CONFIGS)과 노면 움직임(motion)도 남긴다 (§17-6)."""
    from edge.blackbox.quality import measure as quality  # 정현우 코드 — 읽기 전용으로 부른다

    from .lane_wear_variants import MotionProbe, measure_without_crosswalk

    meter = LaneWearMeter()
    probe = MotionProbe()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    recs, i = [], 0
    while cap.grab():
        if i % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            q = quality(frame)
            if not (q.ok and variants):
                r, segs = meter.measure_debug(frame) if q.ok else (None, [])
                recs.append(record(i, i / fps, q.ok, q.reason, r, segs))
            else:
                parts = meter._segments(frame)
                r, segs = meter._finish(parts)
                rec = record(i, i / fps, q.ok, q.reason, r, segs)
                rec["xw"] = {}
                for name, kw in XW_CONFIGS.items():
                    rx, sx, nd = measure_without_crosswalk(meter, frame, parts=parts, **kw)
                    rec["xw"][name] = {k: v for k, v in record(i, 0, True, "", rx, sx).items()
                                       if k not in ("i", "t", "q_ok", "q_reason")} | {"n_dropped": nd}
                recs.append(rec)
            if variants:
                recs[-1]["motion"] = probe.update(frame, i / fps)
        i += 1
    cap.release()
    return recs


def read_frames(video: Path, idx: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out = {}
    for i in sorted(set(idx)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out[i] = f
    cap.release()
    return out


def overlay(frame: np.ndarray, rec: dict, j: dict | None, meter: LaneWearMeter, width: int = 640) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.polylines(img, [meter.roi_polygon_px(w, h)], True, (255, 200, 0), 2)
    worn = bool(j and j.get("is_worn"))
    for x1, y1, x2, y2 in rec.get("segs", []):
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255) if worn else (0, 255, 0), 4)
    s = img.shape[1] / width
    img = cv2.resize(img, (width, round(h / s)), interpolation=cv2.INTER_AREA)
    sc = j.get("score") if j else None
    txt = f"t={rec['t']:.1f}s score={'-' if sc is None else f'{sc:.2f}'} v={j.get('votes', 0) if j else 0}"
    if j and j.get("confirmed"):
        txt += " CONFIRMED"
    cv2.rectangle(img, (0, 0), (width, 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(tiles: list[np.ndarray], cols: int = 3) -> np.ndarray:
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    out = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        out[r * th:(r + 1) * th, c * tw:c * tw + t.shape[1]] = t[:th]
    return out


def bins_with_events(recs: list[dict], bin_s: float = BIN_S) -> list[tuple[float, float]]:
    """모든 주기·기준·투표의 확정 시각이 하나라도 든 bin_s 초 칸 [(시작, 끝)]. 눈 분류 단위."""
    ks = sorted({int(t // bin_s) for m in EVERY_MULT for thr in THRESHOLDS for k, n in VOTES
                 for t in confirmed_times(replay(recs[::m], thr, k, n))})
    return [(k * bin_s, (k + 1) * bin_s - 1e-3) for k in ks]


def make_sheets(clip_dir: Path, video: Path, recs: list[dict], near: int = 12, seed: int = 0, per_sheet: int = 12) -> dict:
    """확정이 든 칸마다 그 칸에서 점수가 가장 낮은 판정 프레임 1장 (확정을 끌어낸 프레임) → per_sheet 장씩 sheet.
    그리고 기준 근처(|score−1| < 0.1) 프레임 near 장 sheet. 칸 목록을 bins.json 에 남긴다."""
    meter = LaneWearMeter()
    base = {j["i"]: j for j in replay(recs)}
    by_i = {r["i"]: r for r in recs}
    bins = bins_with_events(recs)
    pick = []
    for a, b in bins:
        js = [j for j in base.values() if a <= j["t"] <= b and j["score"] is not None]
        pick.append(min(js, key=lambda j: j["score"])["i"] if js else None)
    rng = np.random.default_rng(seed)
    nr = [j["i"] for j in base.values() if j["score"] is not None and abs(j["score"] - 1.0) < 0.1]
    near_idx = sorted(rng.choice(nr, size=min(near, len(nr)), replace=False).tolist()) if nr else []
    frames = read_frames(video, [i for i in pick if i is not None] + near_idx)
    sheets = clip_dir / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    tiles, names = [], []
    for bi, i in enumerate(pick):
        if i is None or i not in frames:
            continue
        t = overlay(frames[i], by_i[i], base.get(i), meter)
        cv2.putText(t, f"b{bi:03d}", (560, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
        names.append(bi)
    for s0 in range(0, len(tiles), per_sheet):
        cv2.imwrite(str(sheets / f"bins_{names[s0]:03d}-{names[min(s0 + per_sheet, len(tiles)) - 1]:03d}.jpg"),
                    grid(tiles[s0:s0 + per_sheet]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if near_idx:
        tl = [overlay(frames[i], by_i[i], base.get(i), meter) for i in near_idx if i in frames]
        cv2.imwrite(str(sheets / "near_threshold.jpg"), grid(tl, cols=4), [cv2.IMWRITE_JPEG_QUALITY, 85])
    info = {"bin_s": BIN_S, "bins": [{"id": bi, "start": a, "end": b, "frame": pick[bi]} for bi, (a, b) in enumerate(bins)],
            "near_threshold": near_idx}
    (clip_dir / "bins.json").write_text(json.dumps(info, indent=1))
    return info


# §17-6 개선 후보 (lane_wear_variants). 이름 → replay(variant=...)
VARIANTS = {
    "base": {},
    "motion": {"min_motion": 3.0},
    "crosswalk_k4": {"seg": "xw_k4_n3"},
    "crosswalk_k6": {"seg": "xw_k6_n4"},
    "p25_only": {"mode": "p25_only"},
    "rel_len": {"mode": "p25_x_rel_len"},
    "rel_len_cap": {"mode": "p25_x_rel_len_cap"},
    "sqrt_rel_len": {"mode": "p25_x_sqrt_rel_len"},
    "found_guard": {"min_found": 0.3},
    "motion+crosswalk_k6": {"min_motion": 3.0, "seg": "xw_k6_n4"},
    "motion+crosswalk_k6+sqrt_rel_len": {"min_motion": 3.0, "seg": "xw_k6_n4", "mode": "p25_x_sqrt_rel_len"},
    "all (motion+crosswalk_k6+sqrt_rel_len+found_guard)": {"min_motion": 3.0, "seg": "xw_k6_n4",
                                                           "mode": "p25_x_sqrt_rel_len", "min_found": 0.3},
}
VARIANT_THRESHOLDS = (0.4, 0.5, 0.6, 0.8, 1.0)


def variant_rows(recs: list[dict], clusters, labels: dict[int, str]) -> list[dict]:
    """개선 후보별 · 기준별 확정 수와 눈 분류 (주기 3 프레임, 5 중 3). 새 확정이 든 칸은 'unlabeled'."""
    rows = []
    for name, v in VARIANTS.items():
        for thr in VARIANT_THRESHOLDS:
            ts = confirmed_times(replay(recs, thr, CONFIRM_K, CONFIRM_N, v))
            c = {lab: 0 for lab in (*LABELS, "unlabeled")}
            for t in ts:
                c[label_of(t, clusters, labels) or "unlabeled"] += 1
            done = sum(c[x] for x in LABELS)
            rows.append({"variant": name, "threshold": thr, "confirmed": len(ts), **c,
                         "fp_rate": round(c["fp"] / done, 3) if done else None,
                         "real_135_140": sum(135 <= t <= 140 and label_of(t, clusters, labels) == "real" for t in ts)})
    return rows


def parse_stopped(specs: list[str]) -> dict[str, list[tuple[float, float]]]:
    """["clip:30-96,152-172"] → {"clip": [(30, 96), (152, 172)]}"""
    out: dict[str, list[tuple[float, float]]] = {}
    for sp in specs:
        clip, rng = sp.split(":")
        out[clip] = [tuple(float(x) for x in r.split("-")) for r in rng.split(",")]
    return out


def _clip_id(video: Path) -> str:
    return video.stem


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--clips", nargs="+", type=Path, required=True)
    m.add_argument("--every", type=int, default=3, help="블랙박스 --lane-every 와 같은 프레임 주기")
    m.add_argument("--variants", action="store_true", help="§17-6 개선 후보용 값도 남긴다 (횡단보도 거름·노면 움직임)")
    s = sub.add_parser("sheets")
    s.add_argument("--near", type=int, default=12)
    u = sub.add_parser("summarize")
    u.add_argument("--out", type=Path, default=Path("experiments/lane_wear_demo.json"))
    u.add_argument("--stopped", nargs="*", default=[], metavar="CLIP:A-B",
                   help="차가 서 있던 구간 (눈으로 확인). 빼고 다시 센 확정 수를 moving_only 에 남긴다")
    for p in (m, s, u):
        p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args(argv)

    if a.cmd == "measure":
        for v in a.clips:
            d = a.out_dir / _clip_id(v)
            d.mkdir(parents=True, exist_ok=True)
            recs = measure_clip(v, a.every, a.variants)
            (d / "frames.json").write_text(json.dumps({"video": str(v), "every": a.every, "frames": recs}))
            print(v.name, len(recs), "frames", clip_stats(recs, replay(recs)))
        return

    clips = sorted(p.parent for p in a.out_dir.glob("*/frames.json"))
    if a.cmd == "sheets":
        for d in clips:
            c = json.loads((d / "frames.json").read_text())
            info = make_sheets(d, Path(c["video"]), c["frames"], a.near)
            print(d.name, len(info["bins"]), "bins")
        return

    review_p = a.out_dir / "review.json"
    review = json.loads(review_p.read_text()) if review_p.exists() else {}
    out = {"setup": {"lane_every_frames": None, "thresholds": THRESHOLDS, "votes": [f"{k}/{n}" for k, n in VOTES],
                     "bin_s": BIN_S, "review": "Claude 눈 분류 (확정이 든 2초 칸 단위, 칸에서 점수가 가장 낮은 프레임, real/amb/fp)"}, "clips": {}}
    for d in clips:
        c = json.loads((d / "frames.json").read_text())
        recs = c["frames"]
        out["setup"]["lane_every_frames"] = c["every"]
        info = json.loads((d / "bins.json").read_text())
        cl = [(x["start"], x["end"]) for x in info["bins"]]
        labels = {x["id"]: review[f"{d.name}/b{x['id']:03d}"] for x in info["bins"]
                  if f"{d.name}/b{x['id']:03d}" in review}
        judged = replay(recs)
        dur = recs[-1]["t"] - recs[0]["t"] if recs else 1.0
        base = [{"t": round(t, 2), "bin": next((ci for ci, (s0, e0) in enumerate(cl) if s0 <= t <= e0), None),
                 "label": label_of(t, cl, labels)} for t in confirmed_times(judged)]
        out["clips"][d.name] = {
            "duration_s": round(dur, 1), "stats": clip_stats(recs, judged),
            "confirmed_base": base,
            "worn_runs_base": [[round(x, 2), round(y, 2)] for x, y in worn_runs(judged)],
            "bin_labels": {lab: sum(v == lab for v in labels.values()) for lab in LABELS} | {"bins": len(cl)},
            "sweep": sweep(recs, cl, labels, dur, every_mult=EVERY_MULT, every=c["every"]),
        }
        if "xw" in next((r for r in recs if r["q_ok"]), {}):
            out["clips"][d.name]["variants_17_6"] = variant_rows(recs, cl, labels)
        stops = parse_stopped(a.stopped).get(d.name)
        if stops:
            moving = [r for r in recs if not any(x <= r["t"] <= y for x, y in stops)]
            mdur = dur - sum(y - x for x, y in stops)
            out["clips"][d.name]["moving_only"] = {"stopped_s": stops, "sweep": sweep(
                moving, cl, labels, mdur, thresholds=(PROVISIONAL_WEAR_SCORE,), every_mult=EVERY_MULT, every=c["every"])}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])


if __name__ == "__main__":
    main()
