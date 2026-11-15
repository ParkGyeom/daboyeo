"""실제 1080p 주행 프레임에서 '추론 방식만 바꿔서' 작고 먼 결함을 더 잡는지 — 창(window)·조각(tile) 변형 비교.

    uv run python -m daboyeo_ai.eval.recall_res \\
        --model /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \\
        --zip "/Users/parkgyeom/daboyeo/data/roadmark/download/57809/087.도로_로드마크_인식을_위한_주행_영상_데이터/01.데이터/2.Validation/원천데이터/VS4.비온전(복합).zip" \\
        --mustkeep-dir /Users/parkgyeom/daboyeo/data/roadmark/verify_mustkeep \\
        --out-dir /Users/parkgyeom/daboyeo/data/recall_res

변형 (`VARIANTS`, 모두 같은 detector_v2.onnx·640 입력 — 모델 입력 크기가 640 으로 고정이라 960·1280 전체 프레임은
Colab 재내보내기가 필요하다. 여기서는 2×2 조각으로 '1280 비슷한 배율' 을 흉내 낸다):
  base        프레임 전체 640 letterbox (지금 블랙박스)
  roi         도로 영역(높이 45~100%, 가로 전체) 640 — 가로가 1920 그대로라 배율이 base 와 같다 (대조군)
  full_2x2    프레임 전체 2×2 조각 (각 ≈1056×594 → 배율 0.61, 1280 입력과 비슷한 배율, 4번)
  roi_t2      도로 영역 좌우 2조각 (배율 0.61, 2번)
  roi_t3      도로 영역 3조각 (배율 ≈0.94, 3번)
  base+roi_t2 / base+roi_t3   전체 프레임 + 도로 조각 (표지판 등 지평선 위를 지키기 위해)

입력: must-keep 목표 4개와 앞뒤 프레임 (`--mustkeep-dir`, verify_mustkeep 이 풀어 둔 PNG) + VS4 zip 에서 조건(날씨·낮밤·교통)
별로 고르게 뽑은 `--n-sample` 장 (정답 라벨 없음 → 클래스별 탐지 수만 세고 '새로 생긴 탐지' 를 모아 눈으로 본다).
출력(`--out-dir`, 레포 밖 — AI Hub 이미지는 재배포 불가): `frames/`, `mustkeep/<variant>/contact_<target>.jpg`,
`new_dets/<variant>.jpg` (base 에 없던 탐지만 크게 표시한 모음), `per_frame.csv`, `summary.json`.
레포에는 집계만: `--experiments` (기본 `experiments/recall_res.json`).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .verify_mustkeep import contact_sheet, seq_and_index

MUSTKEEP = {
    "0401_light_rainy_smooth_03000136": "pothole",
    "0401_light_rainy_smooth_03000120": "crack",
    "0401_light_rainy_smooth_03000130": "crack",
    "0401_light_clear_smooth_07000023": "crack",
}


def grid_windows(cols: int, rows: int, overlap: float = 0.1, top: float = 0.0) -> tuple:
    from daboyeo_ai.inference.onnx_detector import road_tiles

    xs = road_tiles(cols, overlap=overlap, full_frame=False)
    ys = road_tiles(rows, overlap=overlap, full_frame=False)  # 같은 계산을 세로에 재사용 (x 값만 쓴다)
    span = 1.0 - top
    return tuple((x1, top + y1 * span, x2, top + y2 * span) for (y1, _, y2, _) in ys for (x1, _, x2, _) in xs)


def variants() -> dict[str, tuple | None]:
    from daboyeo_ai.inference import road_tiles

    return {
        "base": None,
        "roi": ((0.0, 0.45, 1.0, 1.0),),
        "full_2x2": grid_windows(2, 2),
        "roi_t2": road_tiles(2, full_frame=False),
        "roi_t3": road_tiles(3, full_frame=False),
        "base+roi_t2": road_tiles(2),
        "base+roi_t3": road_tiles(3),
    }


def condition(stem: str) -> str:
    """`0401_light_rainy_smooth_03000136` → `light_rainy_smooth`."""
    return "_".join(stem.split("_")[1:4])


def sample_from_zip(zip_path: Path, out: Path, n: int, seed: int = 0) -> list[Path]:
    """조건별로 고르게 n 장을 뽑아 out 에 PNG 로 푼다 (이미 있으면 다시 안 푼다). 조건 안에서는 고른 간격."""
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(i.filename for i in zf.infolist() if i.filename.endswith(".png"))
        by_cond: dict[str, list[str]] = defaultdict(list)
        for name in names:
            by_cond[condition(Path(name).stem)].append(name)
        per = max(1, n // len(by_cond))
        picked = []
        for cond in sorted(by_cond):
            lst = by_cond[cond]
            idx = np.linspace(0, len(lst) - 1, min(per, len(lst))).round().astype(int)
            picked += [lst[i] for i in sorted(set(idx))]
        paths = []
        for name in picked:
            dst = out / Path(name).name
            if not dst.exists():
                dst.write_bytes(zf.read(name))
            paths.append(dst)
    return paths


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def new_vs_base(dets: list, base: list, thr: float = 0.3) -> list:
    """base 의 같은 클래스 탐지와 IoU < thr 인 탐지 (= 이 변형으로 새로 생긴 탐지)."""
    return [d for d in dets if not any(b[0] == d[0] and iou(b[2:], d[2:]) >= thr for b in base)]


def draw(img: np.ndarray, dets: list, class_conf: dict, highlight: list | None = None) -> np.ndarray:
    ov = img.copy()
    hl = {id(d) for d in highlight} if highlight is not None else None
    for d in dets:
        c, s, x1, y1, x2, y2 = d
        name = CLASS_NAMES[c]
        above = s >= class_conf[name]
        if hl is not None and id(d) not in hl:
            color, th = (160, 160, 160), 1
        else:
            color, th = ((0, 200, 0), 4) if above else ((0, 220, 255), 1)
        p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(ov, p1, p2, color, th)
        cv2.putText(ov, f"{name} {s:.2f}", (p1[0], max(22, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return ov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, default=Path("ai/models/detector_v2.onnx"))
    ap.add_argument("--card", type=Path, default=Path("ai/models/detector_v2.json"))
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--mustkeep-dir", type=Path, required=True)
    ap.add_argument("--n-sample", type=int, default=300)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--time-frames", type=int, default=30, help="속도 측정에 쓸 프레임 수 (변형·스레드마다)")
    ap.add_argument("--threads", type=int, default=4, help="탐지 수 세기용 스레드 (속도는 1·2 스레드로 따로 잰다)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--experiments", type=Path, default=Path("experiments/recall_res.json"))
    args = ap.parse_args(argv)

    from daboyeo_ai.inference import OnnxDetector, load_class_conf

    class_conf = load_class_conf(args.card)
    thr = [class_conf[n] for n in CLASS_NAMES]
    var = variants()
    out = args.out_dir
    sample = sample_from_zip(args.zip, out / "frames", args.n_sample)
    mk_frames = {p.stem: p for p in sorted(args.mustkeep_dir.glob("*.png"))}
    print(f"sample {len(sample)} frames, must-keep dir {len(mk_frames)} frames")

    # 1) 탐지 (낮은 기준으로 한 번 → 초안 기준으로 거름; 클래스별 NMS 라 결과가 같다)
    results: dict[str, dict[str, list]] = {}
    for name, wins in var.items():
        det = OnnxDetector(args.model, threads=args.threads, conf=args.low_conf, windows=wins)
        res = {}
        for p in list(mk_frames.values()) + sample:
            if p.stem not in res:
                res[p.stem] = det.raw(cv2.imread(str(p)))
        results[name] = res
        print(f"  {name}: done")

    def at_draft(dets):
        return [d for d in dets if d[1] >= thr[d[0]]]

    paths = {p.stem: p for p in sample} | mk_frames
    summary: dict = {"model": args.model.name, "class_conf_draft": class_conf, "n_sample": len(sample),
                     "sample_conditions": dict(Counter(condition(p.stem) for p in sample)), "variants": {}}

    # 2) must-keep
    for name in var:
        mk = {}
        vdir = out / "mustkeep" / name
        vdir.mkdir(parents=True, exist_ok=True)
        for stem, cls in MUSTKEEP.items():
            if stem not in mk_frames:
                mk[stem] = {"class": cls, "found": False}
                continue
            c = CLASS_NAMES.index(cls)
            seq, idx = seq_and_index(stem)
            near = sorted(s for s in mk_frames if seq_and_index(s)[0] == seq and abs(seq_and_index(s)[1] - idx) <= args.window)
            best = {s: max((d[1] for d in results[name][s] if d[0] == c), default=None) for s in near}
            mk[stem] = {"class": cls, "target_best_conf": best[stem],
                        "target_hit": best[stem] is not None and best[stem] >= class_conf[cls],
                        "window_frames": len(near),
                        "window_hits": sum(b is not None and b >= class_conf[cls] for b in best.values()),
                        "window_best_conf": {str(seq_and_index(s)[1]): (round(b, 3) if b else None) for s, b in best.items()}}
            ovs = []
            for s in near:
                ovp = vdir / f"{s}.jpg"
                cv2.imwrite(str(ovp), draw(cv2.imread(str(mk_frames[s])), results[name][s], class_conf), [cv2.IMWRITE_JPEG_QUALITY, 80])
                ovs.append(ovp)
            sheet = contact_sheet(ovs)
            if sheet is not None:
                cv2.imwrite(str(vdir / f"contact_{stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
            for p in ovs:
                p.unlink()
        summary["variants"][name] = {"windows": [list(w) for w in var[name]] if var[name] else None,
                                     "runs_per_frame": len(var[name]) if var[name] else 1, "mustkeep": mk}

    # 3) 샘플: 클래스별 탐지/프레임, base 대비 새 탐지
    rows = []
    (out / "new_dets").mkdir(parents=True, exist_ok=True)
    for name in var:
        cnt, new_cnt, lost_cnt = Counter(), Counter(), Counter()
        frames_with_new = []
        for p in sample:
            d = at_draft(results[name][p.stem])
            b = at_draft(results["base"][p.stem])
            new = new_vs_base(d, b)
            lost = new_vs_base(b, d)
            for x in d:
                cnt[CLASS_NAMES[x[0]]] += 1
            for x in new:
                new_cnt[CLASS_NAMES[x[0]]] += 1
            for x in lost:
                lost_cnt[CLASS_NAMES[x[0]]] += 1
            if new:
                frames_with_new.append((len(new), p.stem, d, new))
            rows.append({"variant": name, "frame": p.stem, "condition": condition(p.stem),
                         **{n: sum(CLASS_NAMES[x[0]] == n for x in d) for n in CLASS_NAMES},
                         **{f"new_{n}": sum(CLASS_NAMES[x[0]] == n for x in new) for n in CLASS_NAMES}})
        n = len(sample)
        by_cond = defaultdict(Counter)
        for r in rows:
            if r["variant"] == name:
                by_cond[r["condition"]].update({k: r[k] for k in CLASS_NAMES})
                by_cond[r["condition"]]["_n"] += 1
        summary["variants"][name]["sample"] = {
            "dets_per_frame": {k: round(cnt[k] / n, 3) for k in CLASS_NAMES},
            "new_vs_base_per_frame": {k: round(new_cnt[k] / n, 3) for k in CLASS_NAMES},
            "lost_vs_base_per_frame": {k: round(lost_cnt[k] / n, 3) for k in CLASS_NAMES},
            "frames_with_any_det": sum(1 for r in rows if r["variant"] == name and any(r[k] for k in CLASS_NAMES)),
            "frames_with_new": len(frames_with_new),
            "dets_per_frame_by_condition": {c: {k: round(v[k] / v["_n"], 3) for k in CLASS_NAMES}
                                            for c, v in sorted(by_cond.items())},
        }
        if name != "base" and frames_with_new:
            frames_with_new.sort(key=lambda t: (-t[0], t[1]))
            ovs = []
            for _, stem, d, new in frames_with_new[:24]:
                ovp = out / "new_dets" / f"_{stem}.jpg"
                cv2.imwrite(str(ovp), draw(cv2.imread(str(paths[stem])), d, class_conf, highlight=new))
                ovs.append(ovp)
            sheet = contact_sheet(ovs, cols=4, width=640)
            cv2.imwrite(str(out / "new_dets" / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
            for p in ovs:
                p.unlink()

    with (out / "per_frame.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # 4) 속도 (1·2 스레드, 초안 기준, 디코드·NMS·조각 자르기 포함)
    tf = [cv2.imread(str(p)) for p in sample[:: max(1, len(sample) // args.time_frames)][: args.time_frames]]
    for name, wins in var.items():
        sp = {}
        for th in (1, 2):
            det = OnnxDetector(args.model, threads=th, conf=class_conf, windows=wins)
            for im in tf[:3]:
                det.raw(im)  # 예열
            ms = []
            for im in tf:
                t0 = time.perf_counter()
                det.raw(im)
                ms.append((time.perf_counter() - t0) * 1000)
            sp[f"{th}t"] = {"mean_ms": round(float(np.mean(ms)), 1), "p95_ms": round(float(np.percentile(ms, 95)), 1)}
        summary["variants"][name]["speed"] = sp
        print(f"  {name}: {sp}")
    summary["speed_note"] = f"Apple M4 Pro, onnxruntime CPU, {len(tf)} frames 1920×1080, 실제 블랙박스 노트북에서 다시 잴 것"

    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    args.experiments.parent.mkdir(parents=True, exist_ok=True)
    args.experiments.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'variant':<13} {'runs':>4} {'ms 1t':>7} {'ms 2t':>7}  must-keep(target conf)  dets/frame p/c/g/s  new/frame p/c/g/s")
    for name, v in summary["variants"].items():
        mk = v["mustkeep"]
        hits = sum(m.get("target_hit", False) for m in mk.values())
        confs = " ".join(f"{m.get('target_best_conf') or 0:.2f}" for m in mk.values())
        dpf = "/".join(f"{x:.2f}" for x in v["sample"]["dets_per_frame"].values())
        npf = "/".join(f"{x:.2f}" for x in v["sample"]["new_vs_base_per_frame"].values())
        print(f"{name:<13} {v['runs_per_frame']:>4} {v['speed']['1t']['mean_ms']:>7} {v['speed']['2t']['mean_ms']:>7}  "
              f"{hits}/4 ({confs})  {dpf}  {npf}")
    print(f"\n→ {out}, {args.experiments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
