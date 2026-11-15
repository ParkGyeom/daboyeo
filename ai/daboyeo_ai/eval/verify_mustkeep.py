"""꼭 살아남아야 하는 결함 장면(must-keep) 확인 — 탐지기가 아직 잡는지, 그 크롭이 V0 검증을 통과하는지.

    uv run python -m daboyeo_ai.eval.verify_mustkeep \\
        --model /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \\
        --frames-dir /Users/parkgyeom/daboyeo/data/roadmark/verify_mustkeep \\
        --target 0401_light_rainy_smooth_03000136:pothole \\
        --target 0401_light_rainy_smooth_03000120:crack --target 0401_light_rainy_smooth_03000130:crack \\
        --target 0401_light_clear_smooth_07000023:crack \\
        --tau 0.05,0.1,0.2 --out-dir /Users/parkgyeom/daboyeo/data/verify_d

`--frames-dir` 에는 목표 프레임과 그 앞뒤 프레임 PNG 가 있다 (AI Hub 로드마크 654 VS4 에서 직접 풀어 둔 것 —
레포에 올리지 않는다). 로드마크 라벨에는 포트홀·균열이 없어서 '정답 박스' 가 없다 → 목표 클래스 탐지가 있는지만
보고, 그게 그 결함인지는 오버레이 이미지를 눈으로 확인한다.

프레임마다: 탐지기(초안 기준 `class_conf_draft`) 와 낮은 기준(`--low-conf`, 0.05 — 놓친 것이 얼마나 아래에
있는지 보려고) 결과, 목표 클래스 탐지마다 크롭 JPEG(`crop_snapshot` 사본) 의 V0 점수.
출력(`--out-dir`, 레포 밖): `overlay/<frame>.jpg` (초록 = 초안 기준 이상, 노랑 = 그 아래, 글자 = conf / v),
`crops/<frame>_<k>.jpg`, `contact_<target>.jpg` (목표 앞뒤 프레임 모음), `summary.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .snapshot import crop_snapshot, pixel_box


def seq_and_index(stem: str) -> tuple[str, int]:
    """`0401_light_rainy_smooth_03000136` → (`0401_light_rainy_smooth_03`, 136). 마지막 6자리가 프레임 번호."""
    head, num = stem.rsplit("_", 1)
    return f"{head}_{num[:-6]}", int(num[-6:])


def contact_sheet(paths: list[Path], cols: int = 4, width: int = 480) -> np.ndarray | None:
    tiles = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        tiles.append(cv2.resize(im, (width, round(im.shape[0] * width / im.shape[1]))))
    if not tiles:
        return None
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, default=Path("ai/models/detector_v2.onnx"))
    ap.add_argument("--card", type=Path, default=Path("ai/models/detector_v2.json"))
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--target", action="append", required=True, help="<frame stem>:<class>")
    ap.add_argument("--window", type=int, default=6, help="목표 프레임 앞뒤로 볼 프레임 수")
    ap.add_argument("--tau", default="0.05,0.1,0.2")
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    from daboyeo_ai.inference import OnnxDetector, load_class_conf
    from daboyeo_ai.inference.verifier import Verifier

    class_conf = load_class_conf(args.card)
    taus = [float(t) for t in args.tau.split(",") if t]
    det = OnnxDetector(args.model, threads=4, conf=args.low_conf)
    ver = Verifier(args.model, threads=1)
    (args.out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "crops").mkdir(parents=True, exist_ok=True)

    frames = {p.stem: p for p in sorted(args.frames_dir.glob("*.png"))}
    cache: dict[str, list[dict]] = {}

    def run_frame(stem: str) -> list[dict]:
        if stem in cache:
            return cache[stem]
        img = cv2.imread(str(frames[stem]))
        h, w = img.shape[:2]
        ov = img.copy()
        out = []
        for k, (c, s, x1, y1, x2, y2) in enumerate(det.raw(img)):
            name = CLASS_NAMES[c]
            box = pixel_box(x1, y1, x2, y2, w, h)
            jpg = crop_snapshot(img, box)
            v = ver.verify(jpg, name)
            above = s >= class_conf[name]
            out.append({"class": name, "conf": round(s, 4), "above_draft": bool(above), "v": round(v, 4),
                        "box": list(box)})
            if name in ("pothole", "crack"):
                (args.out_dir / "crops" / f"{stem}_{k}_{name}.jpg").write_bytes(jpg)
            color = (0, 200, 0) if above else (0, 220, 255)
            cv2.rectangle(ov, box[:2], box[2:], color, 3 if above else 1)
            cv2.putText(ov, f"{name} {s:.2f} / v{v:.2f}", (box[0], max(20, box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(args.out_dir / "overlay" / f"{stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cache[stem] = out
        return out

    summary = {"model": args.model.name, "detector_conf": class_conf, "low_conf": args.low_conf, "taus": taus,
               "targets": {}}
    for t in args.target:
        stem, cls = t.split(":")
        if stem not in frames:
            summary["targets"][stem] = {"class": cls, "found": False}
            print(f"[missing] {stem} 프레임이 {args.frames_dir} 에 없다")
            continue
        seq, idx = seq_and_index(stem)
        near = [s for s in frames if seq_and_index(s)[0] == seq and abs(seq_and_index(s)[1] - idx) <= args.window]
        per_frame = {}
        for s in near:
            dets = [d for d in run_frame(s) if d["class"] == cls]
            per_frame[s] = {
                "best_conf": max((d["conf"] for d in dets), default=None),
                "n_above_draft": sum(d["above_draft"] for d in dets),
                "best_v_above_draft": max((d["v"] for d in dets if d["above_draft"]), default=None),
                "dets": dets,
            }
        me = per_frame[stem]
        tgt = {"class": cls, "found": True, "frame": me,
               "detected_at_draft": me["n_above_draft"] > 0,
               "passes": {f"{tau:g}": bool(me["best_v_above_draft"] is not None and me["best_v_above_draft"] >= tau)
                          for tau in taus},
               "window_frames": len(near),
               "window_detected_at_draft": sum(f["n_above_draft"] > 0 for f in per_frame.values()),
               "window_pass": {f"{tau:g}": sum(f["best_v_above_draft"] is not None and f["best_v_above_draft"] >= tau
                                               for f in per_frame.values()) for tau in taus},
               "window": {s: {k: v for k, v in f.items() if k != "dets"} for s, f in sorted(per_frame.items())}}
        summary["targets"][stem] = tgt
        sheet = contact_sheet([args.out_dir / "overlay" / f"{s}.jpg" for s in sorted(near)])
        if sheet is not None:
            cv2.imwrite(str(args.out_dir / f"contact_{stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"\n{stem} ({cls}): 목표 프레임 best conf {me['best_conf']}, 초안 기준 이상 {me['n_above_draft']}개, "
              f"v {me['best_v_above_draft']} → " + ", ".join(f"τ{k}:{'통과' if p else '탈락'}" for k, p in tgt["passes"].items()))
        print(f"  앞뒤 {len(near)}프레임 중 초안 기준 탐지 {tgt['window_detected_at_draft']}, "
              + ", ".join(f"τ{k} 통과 {n}" for k, n in tgt["window_pass"].items()))
        for s, f in sorted(per_frame.items()):
            print(f"    {seq_and_index(s)[1]:>4}  best conf {f['best_conf']}  ≥초안 {f['n_above_draft']}  "
                  f"v {f['best_v_above_draft']}")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
