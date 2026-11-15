"""탐지 v3(크랙 라벨 규칙 v3) vs v2 — 실제 1080p 주행 프레임에서 균열을 잡는지, 헛것이 늘었는지 (Mac, ONNX).

    uv run python -m daboyeo_ai.eval.compare_v3 \\
        --old /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \\
        --new /Users/parkgyeom/daboyeo/ai/models/detector_v3.onnx \\
        --mustkeep-dir /Users/parkgyeom/daboyeo/data/roadmark/verify_mustkeep \\
        --sample-dir /Users/parkgyeom/daboyeo/data/recall_res/frames \\
        --out-dir /Users/parkgyeom/daboyeo/data/compare_v3

- must-keep: 로드마크 654 VS4 의 균열 3개 + 포트홀 1개 (`recall_res.MUSTKEEP`) 와 앞뒤 ±`--window` 프레임.
- 표본: `recall_res` 가 풀어 둔 300장 (`data/recall_res/frames`, 6개 조건 × 50장). 정답이 없어서 클래스별 탐지 수와
  'v2 에 없던 새 탐지' 를 센다 → 새 탐지 모음 이미지를 눈으로 본다.
- 변형: `base` (지금 블랙박스, 전체 640 한 번) 와 `base+roi_t2` (`road_tiles(2)`).
- 기준: 두 모델 모두 **v2 모델 카드의 `class_conf_draft`** (crack 0.29 등) — `--conf-card` 로 바꿀 수 있다.

RDD2022·179 test mAP 비교는 GPU 가 필요해서 Colab `02_train_detector` 의 '출처별 평가' 셀에서 한다. 그 결과
(`detector_v3.json` 의 `eval_by_source`) 를 `--new-card` 로 주면 Go 판정에 같이 넣는다.
출력: `--out-dir` (레포 밖 — AI Hub 이미지 재배포 금지) 에 `mustkeep/<model>_<variant>/contact_*.jpg`,
`new_dets/<variant>.jpg`, `summary.json`. 레포에는 집계만 `--experiments` (기본 `experiments/compare_v3.json`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from daboyeo_contracts import CLASS_NAMES

from .recall_res import MUSTKEEP, contact_sheet, draw, new_vs_base
from .verify_mustkeep import seq_and_index

# Go 기준 (제안 — 11/18 체크포인트에서 박겸 확정)
GO_MIN_CRACK_HITS = 2          # must-keep 균열 3개 중 카드 기준 이상으로 잡아야 하는 수 (base 변형)
GO_MAX_MAP_DROP = 0.02         # RDD2022 test crack·pothole mAP50 이 v2 보다 이만큼 넘게 떨어지면 No-go
GO_MAX_NEW_PER_FRAME = 0.10    # 300장 표본에서 v2 에 없던 pothole+crack 새 탐지/프레임 (base). v2 base 는 0
GO_MAX_LOST_GS_PER_FRAME = 0.03  # guardrail+sign 을 v2 보다 잃은 수/프레임 (v2 base 는 0.37/프레임 탐지)


def variants() -> dict[str, tuple | None]:
    from daboyeo_ai.inference import road_tiles

    return {"base": None, "base+roi_t2": road_tiles(2)}


def mustkeep_stats(results: dict[str, list], frames: dict[str, Path], class_conf: dict, window: int) -> dict:
    """must-keep 목표마다 목표 프레임 최고 conf·명중, 앞뒤 프레임 중 명중 수."""
    out = {}
    for stem, cls in MUSTKEEP.items():
        if stem not in frames:
            out[stem] = {"class": cls, "found": False}
            continue
        c = CLASS_NAMES.index(cls)
        seq, idx = seq_and_index(stem)
        near = sorted(s for s in frames if seq_and_index(s)[0] == seq and abs(seq_and_index(s)[1] - idx) <= window)
        best = {s: max((d[1] for d in results[s] if d[0] == c), default=None) for s in near}
        out[stem] = {"class": cls, "target_best_conf": None if best[stem] is None else round(best[stem], 3),
                     "target_hit": best[stem] is not None and best[stem] >= class_conf[cls],
                     "window_frames": len(near),
                     "window_hits": sum(b is not None and b >= class_conf[cls] for b in best.values())}
    return out


def sample_stats(new: dict[str, list], old: dict[str, list], thr: list[float]) -> dict:
    """표본 프레임: 새 모델의 클래스별 탐지/프레임, 옛 모델 대비 새로 생긴·잃은 탐지/프레임 (카드 기준 이상만)."""
    cnt, cnt_old, gained, lost = Counter(), Counter(), Counter(), Counter()
    frames_new = 0
    for stem in new:
        d = [x for x in new[stem] if x[1] >= thr[x[0]]]
        b = [x for x in old[stem] if x[1] >= thr[x[0]]]
        g, ls = new_vs_base(d, b), new_vs_base(b, d)
        frames_new += bool(g)
        for src, c in ((d, cnt), (b, cnt_old), (g, gained), (ls, lost)):
            for x in src:
                c[CLASS_NAMES[x[0]]] += 1
    n = max(1, len(new))
    per = lambda c: {k: round(c[k] / n, 3) for k in CLASS_NAMES}  # noqa: E731
    return {"frames": len(new), "dets_per_frame": per(cnt), "old_dets_per_frame": per(cnt_old),
            "new_vs_old_per_frame": per(gained), "lost_vs_old_per_frame": per(lost), "frames_with_new": frames_new}


def go_decision(summary: dict, new_card: dict | None = None, old_card: dict | None = None) -> dict:
    """Go 기준 (모듈 상단 상수) 별 통과 여부. mAP 는 카드에 `eval_by_source.rdd2022_test` 가 있을 때만."""
    base = summary["variants"]["base"]
    hits = sum(m.get("target_hit", False) for m in base["new"]["mustkeep"].values() if m["class"] == "crack")
    s = base["sample"]
    new_pc = s["new_vs_old_per_frame"]["pothole"] + s["new_vs_old_per_frame"]["crack"]
    lost_gs = s["lost_vs_old_per_frame"]["guardrail"] + s["lost_vs_old_per_frame"]["sign"]
    checks = {
        "mustkeep_crack_hits": {"value": hits, "need": f">= {GO_MIN_CRACK_HITS} of 3", "ok": hits >= GO_MIN_CRACK_HITS},
        "sample_new_pothole_crack_per_frame": {"value": round(new_pc, 3), "need": f"<= {GO_MAX_NEW_PER_FRAME}",
                                               "ok": new_pc <= GO_MAX_NEW_PER_FRAME},
        "sample_lost_guardrail_sign_per_frame": {"value": round(lost_gs, 3), "need": f"<= {GO_MAX_LOST_GS_PER_FRAME}",
                                                 "ok": lost_gs <= GO_MAX_LOST_GS_PER_FRAME},
    }
    rdd_new = ((new_card or {}).get("eval_by_source") or {}).get("rdd2022_test", {}).get("v3")
    rdd_old = ((new_card or {}).get("eval_by_source") or {}).get("rdd2022_test", {}).get("v2") or (
        ((old_card or {}).get("eval_by_source") or {}).get("rdd2022_test", {}).get("v2"))
    for cls in ("crack", "pothole"):
        key = f"rdd2022_test_{cls}_mAP50_drop"
        if rdd_new and rdd_old and cls in rdd_new and cls in rdd_old:
            drop = round(rdd_old[cls] - rdd_new[cls], 3)
            checks[key] = {"value": drop, "need": f"<= {GO_MAX_MAP_DROP}", "ok": drop <= GO_MAX_MAP_DROP}
        else:
            checks[key] = {"value": None, "need": f"<= {GO_MAX_MAP_DROP}", "ok": None,
                           "note": "--new-card 에 Colab 출처별 평가(eval_by_source.rdd2022_test.v3) 가 없음"}
    known = [c["ok"] for c in checks.values() if c["ok"] is not None]
    return {"checks": checks, "go": all(known) if len(known) == len(checks) else None,
            "all_known_ok": all(known)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", type=Path, default=Path("ai/models/detector_v2.onnx"))
    ap.add_argument("--new", type=Path, default=Path("ai/models/detector_v3.onnx"))
    ap.add_argument("--conf-card", type=Path, default=Path("ai/models/detector_v2.json"),
                    help="두 모델에 같이 쓸 class_conf_draft 가 있는 카드")
    ap.add_argument("--new-card", type=Path, default=None, help="detector_v3.json (RDD mAP Go 판정용, 선택)")
    ap.add_argument("--mustkeep-dir", type=Path, required=True)
    ap.add_argument("--sample-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--low-conf", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--experiments", type=Path, default=Path("experiments/compare_v3.json"))
    args = ap.parse_args(argv)

    from daboyeo_ai.inference import OnnxDetector, load_class_conf

    class_conf = load_class_conf(args.conf_card)
    thr = [class_conf[n] for n in CLASS_NAMES]
    mk_frames = {p.stem: p for p in sorted(args.mustkeep_dir.glob("*.png"))}
    sample = {p.stem: p for p in sorted(args.sample_dir.glob("*.png"))}
    paths = sample | mk_frames
    print(f"must-keep dir {len(mk_frames)} frames, sample {len(sample)} frames")
    models = {"old": args.old, "new": args.new}

    results: dict[tuple[str, str], dict[str, list]] = {}
    for mk, mpath in models.items():
        for vn, wins in variants().items():
            det = OnnxDetector(mpath, threads=args.threads, conf=args.low_conf, windows=wins)
            results[(mk, vn)] = {s: det.raw(cv2.imread(str(p))) for s, p in paths.items()}
            print(f"  {mpath.name} {vn}: done")

    summary: dict = {"old": args.old.name, "new": args.new.name, "class_conf": class_conf,
                     "conf_card": args.conf_card.name, "window": args.window, "variants": {}}
    (args.out_dir / "new_dets").mkdir(parents=True, exist_ok=True)
    for vn in variants():
        v = {}
        for mk in models:
            res = results[(mk, vn)]
            v[mk] = {"mustkeep": mustkeep_stats(res, mk_frames, class_conf, args.window)}
            vdir = args.out_dir / "mustkeep" / f"{mk}_{vn}"
            vdir.mkdir(parents=True, exist_ok=True)
            for stem in MUSTKEEP:
                if stem not in mk_frames:
                    continue
                seq, idx = seq_and_index(stem)
                near = sorted(s for s in mk_frames if seq_and_index(s)[0] == seq
                              and abs(seq_and_index(s)[1] - idx) <= args.window)
                ovs = []
                for s in near:
                    p = vdir / f"_{s}.jpg"
                    cv2.imwrite(str(p), draw(cv2.imread(str(mk_frames[s])), res[s], class_conf),
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                    ovs.append(p)
                sheet = contact_sheet(ovs)
                if sheet is not None:
                    cv2.imwrite(str(vdir / f"contact_{stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
                for p in ovs:
                    p.unlink()
        new_s = {s: results[("new", vn)][s] for s in sample}
        old_s = {s: results[("old", vn)][s] for s in sample}
        v["sample"] = sample_stats(new_s, old_s, thr)
        # 새 모델에서 새로 생긴 탐지 모음 (많은 순 24장)
        gained = []
        for s in sample:
            d = [x for x in new_s[s] if x[1] >= thr[x[0]]]
            g = new_vs_base(d, [x for x in old_s[s] if x[1] >= thr[x[0]]])
            if g:
                gained.append((len(g), s, d, g))
        gained.sort(key=lambda t: (-t[0], t[1]))
        ovs = []
        for _, s, d, g in gained[:24]:
            p = args.out_dir / "new_dets" / f"_{s}.jpg"
            cv2.imwrite(str(p), draw(cv2.imread(str(paths[s])), d, class_conf, highlight=g))
            ovs.append(p)
        if ovs:
            cv2.imwrite(str(args.out_dir / "new_dets" / f"{vn}.jpg"), contact_sheet(ovs, cols=4, width=640),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        for p in ovs:
            p.unlink()
        summary["variants"][vn] = v

    new_card = json.loads(args.new_card.read_text(encoding="utf-8")) if args.new_card else None
    old_card = json.loads(args.conf_card.read_text(encoding="utf-8"))
    summary["go"] = go_decision(summary, new_card, old_card)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    args.experiments.parent.mkdir(parents=True, exist_ok=True)
    args.experiments.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'variant':<12} {'model':<4} must-keep (target conf; ±{args.window} hits)")
    for vn, v in summary["variants"].items():
        for mk in models:
            cells = [f"{m['class'][0]}:{m.get('target_best_conf') or 0:.2f}"
                     f"{'*' if m.get('target_hit') else ''} {m.get('window_hits', 0)}/{m.get('window_frames', 0)}"
                     for m in v[mk]["mustkeep"].values()]
            print(f"{vn:<12} {mk:<4} " + " | ".join(cells))
        s = v["sample"]
        fmt = lambda d: "/".join(f"{d[k]:.2f}" for k in CLASS_NAMES)  # noqa: E731
        print(f"{'':<12} sample p/c/g/s dets/frame old {fmt(s['old_dets_per_frame'])} → new {fmt(s['dets_per_frame'])}; "
              f"new-vs-old {fmt(s['new_vs_old_per_frame'])}, lost {fmt(s['lost_vs_old_per_frame'])}, "
              f"frames with new {s['frames_with_new']}/{s['frames']}")
    print("\nGo checks:")
    for k, c in summary["go"]["checks"].items():
        print(f"  {k}: {c['value']} (need {c['need']}) -> {c['ok']}")
    print(f"  => go = {summary['go']['go']}  (None = RDD mAP 를 아직 안 넣음)")
    print(f"\n→ {args.out_dir}, {args.experiments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
