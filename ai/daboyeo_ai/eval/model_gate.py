"""시연 모델 교체 판단 — v3 (YOLO11n) 와 더 큰 새 모델 (v4, YOLO11s) 을 같은 평가로 나란히 놓고 기준을 판정한다 (handoff 21장).

새로 추론하지 않는다. 기존 도구가 만든 파일만 읽는다:

- 모델 카드 `ai/models/detector_<tag>.json` → 공개 test 클래스별 mAP50 (+ RDD2022 출처별 mAP50)
- `conf_thresholds` 결과 (`experiments/conf_thresholds_<tag>.json`) → v3 와 같은 규칙의 클래스별 기준
  (pothole·guardrail·sign = 정밀도 ≥ 0.7, crack = 정밀도 ≥ 0.8. `conf` 하위 명령이 만든다)
- `rdd_figs` 캐시 `<out>/preds.json` (RDD2022 test, 0.05 로 한 번 돌린 탐지) → 포트홀·균열 정밀도·재현율
- `eval_detector` 의 `summary.json` (시연 클립 gn_a·gn_b, strict 정답, `--every 30`) → 이벤트 단위 균열·포트홀·시설물
- `bench_cpu run` 의 `.json` (M4 Pro, 시연 클립 60 fps) → 1스레드 10프레임마다 실시간 비율

    uv run python -m daboyeo_ai.eval.model_gate conf experiments/conf_thresholds_v4.json --out experiments/class_conf_v4.json
    uv run python -m daboyeo_ai.eval.model_gate image --preds data/presentation/rdd_pothole/preds.json \\
        --conf ai/models/detector_v3.json --out experiments/image_level_v3.json   # 사진 단위 (handoff 22장, 판정 제외)
    uv run python -m daboyeo_ai.eval.model_gate judge --old v3 --new v4 \\
        --rdd v3=data/presentation/rdd_pothole/preds.json --rdd v4=data/model_gate/rdd_v4/preds.json \\
        --demo v3=data/model_gate/demo_v3/summary.json --demo v4=data/model_gate/demo_v4/summary.json \\
        --out experiments/model_gate_v4.json

교체 기준 (11/21 판단, 셋 다 만족해야 교체):
1. RDD2022 test 포트홀·균열 재현율이 **같은 정밀도에서** 각각 +5%p 이상 (`RECALL_GAIN`).
   같은 정밀도 = 새 모델 기준을 0.01 씩 올려 RDD 정밀도가 v3 (자기 기준에서) 이상이 되는 가장 낮은 기준.
   새 모델 자기 규칙 기준(`class_conf`)의 P/R 도 같이 적는다 (시연에 쓸 기준).
2. 시연 클립 균열: 찾은 정답 ≥ v3 이고 (v3 = 3) 헛것 이벤트 (예측 − 맞음) 가 v3 보다 많지 않음. 새 모델은 자기 규칙 기준으로 잰다.
3. CPU: 1스레드, 60 fps 영상을 10프레임마다 분석할 때 계산 시간 ÷ 영상 시간 (`rt_i10`) < 1 (모든 해상도).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .rdd_figs import country_of, filter_preds, pr_at

RDD_CLASSES = ("pothole", "crack")
P07_CLASSES = ("pothole", "guardrail", "sign")  # 정밀도 ≥ 0.7 규칙. crack 만 ≥ 0.8 (v3 확정, handoff 16장)
RECALL_GAIN = 0.05
MIN_SUPPORT = 20  # 같은 정밀도 기준을 찾을 때 탐지가 이만큼은 있어야 한다 (conf_thresholds 와 같음)
INTERVAL = 10
IMAGE_DEFINITIONS = {
    "box_iou50": "박스 단위: 경보 박스 하나가 같은 클래스 정답 박스와 IoU 0.5 이상 겹치면 맞음 (정답 하나에 경보 하나, 점수 높은 순). "
                 "정밀도 = 맞은 경보 / 전체 경보, 재현율 = 찾은 정답 박스 / 전체 정답 박스",
    "image_recall": "사진 단위 재현율 = 그 클래스 정답 박스가 있는 사진 중, 같은 클래스 경보가 하나라도 뜬 사진 비율 (위치 무관)",
    "image_precision": "사진 단위 정밀도 = 그 클래스 경보가 뜬 사진 중, 그 클래스 정답이 실제로 있는 사진 비율 (위치 무관)",
    "overlap": "겹침 조건 (참고): 경보 박스가 같은 클래스 정답 박스와 조금이라도 겹칠 때만 '알아챔'으로 센 사진 단위 값",
}
GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]  # 0.05 ~ 0.95


def rule_conf(thresholds: dict) -> dict[str, float]:
    """conf_thresholds.json → v3 규칙 기준 (pothole·guardrail·sign 정밀도 ≥ 0.7, crack 정밀도 ≥ 0.8)."""
    out = {c: float(thresholds["class_conf"][c]) for c in P07_CLASSES}
    out["crack"] = float(thresholds["classes"]["crack"]["alternatives"]["precision>=0.8"]["conf"])
    return {c: out[c] for c in ("pothole", "crack", "guardrail", "sign")}


def matched_precision(images: Sequence[dict], cls: str, target_p: float,
                      grid: Sequence[float] = GRID, min_support: int = MIN_SUPPORT) -> dict | None:
    """정밀도가 target_p 이상이고 탐지가 min_support 건 이상인 가장 낮은 기준의 {conf, tp, fp, n_gt, P, R}. 없으면 None."""
    for t in grid:
        r = pr_at(images, {cls: t})[cls]
        if r["tp"] + r["fp"] < min_support:
            return None  # 기준을 더 올리면 탐지만 줄어든다
        if r["P"] is not None and r["P"] >= target_p:
            return r
    return None


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    """두 박스 (x1, y1, x2, y2) 가 조금이라도 겹치는가 (넓이 > 0)."""
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def image_level(images: Sequence[dict], cls: str, conf: float) -> dict:
    """사진 단위 지표 (RDD test, 한 클래스, 기준 conf).

    - 재현율 = 그 클래스 정답 박스가 있는 사진 중, 같은 클래스 경보(conf 이상)가 하나라도 뜬 사진 비율
    - 정밀도 = 그 클래스 경보가 뜬 사진 중, 그 클래스 정답이 실제로 있는 사진 비율
    - `_overlap` 변형 (참고): 경보 박스가 같은 클래스 정답 박스와 조금이라도 겹칠 때만 '알아챔'.
      재현율 = 겹치는 경보가 있는 정답 사진 / 정답 사진, 정밀도 = 겹치는 경보가 있는 경보 사진 / 경보 사진
    """
    n_gt_img = n_alert_img = hit = hit_ov = 0
    for im in images:
        gts = [g["box"] for g in im["gts"] if g["cls"] == cls]
        alerts = [p["box"] for p in filter_preds(im["preds"], {cls: conf}) if p["cls"] == cls]
        ov = any(_overlaps(a, g) for a in alerts for g in gts)
        n_gt_img += bool(gts)
        n_alert_img += bool(alerts)
        hit += bool(gts) and bool(alerts)
        hit_ov += ov

    def r(a, b):
        return round(a / b, 3) if b else None

    return {"conf": conf, "images": len(images), "gt_images": n_gt_img, "alert_images": n_alert_img,
            "hit_images": hit, "hit_images_overlap": hit_ov,
            "recall": r(hit, n_gt_img), "precision": r(hit, n_alert_img),  # 둘 다 분자 = 정답도 경보도 있는 사진
            "recall_overlap": r(hit_ov, n_gt_img), "precision_overlap": r(hit_ov, n_alert_img)}


def image_report(images: Sequence[dict], class_conf: dict[str, float], exclude_countries: Sequence[str] = ()) -> dict:
    """포트홀·균열마다 박스 단위(IoU 0.5) + 사진 단위 (+ 겹침 변형). exclude_countries 의 사진은 뺀다."""
    ims = [im for im in images if country_of(im["name"]) not in set(exclude_countries)]
    out = {"images": len(ims), "excluded_countries": list(exclude_countries)}
    for cls in RDD_CLASSES:
        out[cls] = {"box_iou50": pr_at(ims, {cls: class_conf[cls]})[cls], "image": image_level(ims, cls, class_conf[cls])}
    return out


def rdd_compare(old_images: Sequence[dict], new_images: Sequence[dict],
                old_conf: dict[str, float], new_conf: dict[str, float]) -> dict:
    out = {}
    for cls in RDD_CLASSES:
        old = pr_at(old_images, {cls: old_conf[cls]})[cls]
        own = pr_at(new_images, {cls: new_conf[cls]})[cls]
        same_p = matched_precision(new_images, cls, old["P"]) if old["P"] is not None else None
        gain = round(same_p["R"] - old["R"], 3) if same_p else None
        out[cls] = {"old": old, "new_own_rule": own, "new_same_precision": same_p, "recall_gain": gain,
                    "pass": gain is not None and gain >= RECALL_GAIN - 1e-9}
    return out


def demo_rows(summary: dict) -> dict[str, dict]:
    """eval_detector summary.json → 전체 합(scope all) 클래스별 기준 행 {defect_type: row}."""
    rows = [r for r in summary["rows"] if r["scope"] == "all" and str(r["conf"]).startswith("per_class")]
    return {r["defect_type"]: r for r in rows}


def demo_compare(old: dict, new: dict) -> dict:
    o, n = demo_rows(old), demo_rows(new)
    out = {}
    for dtype in ("crack", "pothole", "guardrail_damage", "sign_damage"):
        if dtype not in o or dtype not in n:
            continue
        out[dtype] = {tag: {"n_gt": int(r["n_gt"]), "found": int(r["n_gt_found"]), "pred": int(r["n_pred"]),
                            "ok": int(r["n_pred_ok"]), "false": int(r["n_pred"]) - int(r["n_pred_ok"]),
                            "precision": r["precision"], "recall": r["recall"], "missed": r["missed_ids"]}
                      for tag, r in (("old", o[dtype]), ("new", n[dtype]))}
    c = out.get("crack")
    out["crack_pass"] = bool(c) and c["new"]["found"] >= c["old"]["found"] and c["new"]["false"] <= c["old"]["false"]
    return out


def bench_rows(bench: dict, interval: int = INTERVAL) -> dict[int, dict]:
    """bench_cpu json → {threads: 가장 느린 해상도의 {median_ms, rt}} (탐지만)."""
    out: dict[int, dict] = {}
    for r in bench["rows"]:
        if r["stage"] != "detect":
            continue
        rt = r.get(f"rt_i{interval}")
        cur = out.get(r["threads"])
        if cur is None or rt > cur["rt"]:
            out[r["threads"]] = {"median_ms": r["median_ms"], "rt": rt, "res": f"{r['width']}x{r['height']}"}
    return out


def bench_compare(old: dict, new: dict) -> dict:
    o, n = bench_rows(old), bench_rows(new)
    return {"video_fps": new.get("video_fps"), "interval": INTERVAL, "old": o, "new": n,
            "pass": 1 in n and n[1]["rt"] is not None and n[1]["rt"] < 1.0}


def public_map50(card: dict) -> dict:
    test = card.get("metrics", {}).get("test", {})
    rdd = card.get("eval_by_source", {}).get("rdd2022_test", {}).get(card.get("version"), {})
    return {"test": {c: test[c]["mAP50"] for c in test}, "rdd2022_test": {c: rdd.get(c) for c in RDD_CLASSES}}


def judge(old_card, new_card, old_conf, new_conf, old_rdd, new_rdd, old_demo, new_demo, old_bench, new_bench) -> dict:
    rdd = rdd_compare(old_rdd, new_rdd, old_conf, new_conf)
    demo = demo_compare(old_demo, new_demo)
    bench = bench_compare(old_bench, new_bench)
    gates = {"rdd_recall": all(rdd[c]["pass"] for c in RDD_CLASSES), "demo_crack": demo["crack_pass"],
             "cpu_realtime": bench["pass"]}
    images = {"old": image_report(old_rdd, old_conf), "new": image_report(new_rdd, new_conf)}
    return {"public_mAP50": {"old": public_map50(old_card), "new": public_map50(new_card)},
            "rdd2022_test_image_level": images,
            "class_conf": {"old": old_conf, "new": new_conf}, "rdd2022_test": rdd, "demo_clips": demo, "cpu": bench,
            "gates": gates, "replace": all(gates.values())}


def _pct(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def print_report(rep: dict, old: str, new: str) -> None:
    pm = rep["public_mAP50"]
    print(f"== 공개 test mAP50 ({old} → {new}) ==")
    for c in ("pothole", "crack", "guardrail", "sign", "all"):
        print(f"  {c:9s} {pm['old']['test'].get(c)} → {pm['new']['test'].get(c)}")
    print(f"  RDD2022 만: pothole {pm['old']['rdd2022_test']['pothole']} → {pm['new']['rdd2022_test']['pothole']}, "
          f"crack {pm['old']['rdd2022_test']['crack']} → {pm['new']['rdd2022_test']['crack']}")
    print(f"\n== 클래스별 기준 (v3 규칙) ==\n  {old}: {rep['class_conf']['old']}\n  {new}: {rep['class_conf']['new']}")
    print("\n== RDD2022 test (IoU 0.5) ==")
    for c, r in rep["rdd2022_test"].items():
        o, own, sp = r["old"], r["new_own_rule"], r["new_same_precision"]
        sp_s = f"conf {sp['conf']:.2f} P {_pct(sp['P'])} R {_pct(sp['R'])}" if sp else "없음"
        print(f"  {c:8s} {old} conf {o['conf']:.2f} P {_pct(o['P'])} R {_pct(o['R'])} | "
              f"{new} 자기 기준 conf {own['conf']:.2f} P {_pct(own['P'])} R {_pct(own['R'])} | "
              f"같은 정밀도 {sp_s} | 재현율 +{_pct(r['recall_gain'])} {'✓' if r['pass'] else '✗'}")
    print("\n== RDD2022 test 사진 단위 (각자 기준, 판정 제외) ==")
    for side, tag in (("old", old), ("new", new)):
        print_image_report(rep["rdd2022_test_image_level"][side], tag)
    print("\n== 시연 클립 gn_a·gn_b (이벤트, strict) ==")
    for d, r in rep["demo_clips"].items():
        if d == "crack_pass":
            continue
        o, n = r["old"], r["new"]
        print(f"  {d:16s} 정답 {o['n_gt']} | {old} 찾음 {o['found']} 경보 {o['pred']} (진짜 {o['ok']}, 헛것 {o['false']}) | "
              f"{new} 찾음 {n['found']} 경보 {n['pred']} (진짜 {n['ok']}, 헛것 {n['false']})")
    cpu = rep["cpu"]
    print(f"\n== CPU ({cpu['video_fps']} fps 영상, {cpu['interval']} 프레임마다, 가장 느린 해상도) ==")
    for th in sorted(set(cpu["old"]) | set(cpu["new"])):
        o, n = cpu["old"].get(th, {}), cpu["new"].get(th, {})
        print(f"  {th}스레드: {old} {o.get('median_ms')} ms (rt {o.get('rt')}) → {new} {n.get('median_ms')} ms (rt {n.get('rt')})")
    g = rep["gates"]
    print(f"\n== 교체 기준 ==\n  RDD 재현율 +{RECALL_GAIN:.0%}p (포트홀·균열): {'✓' if g['rdd_recall'] else '✗'}"
          f"\n  시연 클립 균열 v3 이상·헛것 증가 없음: {'✓' if g['demo_crack'] else '✗'}"
          f"\n  1스레드 실시간: {'✓' if g['cpu_realtime'] else '✗'}"
          f"\n  → {'교체 (' + new + ')' if rep['replace'] else old + ' 유지'}")


def print_image_report(rep: dict, tag: str) -> None:
    ex = f", {'·'.join(rep['excluded_countries'])} 제외" if rep["excluded_countries"] else ""
    for cls in RDD_CLASSES:
        b, i = rep[cls]["box_iou50"], rep[cls]["image"]
        print(f"  {tag} {cls:8s} conf {i['conf']:.2f} ({rep['images']}장{ex}) | 박스 P {_pct(b['P'])} R {_pct(b['R'])} | "
              f"사진 P {_pct(i['precision'])} R {_pct(i['recall'])} (정답 사진 {i['gt_images']}, 경보 사진 {i['alert_images']}) | "
              f"겹침 조건 P {_pct(i['precision_overlap'])} R {_pct(i['recall_overlap'])}")


def _pairs(items: Sequence[str]) -> dict[str, Path]:
    out = {}
    for s in items:
        tag, _, path = s.partition("=")
        if not path:
            raise SystemExit(f"TAG=경로 형식이어야 한다: {s}")
        out[tag] = Path(path)
    return out


def _load(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("conf", help="conf_thresholds.json → v3 규칙 클래스별 기준 파일 ({class_conf: ...})")
    c.add_argument("thresholds", type=Path)
    c.add_argument("--out", type=Path, required=True)
    im = sub.add_parser("image", help="RDD test 사진 단위 지표 (박스 단위와 나란히) — 모델 하나씩, judge 전에도 쓸 수 있다")
    im.add_argument("--preds", type=Path, required=True, help="rdd_figs preds.json")
    im.add_argument("--conf", type=Path, required=True, help="class_conf 파일 (모델 카드 또는 model_gate conf 결과)")
    im.add_argument("--exclude", default="United_States", help="따로 한 번 더 잴 때 뺄 나라 (쉼표). 전체 값은 항상 같이")
    im.add_argument("--out", type=Path)
    j = sub.add_parser("judge", help="v3 vs 새 모델 비교 표 + 교체 기준 판정")
    j.add_argument("--old", default="v3")
    j.add_argument("--new", default="v4")
    j.add_argument("--models-dir", type=Path, default=Path("ai/models"))
    j.add_argument("--exp-dir", type=Path, default=Path("experiments"))
    j.add_argument("--rdd", action="append", default=[], help="TAG=rdd_figs preds.json (두 모델 모두)")
    j.add_argument("--demo", action="append", default=[], help="TAG=eval_detector summary.json (strict, 두 모델 모두)")
    j.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "conf":
        th = _load(a.thresholds)
        conf = rule_conf(th)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"class_conf": conf, "model": th.get("model"), "source": str(a.thresholds),
                                     "rule": "pothole·guardrail·sign 정밀도 ≥ 0.7, crack 정밀도 ≥ 0.8 (v3 와 같은 규칙, handoff 16·21장)"},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
        print(conf)
        return 0

    if a.cmd == "image":
        from daboyeo_ai.inference import load_class_conf

        conf = load_class_conf(a.conf)
        images = _load(a.preds)["images"]
        ex = [c for c in a.exclude.split(",") if c]
        rep = {"preds": str(a.preds), "conf_file": str(a.conf), "class_conf": {c: conf[c] for c in RDD_CLASSES},
               "definitions": IMAGE_DEFINITIONS, "all": image_report(images, conf)}
        if ex:
            rep["excluding"] = image_report(images, conf, ex)
        for k in ("all", "excluding"):
            if k in rep:
                print_image_report(rep[k], _load(a.preds).get("model", ""))
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"저장: {a.out}", file=sys.stderr)
        return 0

    rdd, demo = _pairs(a.rdd), _pairs(a.demo)
    for tag in (a.old, a.new):
        if tag not in rdd or tag not in demo:
            ap.error(f"--rdd {tag}=… 와 --demo {tag}=… 가 필요하다")
    tags = (a.old, a.new)
    card = {t: _load(a.models_dir / f"detector_{t}.json") for t in tags}
    conf = {t: rule_conf(_load(a.exp_dir / f"conf_thresholds_{t}.json")) for t in tags}
    bench = {t: _load(a.exp_dir / f"bench_cpu_{t}.json") for t in tags}
    rep = judge(card[a.old], card[a.new], conf[a.old], conf[a.new],
                _load(rdd[a.old])["images"], _load(rdd[a.new])["images"],
                _load(demo[a.old]), _load(demo[a.new]), bench[a.old], bench[a.new])
    rep = {"old": a.old, "new": a.new, **rep}
    print_report(rep, a.old, a.new)
    if a.out:
        a.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n저장: {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
