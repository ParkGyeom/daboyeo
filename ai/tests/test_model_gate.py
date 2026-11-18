"""model_gate — v3 규칙 기준, 같은 정밀도 재현율, 시연 클립·CPU 판정을 가짜 데이터로 시험한다."""

import json

from daboyeo_ai.eval import model_gate
from daboyeo_ai.eval.model_gate import bench_compare, demo_compare, matched_precision, rdd_compare, rule_conf

B = [0, 0, 10, 10]
FAR = [50, 50, 60, 60]


def _thresholds(p07=None, crack08=0.42):
    p07 = p07 or {"pothole": 0.31, "crack": 0.30, "guardrail": 0.21, "sign": 0.21}
    return {"class_conf": p07,
            "classes": {"crack": {"alternatives": {"precision>=0.8": {"conf": crack08}}}}}


def test_rule_conf_uses_p08_for_crack_only():
    assert rule_conf(_thresholds()) == {"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}


def _images(cls, hits, misses_fp, n_gt_extra=0):
    """hits = 맞는 탐지 점수들, misses_fp = 헛것 점수들. 이미지 하나에 정답·탐지 하나씩."""
    ims = [{"gts": [{"cls": cls, "box": B}], "preds": [{"cls": cls, "score": s, "box": B}]} for s in hits]
    ims += [{"gts": [], "preds": [{"cls": cls, "score": s, "box": FAR}]} for s in misses_fp]
    ims += [{"gts": [{"cls": cls, "box": B}], "preds": []} for _ in range(n_gt_extra)]
    return ims


def test_matched_precision_picks_lowest_conf_reaching_target():
    # 맞음 30개 (0.6), 헛것 10개 (0.2) + 10개 (0.5). 기준 0.21 이상이면 P = 30/40 = 0.75, 0.51 이상이면 1.0
    ims = _images("crack", [0.6] * 30, [0.2] * 10 + [0.5] * 10, n_gt_extra=10)
    r = matched_precision(ims, "crack", 0.75)
    assert r["conf"] == 0.21 and r["P"] == 0.75 and r["R"] == 0.75
    assert matched_precision(ims, "crack", 0.9)["conf"] == 0.51


def test_matched_precision_none_when_support_too_small():
    ims = _images("crack", [0.6] * 5, [0.5] * 5)
    assert matched_precision(ims, "crack", 0.9) is None


def test_rdd_compare_gain_and_pass():
    old = _images("pothole", [0.6] * 30, [0.5] * 10, n_gt_extra=30) + _images("crack", [0.6] * 30, [], 30)
    # 새 모델: 같은 정밀도(0.75)에서 정답 60개 중 45개 → 재현율 0.5 → 0.75
    new = _images("pothole", [0.6] * 45, [0.5] * 15, n_gt_extra=15) + _images("crack", [0.6] * 33, [], 27)
    conf = {"pothole": 0.31, "crack": 0.42}
    r = rdd_compare(old, new, conf, conf)
    assert r["pothole"]["old"]["R"] == 0.5 and r["pothole"]["new_same_precision"]["R"] == 0.75
    assert r["pothole"]["recall_gain"] == 0.25 and r["pothole"]["pass"]
    assert r["crack"]["recall_gain"] == 0.05 and r["crack"]["pass"]  # 경계 +5%p 는 통과


def _summary(found, pred, ok):
    row = {"scope": "all", "conf": "per_class:x.json", "defect_type": "crack", "n_gt": 10, "n_gt_found": found,
           "n_pred": pred, "n_pred_ok": ok, "precision": "", "recall": "", "missed_ids": ""}
    other = dict(row, scope="gn_b", n_gt_found=0)  # 영상별 행은 무시
    fixed = dict(row, conf="0.25", n_gt_found=9)  # 고정 기준 행도 무시
    return {"rows": [other, fixed, row]}


def test_demo_crack_gate():
    old = _summary(3, 6, 4)
    assert demo_compare(old, _summary(4, 7, 5))["crack_pass"]  # 더 찾고 헛것 2 그대로
    assert not demo_compare(old, _summary(5, 9, 6))["crack_pass"]  # 헛것 3 으로 늘어남
    assert not demo_compare(old, _summary(2, 3, 3))["crack_pass"]  # 덜 찾음


def _bench(ms_by_threads):
    rows = [{"stage": "lane_wear", "threads": 1, "width": 640, "height": 360, "median_ms": 6, "rt_i10": 9.0}]
    for th, ms in ms_by_threads.items():
        for w, h in ((640, 360), (1280, 720)):
            rows.append({"stage": "detect", "threads": th, "width": w, "height": h, "median_ms": ms,
                         "rt_i10": round(ms * 6 / 1000, 3)})
    return {"video_fps": 60.0, "rows": rows}


def test_cpu_gate_ignores_lane_wear_and_needs_1thread_realtime():
    assert bench_compare(_bench({1: 21.7}), _bench({1: 60.0, 2: 40.0}))["pass"]  # 60 ms × 6 = 0.36
    assert not bench_compare(_bench({1: 21.7}), _bench({1: 170.0}))["pass"]  # 1.02


def test_conf_cli_writes_class_conf(tmp_path):
    th = tmp_path / "conf_thresholds_v4.json"
    th.write_text(json.dumps(_thresholds({"pothole": 0.3, "crack": 0.28, "guardrail": 0.2, "sign": 0.25}, 0.45)))
    out = tmp_path / "class_conf_v4.json"
    assert model_gate.main(["conf", str(th), "--out", str(out)]) == 0
    from daboyeo_ai.inference import load_class_conf

    assert load_class_conf(out) == {"pothole": 0.3, "crack": 0.45, "guardrail": 0.2, "sign": 0.25}
