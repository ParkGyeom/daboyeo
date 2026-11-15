"""conf_thresholds — 가짜 박스로 매칭·곡선·추천 규칙과 zip → 리포트 흐름을 시험한다."""

import json
import zipfile

import cv2
import numpy as np
import pytest

from daboyeo_ai.eval.conf_thresholds import (
    ClassStats,
    build_report,
    evaluate_part,
    extract_test,
    iou_matrix,
    match_image,
    pr_curve,
    read_labels,
    recommend,
)

GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]


def test_iou_matrix():
    m = iou_matrix([[0, 0, 10, 10], [5, 5, 15, 15]], [[0, 0, 10, 10], [20, 20, 30, 30]])
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == 0
    assert m[1, 0] == pytest.approx(25 / 175)


def test_match_greedy_by_score_and_one_gt_once():
    gt = [[0, 0, 10, 10]]
    preds = [(0.3, [0, 0, 10, 10]), (0.9, [1, 1, 10, 10])]  # 둘 다 겹치지만 점수 높은 쪽이 먼저 가져간다
    assert match_image(preds, gt) == [(0.9, True), (0.3, False)]
    assert match_image(preds, []) == [(0.9, False), (0.3, False)]
    assert match_image([(0.5, [50, 50, 60, 60])], gt) == [(0.5, False)]  # IoU 0
    assert match_image([], gt) == []


def test_match_iou_threshold():
    gt = [[0, 0, 10, 10]]
    half = [(0.8, [0, 0, 10, 5])]  # IoU 0.5
    assert match_image(half, gt, iou_thr=0.5) == [(0.8, True)]
    assert match_image(half, gt, iou_thr=0.6) == [(0.8, False)]


def test_pr_curve_counts():
    scored = [(0.9, True), (0.8, True), (0.6, False), (0.4, True), (0.2, False)]
    c = {r["conf"]: r for r in pr_curve(scored, n_gt=4, grid=[0.1, 0.5, 0.85, 0.95])}
    assert (c[0.1]["tp"], c[0.1]["fp"]) == (3, 2) and c[0.1]["P"] == pytest.approx(0.6) and c[0.1]["R"] == 0.75
    assert c[0.5]["P"] == pytest.approx(2 / 3) and c[0.5]["R"] == 0.5
    assert c[0.85]["P"] == 1.0 and c[0.85]["R"] == 0.25
    assert c[0.95]["P"] is None and c[0.95]["F1"] is None
    assert c[0.5]["F1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))


def synthetic(n=200, seed=0):
    """점수가 높을수록 TP 일 확률이 높은 탐지 n 개 + GT n 개."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.05, 1, n)
    return [(float(x), bool(rng.uniform() < x)) for x in s], n


def test_recommend_lowest_conf_reaching_precision():
    scored, n_gt = synthetic()
    curve = pr_curve(scored, n_gt, GRID)
    rec = recommend(curve, target_precision=0.7, min_support=10)
    assert rec["rule"] == "precision>=0.7" and rec["pick"]["P"] >= 0.7
    lower = [r for r in curve if r["conf"] < rec["conf"]]
    assert all(r["P"] < 0.7 for r in lower)  # 그보다 낮은 기준은 목표 미달
    assert rec["at_default"]["conf"] == 0.25
    assert rec["max_f1"]["F1"] == max(r["F1"] for r in curve if r["tp"] + r["fp"] >= 10)


def test_recommend_falls_back_to_max_f1_then_default():
    scored = [(s, i % 3 == 0) for i, s in enumerate(np.linspace(0.06, 0.94, 90))]  # 정밀도 ~1/3 고정
    curve = pr_curve(scored, 60, GRID)
    rec = recommend(curve, target_precision=0.7, min_support=5)
    assert rec["rule"] == "max_f1" and rec["precision_target"] is None
    assert rec["conf"] == rec["max_f1"]["conf"]
    assert recommend([], 0.7)["rule"] == "default" and recommend([], 0.7)["conf"] == 0.25


def test_min_support_ignores_tiny_high_conf_tail():
    scored = [(0.9, True)] + [(0.3, False)] * 30 + [(0.3, True)] * 10
    curve = pr_curve(scored, 20, GRID)
    assert recommend(curve, 0.7, min_support=1)["conf"] == 0.35  # 1건짜리 P=1.0
    assert recommend(curve, 0.7, min_support=5)["rule"] == "max_f1"


def test_read_labels_to_pixels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.1\n\n3 0.1 0.1 0.2 0.2\n")
    (c1, b1), (c2, b2) = read_labels(p, 100, 200)
    assert (c1, c2) == (2, 3) and b1 == pytest.approx([40, 90, 60, 110]) and b2 == pytest.approx([0, 0, 20, 40])
    assert read_labels(tmp_path / "none.txt", 1, 1) == []


class FakeDetector:
    """이미지마다 GT 위치에 sign(3) 0.9 하나 + 엉뚱한 곳에 sign 0.3 하나 + pothole 0.8 하나."""

    def raw(self, img):
        return [(3, 0.9, 10, 10, 50, 50), (3, 0.3, 60, 60, 90, 90), (0, 0.8, 10, 10, 50, 50)]


def test_zip_to_report(tmp_path):
    zp = tmp_path / "krid.zip"
    _, buf = cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))
    with zipfile.ZipFile(zp, "w") as z:
        for i in range(30):
            z.writestr(f"det/images/test/k{i}.jpg", buf.tobytes())
            z.writestr(f"det/labels/test/k{i}.txt", "3 0.3 0.3 0.4 0.4\n")
        z.writestr("cls/test/normal/x.jpg", buf.tobytes())  # 안 푼다
    root = extract_test(zp, tmp_path / "x")
    assert not (tmp_path / "x" / "cls").exists()
    stats = {c: ClassStats() for c in ("pothole", "crack", "guardrail", "sign")}
    assert evaluate_part(FakeDetector(), root, "krid", ["guardrail", "sign"], stats, 0.5) == 30
    assert stats["sign"].n_gt == 30 and len(stats["sign"].scored) == 60
    assert stats["pothole"].images == 0  # krid 는 pothole 출처가 아니다
    report, rows = build_report(stats, GRID, 0.7, 20, 0.5, "fake.onnx")
    s = report["classes"]["sign"]
    assert report["class_conf"]["sign"] == s["conf"] == 0.35  # 0.3 짜리 FP 를 넘는 첫 기준 (P 0.5 → 1.0)
    assert s["P"] == 1.0 and s["R"] == 1.0 and s["at_0.25"]["P"] == 0.5
    assert report["classes"]["crack"]["rule"] == "default" and report["class_conf"]["crack"] == 0.25
    assert report["classes"]["crack"]["missing_sources"] == ["rdd2022"]
    assert set(report["class_conf"]) == {"pothole", "crack", "guardrail", "sign"}
    assert s["alternatives"]["precision>=0.9"]["conf"] == 0.35
    json.dumps(report)  # 직렬화 가능
    assert {r["class"] for r in rows} == {"guardrail", "sign"}
