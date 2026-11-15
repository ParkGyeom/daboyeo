import pytest

from daboyeo_ai.eval.verify_eval import baseline_fp_removed, build_report, pick_tau, sweep
from daboyeo_ai.eval.verify_mustkeep import seq_and_index


def test_baseline_raises_conf_to_keep_n_tp():
    conf = [0.9, 0.8, 0.5, 0.4, 0.3, 0.35]
    tp = [True, False, True, False, True, False]
    # TP 2개를 남기려면 기준 = 0.5 → 0.4, 0.35 FP 가 지워진다 (0.8 FP 는 남음)
    assert baseline_fp_removed(conf, tp, 2) == (2, 0.5)
    assert baseline_fp_removed(conf, tp, 3) == (0, 0.3)
    assert baseline_fp_removed(conf, tp, 0) == (3, None)
    with pytest.raises(ValueError):
        baseline_fp_removed(conf, tp, 4)


def test_sweep_compares_verifier_with_baseline():
    conf = [0.9, 0.8, 0.5, 0.4, 0.3, 0.35]
    tp = [True, False, True, False, True, False]
    v = [0.8, 0.0, 0.6, 0.1, 0.5, 0.05]  # 검증기가 FP 를 모두 낮게 → 완벽
    rows = sweep(conf, tp, v, [0.0, 0.3])
    r0, r1 = rows
    assert (r0["tp_kept"], r0["fp_removed"]) == (3, 0)
    assert (r1["tp_kept"], r1["fp_removed"], r1["precision_after"]) == (3, 3, 1.0)
    # 같은 TP 3개를 남기는 신뢰도 기준(0.3)은 FP 를 하나도 못 지운다 → 이득 3
    assert (r1["baseline_fp_removed"], r1["gain_fp"]) == (0, 3)
    assert pick_tau(rows)["tau"] == 0.3


def test_build_report_band_and_overall():
    recs = [
        {"class": "pothole", "conf": 0.9, "tp": 1, "v": 0.9, "v_any": 0.9, "ms": 10.0},
        {"class": "pothole", "conf": 0.3, "tp": 1, "v": 0.5, "v_any": 0.5, "ms": 12.0},
        {"class": "pothole", "conf": 0.4, "tp": 0, "v": 0.0, "v_any": 0.2, "ms": 11.0},
        {"class": "sign", "conf": 0.7, "tp": 0, "v": 0.6, "v_any": 0.6, "ms": 9.0},
    ]
    rep, rows = build_report(recs, {"pothole": 0.27}, "m.onnx", None, [0.0, 0.1, 0.5])
    band = rep["classes"]["pothole"]["band"]
    assert (band["n_tp"], band["n_fp"]) == (1, 1)  # conf 0.9 는 구간 밖
    assert band["pick_retention>=0.9"]["tau"] == 0.1
    assert rep["overall"]["all"]["fp_removed"] == 1
    assert rep["ms_per_crop"]["n"] == 4
    assert {r["subset"] for r in rows} == {"all", "band"}


def test_seq_and_index():
    assert seq_and_index("0401_light_rainy_smooth_03000136") == ("0401_light_rainy_smooth_03", 136)


def test_baseline_tp_kept_and_fp40():
    from daboyeo_ai.eval.verify_eval import at_fp_removal, baseline_tp_kept

    conf = [0.9, 0.8, 0.5, 0.4, 0.3, 0.35]
    tp = [True, False, True, False, True, False]
    assert baseline_tp_kept(conf, tp, 2) == 2  # FP 0.35·0.4 를 지우면 기준 > 0.4 → TP 0.3 탈락
    assert baseline_tp_kept(conf, tp, 0) == 3
    v = [0.8, 0.0, 0.6, 0.1, 0.5, 0.05]
    rows = sweep(conf, tp, v, [0.0, 0.07, 0.3])
    r = at_fp_removal(conf, tp, v, rows)
    # τ 0.07 은 FP 2개(0.0, 0.05) 를 지우고 TP 는 모두 남긴다; 신뢰도로 FP 2개를 지우면 TP 0.3 이 빠진다
    assert r["tau"] == 0.07 and r["tp_retention"] == 1.0 and r["baseline_tp_retention"] == pytest.approx(2 / 3)
