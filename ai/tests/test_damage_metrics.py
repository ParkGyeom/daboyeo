import pytest

from daboyeo_ai.eval.damage_metrics import binary_report, pick_threshold, report_by_group


def test_pick_threshold_max_f1_prefers_higher_on_tie():
    y = [0, 0, 1, 0, 1, 1]
    p = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
    t, f1 = pick_threshold(y, p)
    # t=0.3: TP3 FP1 → F1 6/7 ; t=0.8: TP2 FP0 → 0.8 ; t=0.1: TP3 FP3 → 2/3
    assert t == pytest.approx(0.3) and f1 == pytest.approx(6 / 7)

    t, f1 = pick_threshold([1, 1, 0], [0.9, 0.9, 0.1])  # 0.9 와 그 사이 아무 값이나 F1=1 → 높은 쪽
    assert t == pytest.approx(0.9) and f1 == 1.0


def test_pick_threshold_without_positives():
    assert pick_threshold([0, 0], [0.2, 0.7]) == (0.5, 0.0)


def test_binary_report_counts_and_rates():
    r = binary_report([1, 1, 0, 0, 0], [0.9, 0.2, 0.8, 0.1, 0.3], 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 2)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5
    assert r["fpr_normal"] == pytest.approx(0.3333, abs=1e-4)
    assert (r["n_damaged"], r["n_normal"]) == (2, 3)


def test_binary_report_empty_denominators_are_none():
    r = binary_report([0, 0], [0.1, 0.2], 0.5)
    assert r["precision"] is None and r["recall"] is None and r["f1"] is None and r["fpr_normal"] == 0.0


def test_report_by_group():
    out = report_by_group([1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1], 0.5, ["guardrail", "guardrail", "sign", None])
    assert set(out) == {"all", "guardrail", "sign", "unknown"}
    assert out["guardrail"]["tp"] == 1 and out["guardrail"]["fp"] == 1
    assert out["sign"]["fn"] == 1 and out["unknown"]["tn"] == 1
    assert out["all"]["n_damaged"] == 2


def test_score_folder_and_compare_probs(tmp_path):
    import cv2
    import numpy as np

    from daboyeo_ai.eval.damage_metrics import compare_probs, score_folder

    for label, val in (("damaged", 255), ("normal", 0)):
        d = tmp_path / "val" / label
        d.mkdir(parents=True)
        cv2.imwrite(str(d / f"krid_{label}_0.png"), np.full((8, 8, 3), val, np.uint8))
    (tmp_path / "val" / "normal" / "notes.txt").write_text("skip")
    names, y, p = score_folder(tmp_path, "val", lambda img: img.mean() / 255)
    assert names == ["krid_damaged_0", "krid_normal_0"]
    assert y.tolist() == [1, 0] and p.tolist() == [1.0, 0.0]

    r = compare_probs([0.9, 0.1, 0.49], [0.9, 0.1, 0.51], threshold=0.5)
    assert r["n"] == 3 and r["max_abs_diff"] == 0.02 and r["label_agree"] == 0.6667
