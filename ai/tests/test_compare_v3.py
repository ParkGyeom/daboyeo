from daboyeo_ai.eval.compare_v3 import go_decision, sample_stats
from daboyeo_ai.eval.recall_res import MUSTKEEP

THR = [0.27, 0.29, 0.26, 0.24]


def test_sample_stats_counts_new_and_lost_above_threshold():
    old = {"a": [(3, 0.9, 0, 0, 10, 10)], "b": []}
    new = {"a": [(3, 0.9, 0, 0, 10, 10), (1, 0.5, 50, 50, 90, 60), (1, 0.1, 0, 0, 5, 5)], "b": [(2, 0.2, 0, 0, 9, 9)]}
    s = sample_stats(new, old, THR)
    assert s["new_vs_old_per_frame"]["crack"] == 0.5          # 0.1 짜리는 기준 미만이라 안 셈
    assert s["dets_per_frame"]["sign"] == 0.5 and s["lost_vs_old_per_frame"]["sign"] == 0
    assert s["frames_with_new"] == 1


def _summary(hits: int, new_c: float, lost_s: float) -> dict:
    cracks = [k for k, v in MUSTKEEP.items() if v == "crack"]
    mk = {k: {"class": v, "target_hit": k in cracks[:hits]} for k, v in MUSTKEEP.items()}
    z = {"pothole": 0, "crack": 0, "guardrail": 0, "sign": 0}
    return {"variants": {"base": {"new": {"mustkeep": mk}, "sample": {
        "new_vs_old_per_frame": z | {"crack": new_c}, "lost_vs_old_per_frame": z | {"sign": lost_s}}}}}


def test_go_needs_rdd_map_and_all_checks():
    assert go_decision(_summary(2, 0.05, 0.0))["go"] is None           # RDD mAP 없음 → 판정 보류
    card = {"eval_by_source": {"rdd2022_test": {"v2": {"crack": 0.584, "pothole": 0.418},
                                                "v3": {"crack": 0.570, "pothole": 0.410}}}}
    assert go_decision(_summary(2, 0.05, 0.0), card)["go"] is True
    assert go_decision(_summary(1, 0.05, 0.0), card)["go"] is False      # 균열 1/3
    assert go_decision(_summary(3, 0.2, 0.0), card)["go"] is False       # 새 탐지 폭증
    worse = {"eval_by_source": {"rdd2022_test": {"v2": {"crack": 0.584, "pothole": 0.418},
                                                 "v3": {"crack": 0.55, "pothole": 0.418}}}}
    assert go_decision(_summary(3, 0.0, 0.0), worse)["checks"]["rdd2022_test_crack_mAP50_drop"]["ok"] is False
