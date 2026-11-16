import numpy as np

from daboyeo_ai.eval.demo_clip import band_counts, conf_quantiles, crop_tile, decide, one_sided, per_minute, review_counts

THR = [0.27, 0.29, 0.26, 0.24]


def test_per_minute_and_conf_use_threshold():
    frames = {"0.0": [(1, 0.5, 0, 0, 10, 10), (1, 0.1, 0, 0, 5, 5)], "0.5": [(3, 0.9, 0, 0, 9, 9)]}
    pm = per_minute(frames, THR, minutes=0.5)
    assert pm["crack"] == 2.0 and pm["sign"] == 2.0 and pm["pothole"] == 0
    q = conf_quantiles(frames, THR)
    assert q["crack"]["n"] == 1 and q["crack"]["p50"] == 0.5 and q["pothole"] == {"n": 0}


def test_band_counts_by_box_center():
    frames = {"0": [(1, 0.5, 0, 980, 50, 1040), (1, 0.5, 0, 500, 50, 600), (3, 0.3, 0, 900, 20, 1000)]}
    b = band_counts(frames, THR, frame_h=1080)
    assert b["8%"]["crack"] == 1 and b["8%"]["sign"] == 0     # 중심 1010 >= 993.6, 950 < 993.6
    assert b["15%"]["sign"] == 1                              # 950 >= 918


def test_one_sided_respects_both_thresholds():
    a = {"t": [(1, 0.5, 0, 0, 10, 10), (2, 0.5, 100, 100, 120, 120)]}
    b = {"t": [(1, 0.6, 0, 0, 10, 10), (2, 0.1, 100, 100, 120, 120)]}   # b 의 가드레일은 기준 미만
    assert one_sided(a, b, THR, THR)["t"] == [(2, 0.5, 100, 100, 120, 120)]


def _crops(spec):
    out = {}
    for i, (kind, cls, lab) in enumerate(spec):
        out[f"c{i}"] = ({"clip": "gn_a", "kind": kind, "cls": cls}, lab)
    return {k: v[0] for k, v in out.items()}, {k: v[1] for k, v in out.items() if v[1]}


def test_decide_rules():
    crops, review = _crops([("v3_only", "crack", "real"), ("v3_only", "crack", "real"), ("v3_only", "crack", "fp")])
    assert decide(review_counts(crops, review), frames=100)["adopt_v3"] is True
    crops, review = _crops([("v3_only", "crack", "real"), ("v3_only", "crack", "fp"), ("v3_only", "crack", "fp")])
    assert decide(review_counts(crops, review), frames=100)["adopt_v3"] is False      # FP 가 real 보다 더 늘어남
    crops, review = _crops([("v3_only", "crack", "real")] * 1 + [("v2_only", "sign", "real")] * 4)
    d = decide(review_counts(crops, review), frames=100)
    assert d["checks"]["other_real_net_loss_per_frame"]["value"] == 0.04 and d["adopt_v3"] is False
    crops, review = _crops([("v2_only", "crack", "amb")])
    assert decide(review_counts(crops, review), frames=10)["adopt_v3"] is False      # 균열 real 이득 0 → 바꿀 이유 없음


def test_review_counts_marks_unreviewed():
    crops = {"x": {"clip": "gn_b", "kind": "v2_only", "cls": "sign"}}
    assert review_counts(crops, {})["gn_b"]["v2_only"]["sign"] == {"unreviewed": 1}


def test_crop_tile_shape():
    img = np.zeros((1080, 1920, 3), np.uint8)
    t = crop_tile(img, (1, 0.5, 1900, 1060, 1920, 1080), "x")
    assert t.shape[0] == 220 + 26 and t.shape[2] == 3


def test_region_counts_top_caption():
    from daboyeo_ai.eval.demo_clip import TOP_CAPTION, region_counts

    frames = {"0": [(3, 0.5, 709, 47, 1178, 189), (3, 0.5, 100, 47, 300, 189)]}
    assert region_counts(frames, THR, 1920, 1080, TOP_CAPTION)["sign"] == 1


def test_review_at_threshold():
    from daboyeo_ai.eval.demo_clip import review_at

    crops = {"a": {"kind": "v3_only", "cls": "crack", "conf": 0.5}, "b": {"kind": "v3_only", "cls": "crack", "conf": 0.35},
             "c": {"kind": "v2_only", "cls": "crack", "conf": 0.9}}
    assert review_at(crops, {"a": "real", "b": "fp"}, "v3_only", "crack", 0.42) == {"real": 1}
