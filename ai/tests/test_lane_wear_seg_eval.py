from daboyeo_ai.eval.lane_wear_seg import bin_key, seg_view, tally


def test_bin_key_two_second_cells():
    assert bin_key("c", 0.0) == "c/0"
    assert bin_key("c", 137.1) == "c/136"
    assert bin_key("c", 137.99) == bin_key("c", 136.0)
    assert bin_key("c", 138.0) == "c/138"


def test_tally_counts_and_fp_rate():
    review = {"c/136": "real", "c/50": "fp", "c/52": "amb"}
    r = tally([137.0, 50.5, 51.0, 52.1, 300.0], "c", review)
    assert (r["real"], r["amb"], r["fp"], r["unlabeled"]) == (1, 1, 2, 1)
    assert r["fp_rate"] == 0.5
    assert tally([], "c", review)["fp_rate"] is None


def test_seg_view_overrides_measurement():
    rec = {"t": 1.0, "lane_found": False, "xw": {"seg": {"lane_found": True, "contrast_p25": 0.2}}}
    v = seg_view(rec, "seg")
    assert v["lane_found"] is True and v["t"] == 1.0
    assert seg_view({"t": 2.0, "q_ok": False}, "seg") == {"t": 2.0, "q_ok": False}
