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


def test_window_rows_restarts_tracker_inside_window():
    from daboyeo_ai.eval.lane_wear_seg import window_rows

    def rec(t, p25):
        return {"i": int(t * 20), "t": t, "q_ok": True, "q_reason": "", "lane_found": True, "contrast": 0.5,
                "contrast_p25": p25, "paint_len": 1.0, "n_segments": 2, "lane_pixel_ratio": 0.01}
    # 0~10초 선명(높은 대비) 뒤 10~12초 흐림: 전체로 돌리면 흐린 구간이 기준보다 낮아 확정, 흐림만 잘라 돌리면 기준이 흐림이라 확정 없음
    recs = [rec(i / 20, 1.0) for i in range(200)] + [rec(10 + i / 20, 0.2) for i in range(40)]
    full = window_rows("c", recs, {}, 0, 12, variants=("base (17장 지금)",), thresholds=(1.0,))
    cut = window_rows("c", recs, {}, 10, 12, variants=("base (17장 지금)",), thresholds=(1.0,))
    assert full[0]["confirmed"] >= 1 and full[0]["unlabeled"] == full[0]["confirmed"]
    assert cut[0]["confirmed"] == 0 and cut[0]["window"] == [10, 12]
