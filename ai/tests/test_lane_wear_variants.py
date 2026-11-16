import numpy as np
import pytest
from daboyeo_ai.eval.lane_wear_variants import MotionProbe, VariantTracker, crosswalk_drop, road_motion, road_small
from daboyeo_ai.lane_wear import LaneWearResult

W, H = 640, 360


def _seg_at(x_bottom, y0=250, y1=340, lean=0.0):
    return (x_bottom + lean * (y0 - y1), y0, x_bottom, y1)


def test_crosswalk_drop_dense_stripes_but_not_lanes():
    lanes = [_seg_at(60, lean=-0.5), _seg_at(580, lean=0.5)]           # 차로 폭만큼 떨어진 두 선
    stripes = [_seg_at(x) for x in (260, 300, 340, 380)]               # 차선 폭 상한(약 18px)의 2~3배 간격
    assert crosswalk_drop(lanes, W, H) == set()
    drop = crosswalk_drop(lanes + stripes, W, H)
    assert drop == {2, 3, 4, 5}
    # 같은 선의 조각 두 개는 한 선으로 센다 → 서로 다른 선 2개뿐이라 거르지 않음
    assert crosswalk_drop([_seg_at(300), _seg_at(303), _seg_at(345)], W, H) == set()


def test_motion_probe_still_vs_moving():
    rng = np.random.default_rng(0)
    still = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    p = MotionProbe(lag_s=0.5)
    assert p.update(still, 0.0) is None
    assert p.update(still, 0.3) is None
    assert p.update(still, 0.6) == 0.0
    moved = np.roll(still, 40, axis=0)
    assert p.update(moved, 1.2) > 10
    assert road_motion(road_small(still), road_small(still)) == 0.0


def _r(p25, paint=1.0, found=True):
    return LaneWearResult(lane_found=found, contrast=p25 if found else None, is_worn=False, confidence=0.0,
                          lane_pixel_ratio=0.0, n_segments=1, contrast_p25=p25 if found else None,
                          paint_len=paint if found else 0.0)


def test_variant_modes_ignore_short_paint_on_clear_lane():
    for mode, worn in (("p25_x_len", True), ("p25_only", False), ("p25_x_rel_len", True),
                       ("p25_x_rel_len_cap", True), ("p25_x_sqrt_rel_len", True)):
        tr = VariantTracker(mode=mode, warmup=5)
        for _ in range(10):
            tr.update(_r(0.5, 2.0))
        j = tr.update(_r(0.5, 0.8))   # 대비는 평소 그대로, 찾은 길이만 0.4배
        assert j.is_worn == worn, mode
        if mode == "p25_x_sqrt_rel_len":
            assert j.score == pytest.approx(np.sqrt(0.4))


def test_min_found_guard_blocks_judging_on_sparse_drive():
    tr = VariantTracker(min_found=0.3, warmup=5)
    for i in range(100):
        tr.update(_r(0.5, 1.0, found=(i % 10 == 0)))   # 10% 만 찾음
    j = tr.update(_r(0.1, 0.2))
    assert not j.is_worn and j.score is None and not j.confirmed
    with pytest.raises(ValueError):
        VariantTracker(mode="nope")
