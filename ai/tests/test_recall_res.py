import pytest

from daboyeo_ai.eval.recall_res import condition, grid_windows, new_vs_base, variants


def test_condition_from_stem():
    assert condition("0401_light_rainy_smooth_03000136") == "light_rainy_smooth"


def test_grid_windows_cover_frame():
    g = grid_windows(2, 2)
    assert len(g) == 4
    assert min(w[0] for w in g) == 0 and max(w[2] for w in g) == 1
    assert min(w[1] for w in g) == 0 and max(w[3] for w in g) == 1
    assert all(w[0] < w[2] and w[1] < w[3] for w in g)


def test_new_vs_base_same_class_iou():
    base = [(0, 0.5, 0, 0, 100, 100)]
    dets = [(0, 0.6, 5, 5, 105, 105), (1, 0.6, 0, 0, 100, 100), (0, 0.4, 500, 500, 600, 600)]
    assert new_vs_base(dets, base) == dets[1:]


def test_variants_runs():
    v = variants()
    assert v["base"] is None and len(v["base+roi_t2"]) == 3 and v["base+roi_t2"][0] == (0.0, 0.0, 1.0, 1.0)
    assert v["roi"] == ((0.0, 0.45, 1.0, 1.0),)
    with pytest.raises(KeyError):
        v["nope"]
