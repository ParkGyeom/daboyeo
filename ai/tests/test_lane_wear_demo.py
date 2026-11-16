import numpy as np
from daboyeo_ai.eval.lane_wear_demo import (
    clip_stats,
    cluster,
    confirmed_times,
    grid,
    label_of,
    parse_stopped,
    record,
    replay,
    sweep,
    to_result,
    worn_runs,
)
from daboyeo_ai.lane_wear import LaneWearResult


def _rec(i, p25, paint=1.0, found=True, q_ok=True, dt=0.05):
    r = LaneWearResult(lane_found=found, contrast=p25 if found else None, is_worn=False, confidence=0.0,
                       lane_pixel_ratio=0.01, n_segments=2 if found else 0,
                       contrast_p25=p25 if found else None, paint_len=paint if found else 0.0)
    return record(i, i * dt, q_ok, "" if q_ok else "dark", r if q_ok else None, [(0, 0, 1, 1)] if found else [])


def _drive(worn_at: set[int], n=60):
    # 기준 p25 0.5, paint 1.2 → score 1.2 (선명). 마모 프레임은 p25 0.25 → score 0.6
    return [_rec(i, 0.25 if i in worn_at else 0.5, 1.2) for i in range(n)]


def test_record_roundtrip_to_result():
    rec = _rec(3, 0.4, 0.9)
    r = to_result(rec)
    assert r.lane_found and r.contrast_p25 == 0.4 and r.paint_len == 0.9 and rec["segs"] == [[0, 0, 1, 1]]
    assert not to_result(_rec(4, 0.4, q_ok=False)).lane_found


def test_replay_skips_quality_fail_and_confirms_once_per_run():
    recs = _drive(set(range(30, 36)))
    recs[31]["q_ok"] = False                      # 품질 탈락 프레임은 tracker 에 안 들어감
    j = replay(recs)
    assert len(j) == 59 and all(x["i"] != 31 for x in j)
    assert confirmed_times(j) == [recs[33]["t"]]   # 30·32·33 → 3/5
    assert len(confirmed_times(replay(recs, k=5, n=5))) == 1   # 30,32,33,34,35 = 5개 연속 판정
    assert confirmed_times(replay(recs, threshold=0.5)) == []  # score 0.6 > 0.5


def test_worn_runs_and_cluster_and_label():
    j = replay(_drive(set(range(30, 36))))
    runs = worn_runs(j)
    assert len(runs) == 1 and runs[0][0] == 32 * 0.05
    cl = cluster([1.0, 1.5, 2.9, 10.0], gap=1.5)
    assert cl == [(1.0, 2.9), (10.0, 10.0)]
    assert label_of(2.0, cl, {0: "fp"}) == "fp" and label_of(10.0, cl, {0: "fp"}) is None and label_of(5, cl, {}) is None


def test_clip_stats_and_sweep_counts_labels():
    recs = _drive({30, 31, 32, 80, 81, 82}, n=100) + [_rec(100, 0.5, found=False)]
    j = replay(recs)
    s = clip_stats(recs, j)
    assert s["frames"] == 101 and s["lane_found_of_pass"] == round(100 / 101, 3) and s["judged"] == 80
    assert s["frame_worn_rate"] == round(6 / 80, 3)
    ts = confirmed_times(j)
    cl = cluster(ts)
    rows = sweep(recs, cl, {0: "real", 1: "fp"}, duration_s=60, thresholds=(1.0,), votes=((3, 5),))
    assert rows[0]["confirmed"] == 2 and rows[0]["real"] == 1 and rows[0]["fp"] == 1 and rows[0]["fp_rate"] == 0.5


def test_grid_shape():
    g = grid([np.zeros((10, 20, 3), np.uint8)] * 4, cols=3)
    assert g.shape == (20, 60, 3)


def test_parse_stopped():
    assert parse_stopped(["c1:30-96,152-172"]) == {"c1": [(30.0, 96.0), (152.0, 172.0)]}
