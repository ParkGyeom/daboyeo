import pytest
from daboyeo_ai.eval.ab_overlap import (
    CLIP_START_S,
    CLIPS,
    Track,
    haversine_m,
    locate,
    nearest_counterpart,
    observations,
    pair_events,
    server_group,
    source_video,
    to_source_s,
)

M_LAT = 1 / 111_320  # 북쪽 1 m 의 위도


def north_track(stop=False):
    """0~10초 북쪽으로 10 m/s. stop 이면 4~6초 멈춤."""
    rows, lat = [], 37.5
    for i in range(11):
        rows.append((float(i), lat, 127.0))
        if not (stop and 4 <= i < 6):
            lat += 10 * M_LAT
    return Track(rows)


def test_track_interp_ahead_and_extrap():
    tr = north_track()
    f = tr.fix(2.5)
    assert haversine_m(37.5, 127.0, f.lat, f.lon) == pytest.approx(25, abs=0.1) and not f.extrap
    f = tr.fix(2.5, ahead_m=5)
    assert haversine_m(37.5, 127.0, f.lat, f.lon) == pytest.approx(30, abs=0.1)
    f = tr.fix(12)  # 끝 2초 뒤: 끝 구간 속도로 늘인다
    assert f.extrap and haversine_m(37.5, 127.0, f.lat, f.lon) == pytest.approx(120, abs=0.5)
    assert tr.fix(30) is None


def test_track_stop_keeps_position():
    tr = north_track(stop=True)
    a, b = tr.fix(4.2), tr.fix(5.8)
    assert haversine_m(a.lat, a.lon, b.lat, b.lon) < 0.01
    f = tr.fix(4.0, ahead_m=5)  # 멈춘 곳에서 앞으로 5 m
    assert haversine_m(37.5, 127.0, f.lat, f.lon) == pytest.approx(45, abs=0.1)


def test_pairs_same_type_by_distance():
    tr = north_track()
    a = locate([{"id": "A1", "type": "crack", "t0": 1, "t1": 3}, {"id": "A2", "type": "pothole", "t0": 8, "t1": 8}], tr)
    b = locate([{"id": "B1", "type": "crack", "t0": 3, "t1": 3}, {"id": "B2", "type": "crack", "t0": 8, "t1": 8}], tr)
    p = pair_events(a, b, max_m=50)
    assert [(x["a"], x["b"], x["dist_m"]) for x in p] == [("A1", "B1", 10.0)]   # A2 는 유형이 다르다
    assert len(pair_events(a, b, max_m=59, same_type=False)) == 3   # A2-B2 0, A1-B1 10, A2-B1 50 (A1-B2 60 은 밖)
    assert nearest_counterpart(a, b) == {"A1": {"id": "B1", "dist_m": 10.0}, "A2": None}


def obs(dev, t, north_m, typ="crack"):
    return {"device": dev, "type": typ, "t": t, "conf": 0.5, "lat": 37.5 + north_m * M_LAT, "lon": 127.0}


def test_server_group_like_dedup():
    d = server_group([obs("A", 0, 0), obs("B", 0, 15), obs("B", 1, 60), obs("A", 2, 70, "pothole")], 20)
    assert [(x["type"], x["devices"]) for x in d] == [("crack", ["A", "B"]), ("crack", ["B"]), ("pothole", ["A"])]
    assert haversine_m(37.5, 127.0, d[0]["lat"], d[0]["lon"]) == pytest.approx(7.5, abs=0.1)  # 위치 = 관측 평균
    assert len(server_group([obs("A", 0, 0), obs("B", 0, 25)], 20)) == 2


def test_observations_threshold_merge_and_gps_span():
    tr = north_track()
    raw = {"1.0": [(1, 0.5, 100, 500, 200, 540), (1, 0.3, 900, 500, 950, 540)],   # 두 번째는 기준 미만
           "1.5": [(1, 0.6, 100, 520, 210, 570)],                                  # 같은 트랙 (아래로 커짐)
           "20.0": [(1, 0.9, 100, 500, 200, 540)]}                                 # GPS 밖
    thr = {"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}
    o = observations(raw, thr, tr, "A")
    assert len(o) == 1 and o[0]["t"] == 1.5 and o[0]["frames"] == 2 and o[0]["conf"] == 0.6
    assert len(observations(raw, thr, tr, "A", merge=False)) == 2


def test_clip_start_is_keyframe_before_name():
    """이름의 2050·3320 은 분:초. 실제 시작은 앞 키프레임이라 이름 시각보다 0~3초 앞 (docs/plan.md 0.8)."""
    assert set(CLIP_START_S) == set(CLIPS.values())
    for clip, start in CLIP_START_S.items():
        mmss = clip.split("_")[2].split("-")[0]
        named = int(mmss[:2]) * 60 + int(mmss[2:])
        assert 0 <= named - start < 3
    assert CLIP_START_S["gn_a_2050-2510"] == 1247.646 and CLIP_START_S["gn_b_3320-3620"] == 1998.0
    assert to_source_s("gn_a_2050-2510", 71.5) == 1319.146
    assert source_video("gn_b_3320-3620") == "gn_b"


def test_clip_track_from_anchors_uses_clip_start():
    """기준점 → 클립 경로: 원본 시각 T 의 위치 = 클립 시각 T - CLIP_START_S 의 위치."""
    from tools.gps.reconstruct import default_paths, reconstruct

    clip = CLIPS["A"]
    tr = Track.for_clip(clip)
    src, _, _ = reconstruct(*default_paths("gn_a"), "gn_a")
    p = src.points[200]
    f = tr.fix(p.t_s - CLIP_START_S[clip])
    assert haversine_m(p.lat, p.lon, f.lat, f.lon) < 0.5
