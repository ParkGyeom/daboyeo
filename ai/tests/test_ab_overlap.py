import pytest
from daboyeo_ai.eval.ab_overlap import (
    Track,
    haversine_m,
    locate,
    nearest_counterpart,
    observations,
    pair_events,
    server_group,
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
