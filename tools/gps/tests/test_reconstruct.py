"""GPS 경로 재구성·hold-out 오차 (docs/plan.md 0.7·0.8). 합성 직선·꺾인 경로로 기대값을 직접 계산해 맞춘다."""

from __future__ import annotations

import csv
import math

import pytest
from daboyeo_contracts import GpsTrack
from daboyeo_contracts.gps import EARTH_RADIUS_M, haversine_m

from tools.gps import holdout_error, preview_map, reconstruct
from tools.gps.reconstruct import (
    AnchorError,
    Route,
    load_anchors,
    parse_time,
    snap_anchors,
)

LAT0, LON0 = 37.4979, 127.0276
M_PER_DEG_LAT = math.radians(1) * EARTH_RADIUS_M


def ll(east_m: float, north_m: float) -> tuple[float, float]:
    """기준점에서 동쪽·북쪽으로 m 만큼 간 (lat, lon)."""
    return (LAT0 + north_m / M_PER_DEG_LAT,
            LON0 + east_m / (M_PER_DEG_LAT * math.cos(math.radians(LAT0))))


def write_gpx(path, pts_m) -> str:
    pts = "".join(f'<trkpt lat="{la:.8f}" lon="{lo:.8f}"/>' for la, lo in (ll(*p) for p in pts_m))
    path.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
        f"<trk><name>t</name><trkseg>{pts}</trkseg></trk></gpx>", encoding="utf-8")
    return str(path)


def write_anchors(path, rows) -> str:
    """rows: (t, (east_m, north_m) | None, memo)"""
    lines = ["video_id,t_s,lat,lon,memo"]
    for t, pos, memo in rows:
        lat, lon = ("", "") if pos is None else (f"{ll(*pos)[0]:.8f}", f"{ll(*pos)[1]:.8f}")
        lines.append(f"v,{t},{lat},{lon},{memo}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def dist_m(latlon, pos_m) -> float:
    return haversine_m(*latlon, *ll(*pos_m))


STRAIGHT = [(0, 0), (0, 1000)]  # 북쪽 1 km
BENT = [(0, 0), (0, 300), (400, 300)]  # 북쪽 300 m → 동쪽 400 m


def test_parse_time():
    assert parse_time("1250.5") == 1250.5
    assert parse_time("20:50.5") == 1250.5
    assert parse_time("0:20:50") == 1250
    with pytest.raises(ValueError):
        parse_time(" ")


def test_straight_route_distance_proportional(tmp_path):
    gpx = write_gpx(tmp_path / "r.gpx", STRAIGHT)
    anchors = write_anchors(tmp_path / "a.csv", [(0, (0, 0), "출발"), (100, (0, 1000), "도착")])
    track, route, _ = reconstruct.reconstruct(gpx, anchors, step_s=0.5)
    assert route.length_m == pytest.approx(1000, abs=0.5)
    assert len(track.points) == 201
    assert dist_m(track.at(50), (0, 500)) < 0.5
    assert dist_m(track.at(12.25), (0, 122.5)) < 0.5
    assert track.points[0].speed_kmh == pytest.approx(36.0, abs=0.1)


def test_bent_route_follows_the_road_not_the_chord(tmp_path):
    gpx = write_gpx(tmp_path / "r.gpx", BENT)
    anchors = write_anchors(tmp_path / "a.csv", [(0, (0, 0), ""), (70, (400, 300), "")])
    track, _, _ = reconstruct.reconstruct(gpx, anchors, step_s=0.5)
    assert dist_m(track.at(30), (0, 300)) < 0.5  # 700 m 중 300 m = 모퉁이
    assert dist_m(track.at(50), (200, 300)) < 0.5


def test_anchor_off_the_road_is_snapped(tmp_path):
    gpx = write_gpx(tmp_path / "r.gpx", STRAIGHT)
    anchors = write_anchors(tmp_path / "a.csv", [(0, (0, 0), ""), (10, (12, 400), "길가"), (20, (0, 1000), "")])
    route = Route(reconstruct.load_gpx(gpx))
    snapped = snap_anchors(route, load_anchors(anchors))
    assert snapped[1].s == pytest.approx(400, abs=0.5)
    assert snapped[1].snap_m == pytest.approx(12, abs=0.5)


def test_stop_anchor_without_position(tmp_path):
    gpx = write_gpx(tmp_path / "r.gpx", STRAIGHT)
    anchors = write_anchors(tmp_path / "a.csv", [
        (0, (0, 0), ""), (10, (0, 100), "신호 정지"), (40, None, "출발"), (50, (0, 200), "")])
    track, _, _ = reconstruct.reconstruct(gpx, anchors, step_s=1)
    assert dist_m(track.at(25), (0, 100)) < 0.5
    assert dist_m(track.at(45), (0, 150)) < 0.5
    assert {p.speed_kmh for p in track.points if 10 <= p.t_s < 40} == {0.0}


def test_loop_route_snaps_forward(tmp_path):
    # 사각형을 한 바퀴 돈 뒤 같은 길을 다시 지나 북쪽으로: (0,100) 은 처음엔 100 m, 두 번째엔 700 m 지점
    gpx = write_gpx(tmp_path / "r.gpx", [(0, 0), (0, 200), (100, 200), (100, 0), (0, 0), (0, 300)])
    anchors = write_anchors(tmp_path / "a.csv", [(0, (0, 50), ""), (40, (100, 100), ""), (70, (0, 100), "")])
    route = Route(reconstruct.load_gpx(gpx))
    assert [round(p.s) for p in snap_anchors(route, load_anchors(anchors))] == [50, 400, 700]


@pytest.mark.parametrize("rows, msg", [
    ([(0, (0, 0), ""), (20, (0, 500), ""), (10, (0, 800), "")], "시각"),  # 시간이 거꾸로
    ([(0, (0, 0), ""), (10, (0, 500), ""), (20, (0, 300), "")], "순서"),  # 경로를 거꾸로 감
    ([(0, (0, 0), ""), (10, (80, 500), ""), (20, (0, 900), "")], "떨어져"),  # 경로에서 80 m
    ([(0, None, ""), (10, (0, 500), "")], "첫 기준점"),
    ([(0, (0, 0), "")], "2개 이상"),
])
def test_bad_anchors_are_errors(tmp_path, rows, msg):
    gpx = write_gpx(tmp_path / "r.gpx", STRAIGHT)
    anchors = write_anchors(tmp_path / "a.csv", rows)
    with pytest.raises(AnchorError, match=msg):
        reconstruct.reconstruct(gpx, anchors)


def test_anchor_csv_filters_video_and_skips_comments(tmp_path):
    p = tmp_path / "a.csv"
    lat, lon = ll(0, 0)
    p.write_text(f"# 설명 줄\nvideo_id,t_s,lat,lon,memo\nv,0:10,{lat},{lon},a\nw,0:20,{lat},{lon},b\n,0:30,,,c\n",
                 encoding="utf-8")
    assert [a.t_s for a in load_anchors(p, "v")] == [10, 30]
    assert len(load_anchors(p)) == 3
    p.write_text(f"t_s,lat,lon,memo\n1,{lat},{lon},교차로, 정지선\n", encoding="utf-8")
    assert load_anchors(p)[0].memo == "교차로, 정지선"  # 따옴표 없는 쉼표


def test_holdout_zero_at_constant_speed(tmp_path):
    gpx = write_gpx(tmp_path / "r.gpx", STRAIGHT)
    anchors = write_anchors(tmp_path / "a.csv", [(t, (0, t * 10), "") for t in (0, 25, 50, 75, 100)])
    _, _, results = holdout_error.evaluate(gpx, anchors)
    assert len(results) == 3  # 첫·마지막은 빼지 않는다
    assert max(h.error_m for h in results) < 0.5


def test_holdout_error_is_straight_line_distance(tmp_path):
    # 가운데 기준점(북 100 m)을 빼면 t=10 은 700 m 의 절반 = 350 m → (동 50, 북 300). 거리 = √(50² + 200²)
    gpx = write_gpx(tmp_path / "r.gpx", BENT)
    anchors = write_anchors(tmp_path / "a.csv", [(0, (0, 0), ""), (10, (0, 100), "육교"), (20, (400, 300), "")])
    route, snapped, results = holdout_error.evaluate(gpx, anchors)
    assert results[0].error_m == pytest.approx(math.hypot(50, 200), abs=0.5)
    row = holdout_error.summarize("v", route, snapped, results)
    assert row["n_anchors"] == 3 and row["n_holdout"] == 1
    assert float(row["max_m"]) == pytest.approx(206.2, abs=0.5)
    assert row["max_at"] == "0:10.0 육교"


def test_cli_end_to_end(tmp_path):
    gpx = write_gpx(tmp_path / "r.gpx", BENT)
    anchors = write_anchors(tmp_path / "a.csv", [(t, None if t == 15 else (0, 0) if t == 0 else
                                                  (0, 100) if t == 10 else (0, 300) if t == 30 else (400, 300), "")
                                                 for t in (0, 10, 15, 30, 70)])
    out = tmp_path / "v.gps.csv"
    reconstruct.main(["--gpx", gpx, "--anchors", anchors, "--out", str(out), "--clip-start", "0:05"])
    track = GpsTrack.from_csv(out)  # 가상 블랙박스가 읽는 형식 그대로
    assert track.start_s == 0 and track.end_s == 65
    assert dist_m(track.at(5), (0, 100)) < 0.5  # 원본 10 초 = 클립 5 초

    err = tmp_path / "gps_error.csv"
    err.write_text("video_id,n_anchors,n_holdout,mean_m,median_m,max_m,max_at,route_m\nother,9,7,1.0,1.0,2.0,x,100\n"
                   "v,1,1,1,1,1,old,1\n", encoding="utf-8")
    holdout_error.main(["--gpx", gpx, "--anchors", anchors, "--name", "v", "--out", str(err)])
    rows = {r["video_id"]: r for r in csv.DictReader(err.open(encoding="utf-8"))}
    assert set(rows) == {"other", "v"} and rows["v"]["n_anchors"] == "5" and rows["v"]["n_holdout"] == "3"
    assert rows["v"]["n_moving"] == "1"  # t=10·15 는 멈춤 경계 (15 는 위치를 비운 출발)

    html = tmp_path / "v.html"
    preview_map.main(["--gpx", gpx, "--anchors", anchors, "--gps", str(out), "--name", "v", "--out", str(html)])
    text = html.read_text(encoding="utf-8")
    assert "<svg" in text and text.count('class="anchor"') == 5 and 'class="hold"' in text


def test_holdout_marks_edit_cut_edges(tmp_path):
    # 10~11 초 사이에 500 m 를 건너뜀 (편집 컷) → 10·11 초 기준점은 경계, 나머지는 보통 기준점
    gpx = write_gpx(tmp_path / "r.gpx", STRAIGHT)
    anchors = write_anchors(tmp_path / "a.csv", [(0, (0, 0), ""), (5, (0, 50), ""), (10, (0, 100), "컷 전"),
                                                 (11, (0, 600), "컷 뒤"), (16, (0, 650), ""), (21, (0, 700), "")])
    _, _, results = holdout_error.evaluate(gpx, anchors)
    assert {h.snapped.anchor.t_s: h.stop_edge for h in results} == {5: False, 10: True, 11: True, 16: False}
