import json

import pytest
from daboyeo_contracts import GpsTrack

from tools.gps.bdd_to_csv import convert, info_to_track, main

START_MS = 1_503_828_163_000


def gps(dt_s, lat, lon, speed=10.0, **kw):
    return {"timestamp": START_MS + int(dt_s * 1000), "latitude": lat, "longitude": lon, "speed": speed, **kw}


def info(points, **kw):
    return {"filename": "abc-123.mov", "startTime": START_MS, "endTime": START_MS + 40_000, "gps": points, **kw}


def test_times_relative_to_start_and_speed_kmh():
    track = info_to_track(info([gps(-0.4, 40.0, -74.0), gps(0.6, 40.0001, -74.0, speed=5.0), gps(1.6, 40.0002, -74.0)]))
    assert [round(p.t_s, 3) for p in track.points] == [-0.4, 0.6, 1.6]
    assert track.points[1].speed_kmh == pytest.approx(18.0)
    # 영상 시작 직전 점이 있어서 t=0 을 보간할 수 있다
    lat, _ = track.at(0.0)
    assert lat == pytest.approx(40.0 + 0.0001 * 0.4)


def test_drops_bad_points():
    track = info_to_track(info([
        gps(0, 40.0, -74.0),
        gps(1, 0.0, 0.0),                        # (0,0)
        gps(2, None, -74.0),                     # 좌표 없음
        gps(2, 40.0002, -74.0, speed=-1),        # 음수 속도 = 모름
        gps(2, 40.0003, -74.0),                  # 같은 시각
        gps(1.5, 40.0004, -74.0),                # 거꾸로
        gps(3, 95.0, -74.0),                     # 범위 밖
        {"latitude": 40.0005, "longitude": -74.0},  # 시각 없음
        gps(4, 40.0006, -74.0),
    ]))
    assert [p.t_s for p in track.points] == [0.0, 2.0, 4.0]
    assert track.points[1].speed_kmh is None


def test_seconds_timestamps_and_locations_fallback_and_no_start():
    pts = [{"timestamp": 1_503_828_163 + i, "latitude": 40 + i * 1e-4, "longitude": -74.0} for i in range(3)]
    track = info_to_track({"locations": pts})
    assert [p.t_s for p in track.points] == [0.0, 1.0, 2.0]
    assert all(p.speed_kmh is None for p in track.points)


def test_errors():
    with pytest.raises(ValueError, match="위치 목록"):
        info_to_track({"startTime": START_MS, "gps": []})
    with pytest.raises(ValueError, match="2개 이상"):
        info_to_track(info([gps(0, 40.0, -74.0)]))


def test_convert_writes_contract_csv_named_after_video(tmp_path):
    src = tmp_path / "info" / "abc-123.json"
    src.parent.mkdir()
    src.write_text(json.dumps(info([gps(i, 40 + i * 1e-4, -74.0) for i in range(41)])))
    out, _ = convert(src, tmp_path / "videos")
    assert out == tmp_path / "videos" / "abc-123.gps.csv"
    assert out.read_text().splitlines()[0] == "t_s,lat,lon,speed_kmh"
    back = GpsTrack.from_csv(out)
    assert (back.start_s, back.end_s) == (0.0, 40.0)
    assert back.length_m() == pytest.approx(445, abs=5)  # 0.004° 위도 ≈ 445 m


def test_main_reports_failures(tmp_path, capsys):
    good = tmp_path / "a.json"
    good.write_text(json.dumps(info([gps(0, 40.0, -74.0), gps(1, 40.0001, -74.0)])))
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"gps": []}))
    assert main([str(good), str(bad)]) == 1
    assert (tmp_path / "abc-123.gps.csv").exists()
    assert "b.json" in capsys.readouterr().err
