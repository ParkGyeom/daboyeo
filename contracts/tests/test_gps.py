import pytest

from daboyeo_contracts import GpsPoint, GpsTrack, haversine_m


def make_track():
    return GpsTrack(
        [
            GpsPoint(0.0, 37.5500, 126.9200),
            GpsPoint(10.0, 37.5510, 126.9200),
            GpsPoint(20.0, 37.5510, 126.9220),
        ]
    )


def test_haversine_known_distance():
    # 위도 0.001도 ≈ 111.2 m
    assert haversine_m(37.55, 126.92, 37.551, 126.92) == pytest.approx(111.2, abs=0.5)


def test_interpolates_linearly_and_hits_points_exactly():
    t = make_track()
    assert t.at(0.0) == (37.5500, 126.9200)
    assert t.at(10.0) == pytest.approx((37.5510, 126.9200))
    assert t.at(20.0) == pytest.approx((37.5510, 126.9220))
    assert t.at(5.0) == pytest.approx((37.5505, 126.9200))
    assert t.at(15.0) == pytest.approx((37.5510, 126.9210))


def test_out_of_range_raises_unless_clamped():
    t = make_track()
    with pytest.raises(ValueError, match="outside"):
        t.at(25.0)
    assert t.at(25.0, clamp=True) == pytest.approx((37.5510, 126.9220))
    assert t.at(-1.0, clamp=True) == (37.5500, 126.9200)


def test_rejects_non_increasing_time():
    with pytest.raises(ValueError, match="increase"):
        GpsTrack([GpsPoint(0, 37.5, 126.9), GpsPoint(0, 37.6, 126.9)])


def test_csv_round_trip(tmp_path):
    t = make_track()
    path = tmp_path / "track.gps.csv"
    t.to_csv(path)
    assert path.read_text().splitlines()[0] == "t_s,lat,lon"
    loaded = GpsTrack.from_csv(path)
    assert loaded.at(15.0) == pytest.approx(t.at(15.0))
    assert loaded.length_m() == pytest.approx(t.length_m(), rel=1e-6)


def test_csv_with_optional_speed(tmp_path):
    path = tmp_path / "bdd.csv"
    path.write_text("t_s,lat,lon,speed_kmh\n0,37.55,126.92,30\n1,37.5501,126.92,\n")
    t = GpsTrack.from_csv(path)
    assert t.points[0].speed_kmh == 30.0
    assert t.points[1].speed_kmh is None


def test_csv_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("time,lat,lon\n0,37.55,126.92\n")
    with pytest.raises(ValueError, match="missing columns"):
        GpsTrack.from_csv(path)
