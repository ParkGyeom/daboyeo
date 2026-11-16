"""GPS 형식 A·B·C 예시 줄 (docs/DATA.md 3장)을 바르게 읽는지, 범위 밖 줄을 거르는지 (B4 완료 기준)."""

import pytest

from edge.blackbox.gps import detect_format, hhmmss_to_seconds, parse_line, read_gps

A = "22719.6,36.6968345,127.4048083,20.275,312.16,0.27,1.95,5.45"
B = "1,30000.00,37.12892833,126.88052500,48.380,349,,,"
C = "10848.0,37.18712667,127.09456167,39.69,171,null,null,null,null"


@pytest.mark.parametrize("line,form", [(A, "A"), (B, "B"), (C, "C")])
def test_detect_format(line, form):
    assert detect_format(line.split(",")) == form


def test_format_a():
    f = parse_line(A)
    assert (f.lat, f.lon, f.heading) == (36.6968345, 127.4048083, 312.16)
    assert f.time_s == pytest.approx(2 * 3600 + 27 * 60 + 19.6)  # 02:27:19.6
    assert f.speed_kmh == pytest.approx(20.275 * 1.852)


def test_format_b():
    f = parse_line(B)
    assert (f.lat, f.lon, f.heading) == (37.12892833, 126.880525, 349)
    assert f.time_s == 3 * 3600  # 03:00:00
    assert f.speed_kmh == pytest.approx(48.38 * 1.852)


def test_format_c():
    f = parse_line(C)
    assert (f.lat, f.lon, f.heading) == (37.18712667, 127.09456167, 171)
    assert f.time_s == 1 * 3600 + 8 * 60 + 48  # 01:08:48


@pytest.mark.parametrize(
    "line",
    [
        "22719.6,0.0001,0.0002,20.275,312.16,0.27,1.95,5.45",  # 좌표가 0 근처인 불량 행
        "22719.6,35.68,139.69,20.0,10.0,0,0,0",  # 한국 밖 (도쿄)
        "22719.6,,,20.0,10.0,null,null,null,null",  # 좌표 없음
        "",
        "1,2,3",
    ],
)
def test_bad_lines_rejected(line):
    assert parse_line(line) is None


def test_hhmmss():
    assert hhmmss_to_seconds(235924.41) == pytest.approx(23 * 3600 + 59 * 60 + 24.41)
    assert hhmmss_to_seconds(76000) is None  # 60분은 없다


def test_read_gps_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text(B + "\n")
    assert read_gps(p).lat == 37.12892833
    assert read_gps(tmp_path / "없음.csv") is None
