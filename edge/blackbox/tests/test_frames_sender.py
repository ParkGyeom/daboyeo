import cv2
import numpy as np
import pytest

from edge.blackbox.frames import list_frames, parse_segment
from edge.blackbox.run import add_gps_noise, adjust_brightness
from edge.blackbox.sender import Sender, crop_snapshot


def test_parse_segment():
    assert parse_segment("0401_light_rainy_smooth:03000071-03000182") == ("0401_light_rainy_smooth", 3000071, 3000182)
    for bad in ["0401_light_rainy_smooth", "x:3000071-3000182", "x:03000182-03000071"]:
        with pytest.raises(ValueError):
            parse_segment(bad)


def test_list_frames_skips_missing(tmp_path):
    (tmp_path / "IMAGE").mkdir()
    for n in (1, 3):
        (tmp_path / "IMAGE" / f"seg_{n:08d}.png").write_bytes(b"")
    frames = list_frames(tmp_path, "seg:00000001-00000003")
    assert [f.name for f in frames] == ["seg_00000001", "seg_00000003"]
    assert frames[0].gps_path == tmp_path / "GPS" / "seg_00000001.csv"


def test_crop_snapshot_small_jpeg():
    img = np.full((1080, 1920, 3), 120, np.uint8)
    jpg = crop_snapshot(img, (100, 100, 1800, 1000))
    out = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert jpg.startswith(b"\xff\xd8") and max(out.shape[:2]) <= 640
    # 작은 결함은 줄이지 않고 주변만 자른다
    small = cv2.imdecode(np.frombuffer(crop_snapshot(img, (500, 500, 540, 520)), np.uint8), cv2.IMREAD_COLOR)
    assert small.shape[:2] == (20 + 2 * 26, 40 + 2 * 36)


def test_dry_run_counts_bytes():
    s = Sender("http://unused", dry_run=True)
    assert s.send({"a": 1}, b"\xff\xd8xx") is None
    assert s.sent == 1 and s.bytes_sent == len('{"a": 1}') + 4


def test_variants():
    import random

    lat, lon = add_gps_noise(37.5, 127.0, 0, random.Random(0))
    assert (lat, lon) == (37.5, 127.0)
    lat, lon = add_gps_noise(37.5, 127.0, 7, random.Random(0))
    assert 0 < abs(lat - 37.5) * 111_320 < 40
    img = np.full((2, 2, 3), 200, np.uint8)
    assert adjust_brightness(img, 0.5)[0, 0, 0] == 100
    assert adjust_brightness(img, 2.0)[0, 0, 0] == 255


def test_variant_args():
    from edge.blackbox.run import parse_args
    from edge.blackbox.variants import VARIANTS, run_args

    assert [v["device_id"] for v in VARIANTS] == ["bb-01", "bb-02", "bb-03"]
    a = parse_args(["--segment", "x:00000001-00000002", "--model", "m.onnx", *run_args(VARIANTS[1])])
    assert (a.device_id, a.gps_noise, a.brightness, a.seed) == ("bb-02", 7.0, 0.85, 2)
    a = parse_args(["--segment", "x:00000001-00000002", "--model", "m.onnx", *run_args(VARIANTS[2])])
    assert (a.device_id, a.start_offset) == ("bb-03", 2)


def test_pacer_does_not_wait_twice_for_missing_gps():
    import pytest

    from edge.blackbox.run import Pacer

    p = Pacer()
    # A(t=0) → B(시각 없음) → C(t=2.4): 합계가 실제 간격 2.4초여야 한다 (예전에는 1.0 + 2.4 = 3.4초)
    waits = [p.next(0.0), p.next(None), p.next(2.4)]
    assert waits[0] == 0 and waits[1] == 1.0 and sum(waits) == pytest.approx(2.4)
    assert p.next(10.0) == 3.0  # 구간이 끊겨도 3초 넘게 기다리지 않는다
    assert p.next(1.0) == 3.0   # 자정을 넘긴 경우도 간격으로 친다(상한 3초)
    q = Pacer()
    assert q.next(None) == 0.0  # 첫 프레임은 기다리지 않는다
