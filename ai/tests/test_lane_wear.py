import time

import cv2
import numpy as np
import pytest
from daboyeo_ai.lane_wear import PROVISIONAL_THRESHOLD, LaneWearMeter, LaneWearResult

W, H = 1280, 720
ROAD = 80  # 회색 노면 밝기


def road_frame(seed=0, road=ROAD):
    """하늘 + 잡음 있는 회색 노면 (1280×720 BGR)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), road, np.uint8)
    img[: H // 2] = (200, 170, 140)  # 하늘
    noise = rng.normal(0, 4, (H, W, 1))
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def draw_lanes(img, color, thickness=12, dashed=False):
    """원근이 있는 좌우 차선 두 개."""
    for (xb, yb), (xt, yt) in [((260, 690), (590, 440)), ((1020, 690), (690, 440))]:
        if not dashed:
            cv2.line(img, (xb, yb), (xt, yt), color, thickness)
            continue
        for i in range(0, 6, 2):  # 6등분 중 짝수 칸만 칠함 (점선 3토막)
            a, b = i / 6, (i + 1) / 6
            p = (round(xb + (xt - xb) * a), round(yb + (yt - yb) * a))
            q = (round(xb + (xt - xb) * b), round(yb + (yt - yb) * b))
            cv2.line(img, p, q, color, thickness)
    return img


@pytest.fixture(scope="module")
def meter():
    return LaneWearMeter()


def test_bright_white_lane_found_not_worn(meter):
    r = meter.measure(draw_lanes(road_frame(), (210, 210, 210)))
    assert isinstance(r, LaneWearResult)
    assert r.lane_found and not r.is_worn
    assert r.contrast > 1.0
    assert r.n_segments >= 2 and r.lane_pixel_ratio > 0
    assert 0.5 < r.confidence <= 1.0


def test_dashed_lane_found(meter):
    r = meter.measure(draw_lanes(road_frame(1), (210, 210, 210), dashed=True))
    assert r.lane_found and not r.is_worn


def test_faded_lane_found_and_worn(meter):
    r = meter.measure(draw_lanes(road_frame(2), (ROAD + 14,) * 3))
    assert r.lane_found, r
    assert r.contrast < PROVISIONAL_THRESHOLD and r.is_worn
    assert 0.0 < r.confidence <= 1.0


def test_contrast_is_relative_to_road_brightness():
    """같은 비율의 대비면 밝은 노면·어두운 노면에서 비슷한 값이 나온다."""
    m = LaneWearMeter()
    dark = m.measure(draw_lanes(road_frame(3, road=60), (120,) * 3))
    light = m.measure(draw_lanes(road_frame(4, road=110), (220,) * 3))
    assert dark.lane_found and light.lane_found
    assert dark.contrast == pytest.approx(light.contrast, abs=0.15)


def test_no_lane_not_found(meter):
    r = meter.measure(road_frame(5))
    assert not r.lane_found and not r.is_worn
    assert r.contrast is None and r.confidence == 0.0


def test_rough_texture_without_lane_is_not_worn(meter):
    """거친 노면 질감의 흩어진 밝은 점이 이어져 가짜 '흐린 차선'으로 잡히면 안 된다."""
    rng = np.random.default_rng(12)
    tex = cv2.GaussianBlur(rng.normal(0, 20, (H, W)).astype(np.float32), (3, 3), 0)
    img = np.clip(road_frame(12).astype(np.float32) + tex[..., None], 0, 255).astype(np.uint8)
    assert not meter.measure(img).is_worn


def test_horizontal_stop_line_is_not_a_lane(meter):
    img = road_frame(6)
    cv2.line(img, (300, 620), (980, 620), (220, 220, 220), 14)
    assert not meter.measure(img).lane_found


def test_yellow_lane_found(meter):
    r = meter.measure(draw_lanes(road_frame(7), (30, 180, 200)))  # BGR 노랑
    assert r.lane_found and not r.is_worn


def test_threshold_controls_decision():
    img = draw_lanes(road_frame(8), (ROAD + 30,) * 3)
    c = LaneWearMeter().measure(img).contrast
    assert LaneWearMeter(threshold=c + 0.1).measure(img).is_worn
    assert not LaneWearMeter(threshold=c - 0.1).measure(img).is_worn


def test_small_frame_is_not_upscaled(meter):
    img = cv2.resize(draw_lanes(road_frame(9), (210, 210, 210)), (480, 270), interpolation=cv2.INTER_AREA)
    assert meter.measure(img).lane_found


@pytest.mark.parametrize("bad", [
    np.zeros((H, W), np.uint8),            # 회색조
    np.zeros((H, W, 4), np.uint8),         # BGRA
    np.zeros((H, W, 3), np.float32),       # dtype
    np.zeros((10, 10, 3), np.uint8),       # 너무 작음
])
def test_bad_shape_raises(meter, bad):
    with pytest.raises(ValueError):
        meter.measure(bad)


def test_non_array_raises(meter):
    with pytest.raises(TypeError):
        meter.measure([[0, 0, 0]])


def test_bad_args_raise():
    with pytest.raises(ValueError):
        LaneWearMeter(threshold=0)
    with pytest.raises(ValueError):
        LaneWearMeter(roi=((0, 0), (1, 1)))


def test_dark_crack_lines_are_not_lanes(meter):
    """균열·이음매는 노면보다 어두운 선이다. 차선처럼 원근으로 모여도 차선으로 잡으면 안 된다 (극성)."""
    r = meter.measure(draw_lanes(road_frame(13), (30, 30, 30), thickness=6))
    assert not r.lane_found and r.contrast is None


def test_crack_next_to_lane_does_not_make_contrast_negative(meter):
    """실제 프레임에서 균열이 차선으로 섞여 contrast 가 음수로 나오던 문제 (2026-11-15)."""
    img = draw_lanes(road_frame(14), (210, 210, 210))
    for (xb, yb), (xt, yt) in [((330, 690), (600, 440)), ((950, 690), (680, 440))]:
        cv2.line(img, (xb, yb), (xt, yt), (25, 25, 25), 6)  # 차선 안쪽에 나란한 어두운 균열
    r = meter.measure(img)
    assert r.lane_found and r.contrast > 1.0


def test_contrast_uses_adjacent_road_not_whole_roi(meter):
    """그늘 속 차선은 그늘 노면과 비교해야 한다. ROI 전체(햇빛 받은 노면 포함)와 비교하면 음수가 된다."""
    img = road_frame(19, road=40)
    img[H // 2:, 640:] = np.clip(img[H // 2:, 640:].astype(np.int16) + 90, 0, 255).astype(np.uint8)  # 오른쪽 햇빛
    cv2.line(img, (100, 690), (560, 400), (62, 62, 62), 12)  # 그늘 속 차선 (그늘 노면보다 약 50% 밝음)
    cv2.line(img, (380, 690), (600, 400), (62, 62, 62), 12)
    r = meter.measure(img)
    assert r.lane_found
    assert 0.3 < r.contrast < 0.8


def test_one_sided_edge_is_not_a_lane(meter):
    """연석·그림자 경계처럼 한쪽만 밝은 가장자리는 줄무늬가 아니다."""
    img = road_frame(15)
    pts = np.array([[0, 720], [0, 600], [560, 380], [600, 380], [260, 720]], np.int32)
    cv2.fillPoly(img, [pts], (170, 170, 170))  # 왼쪽 보도: 노면보다 넓고 밝은 면
    assert not meter.measure(img).lane_found


def test_contrast_is_positive_on_real_like_noise(meter):
    rng = np.random.default_rng(16)
    img = draw_lanes(road_frame(16), (ROAD + 25,) * 3)
    tex = cv2.GaussianBlur(rng.normal(0, 10, (H, W)).astype(np.float32), (5, 5), 0)
    img = np.clip(img.astype(np.float32) + tex[..., None], 0, 255).astype(np.uint8)
    r = meter.measure(img)
    assert r.lane_found and r.contrast > 0


def test_vertical_pole_above_road_is_not_a_lane(meter):
    img = road_frame(17)
    cv2.line(img, (660, 260), (660, 480), (230, 230, 230), 8)  # 화면 가운데 위쪽 수직 기둥
    assert not meter.measure(img).lane_found


@pytest.mark.parametrize("size", [(640, 360), (1280, 720), (1920, 1080)])
def test_resolution_independent(size):
    """화소 단위 값이 작업 해상도에 맞춰 줄어들어, 같은 장면이면 해상도가 달라도 결과가 같다."""
    m = LaneWearMeter()
    base = draw_lanes(road_frame(18), (ROAD + 40,) * 3, thickness=14)
    ref = m.measure(cv2.resize(base, (1280, 720), interpolation=cv2.INTER_AREA))
    r = m.measure(cv2.resize(base, size, interpolation=cv2.INTER_AREA if size[0] <= W else cv2.INTER_LINEAR))
    assert r.lane_found
    assert r.contrast == pytest.approx(ref.contrast, abs=0.1)


def test_debug_segments_in_frame_pixels(meter):
    r, segs = meter.measure_debug(draw_lanes(road_frame(10), (210, 210, 210)))
    assert len(segs) == r.n_segments
    for x1, y1, x2, y2 in segs:
        assert 0 <= x1 < W and 0 <= x2 < W and H // 2 < y1 < H and H // 2 < y2 < H


def test_speed_1280x720(meter):
    """CPU 몇 ms 목표. CI 기계 편차를 감안해 느슨하게 50 ms 로 확인한다."""
    img = draw_lanes(road_frame(11), (210, 210, 210))
    meter.measure(img)
    t = time.perf_counter()
    for _ in range(20):
        meter.measure(img)
    ms = (time.perf_counter() - t) / 20 * 1000
    assert ms < 50, ms


def test_calib_cli_on_folder_and_video(tmp_path, capsys):
    from daboyeo_ai.eval import lane_wear_calib

    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "a_bright.png"), draw_lanes(road_frame(20), (210, 210, 210)))
    cv2.imwrite(str(frames / "b_faded.png"), draw_lanes(road_frame(21), (ROAD + 14,) * 3))
    cv2.imwrite(str(frames / "c_none.png"), road_frame(22))
    out = tmp_path / "calib.csv"
    assert lane_wear_calib.main([str(frames), "--out", str(out), "--overlay-dir", str(tmp_path / "ov")]) == 0
    rows = out.read_text().splitlines()
    assert rows[0].startswith("source,frame,t_s,lane_found,contrast")
    assert len(rows) == 4
    assert len(list((tmp_path / "ov").glob("*.jpg"))) == 3
    printed = capsys.readouterr().out
    assert "차선 찾음 2개" in printed and "p50=" in printed

    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (W, H))
    if not vw.isOpened():
        pytest.skip("이 OpenCV 빌드는 MJPG 쓰기를 지원하지 않음")
    for k in range(20):
        vw.write(draw_lanes(road_frame(30 + k), (210, 210, 210)))
    vw.release()
    rows = lane_wear_calib.run(video, tmp_path / "v.csv", every_s=0.5, threshold=0.3, overlay_dir=None)
    assert [r["frame"] for r in rows] == [0, 5, 10, 15]
    assert rows[1]["t_s"] == pytest.approx(0.5)
    assert all(r["lane_found"] for r in rows)
