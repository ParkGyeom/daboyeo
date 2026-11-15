"""영상 + GPS CSV 입력 (edge/blackbox/source.py) 과 그 입력으로 도는 drive()."""

import cv2
import numpy as np
import pytest

from edge.blackbox import run
from edge.blackbox.detect import Detection
from edge.blackbox.source import VideoSource, default_gps_path

FPS = 10
N = 25  # 2.5초


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "v1.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert w.isOpened()
    for i in range(N):
        img = np.full((48, 64, 3), 120, np.uint8)
        img[:, : 2 + i * 2] = 200  # 프레임마다 다르게
        w.write(img)
    w.release()
    # GPS 는 0~2초만 있다 → 2초 넘는 프레임은 위치 없음
    default_gps_path(path).write_text("t_s,lat,lon\n0,37.5,127.0\n2,37.502,127.0\n")
    return path


def test_default_gps_path():
    assert default_gps_path("data/videos/v1.mp4").as_posix() == "data/videos/v1.gps.csv"


def test_video_source_times_and_interpolated_gps(video):
    src = VideoSource(video)
    assert len(src) == N
    frames = list(src)
    assert len(frames) == N
    f10 = frames[10]
    assert f10.fix.time_s == pytest.approx(1.0)
    assert f10.fix.lat == pytest.approx(37.501)  # 0초와 2초 사이 중간
    assert frames[20].fix is not None and frames[21].fix is None  # 2.0초까지만 위치가 있다
    assert f10.name == "v1@1.00s"


def test_video_source_load_reads_current_frame(video):
    widths = []
    for f in VideoSource(video, start_offset=5):
        img = f.load()
        assert img.shape == (48, 64, 3)
        widths.append(int((img[0, :, 0] > 160).sum()))
    assert len(widths) == N - 5
    assert widths == sorted(widths) and widths[0] < widths[-1]  # 순서대로, 건너뛴 앞 5장 뒤부터


def test_video_source_missing_gps(tmp_path, video):
    with pytest.raises(FileNotFoundError):
        VideoSource(video, gps=tmp_path / "none.csv")


class FakeDetector:
    def __init__(self, *a, **k):
        pass

    def detect(self, image):
        return [Detection("pothole", 0.9, (1, 1, 20, 20))]


def test_drive_video_dry_run(video, monkeypatch, capsys):
    monkeypatch.setattr(run, "Detector", FakeDetector)
    args = run.parse_args(["--video", str(video), "--model", "m.onnx", "--conf", "0.25", "--speed", "0",
                           "--dry-run", "--detect-every", "5", "--lane-every", "0", "--no-quality"])
    r = run.drive(args)
    assert r["frames"] == N
    # 0, 5, 10, 15, 20 번째 프레임만 탐지 (25 프레임째는 없음). 21~24 는 위치 없음
    assert r["detected_frames"] == 5
    assert r["skipped_no_gps"] == 4


def test_video_and_segment_are_exclusive():
    with pytest.raises(SystemExit):
        run.parse_args(["--video", "a.mp4", "--segment", "x:00000001-00000002", "--model", "m.onnx"])
    with pytest.raises(SystemExit):
        run.parse_args(["--model", "m.onnx"])
