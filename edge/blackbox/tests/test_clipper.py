"""결함 구간 클립 (edge/blackbox/clipper.py)."""

import cv2
import numpy as np
import pytest

from edge.blackbox import clipper
from edge.blackbox.clipper import ClipCutter, clip_size, encode_h264

FPS = 60
N = 600  # 10초, 1080p 의 1/4 크기로 만들고 높이만 480 넘게


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))
    assert w.isOpened()
    for i in range(N):
        img = np.full((540, 960, 3), 90, np.uint8)
        cv2.putText(img, str(i), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)
        w.write(img)
    w.release()
    return path


def read(data: bytes, tmp_path):
    p = tmp_path / "out.mp4"
    p.write_bytes(data)
    cap = cv2.VideoCapture(str(p))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _, img = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n, fps, img


def test_clip_size():
    assert clip_size(1920, 1080) == (854, 480)
    assert clip_size(3840, 2160) == (854, 480)
    assert clip_size(640, 360) == (640, 360)  # 작으면 키우지 않는다
    assert clip_size(63, 47) == (64, 48)  # 짝수


def test_cut_window_size_and_overlap(video, tmp_path):
    c = ClipCutter(video)
    data = c.cut(5.0)  # 3~7초 = 4초, 60fps → 30fps 로 줄여 121장
    assert data is not None and data[4:8] == b"ftyp"
    n, fps, img = read(data, tmp_path)
    assert img.shape[:2] == (480, 854)
    assert fps == pytest.approx(30, abs=0.1)
    assert n == 121
    assert c.cut(6.0) is None and c.skipped_overlap == 1  # 앞 클립(3~7초) 안
    # 7.5초: 5.5~9.5초 중 앞 클립 뒤(7.03초~)만. 프레임 422~570 을 2장마다 → 75장
    n, _, _ = read(c.cut(7.5), tmp_path)
    assert n == 75
    n, _, _ = read(c.cut(9.9), tmp_path)  # 9.53초~영상 끝(프레임 599)
    assert n == 14
    assert c.made == 3
    c.close()


def test_cut_at_start(video, tmp_path):
    c = ClipCutter(video)
    n, _, _ = read(c.cut(0.5), tmp_path)  # 0~2.5초
    assert n == 76
    c.close()


def test_encode_without_ffmpeg_uses_opencv(monkeypatch):
    monkeypatch.setattr(clipper.shutil, "which", lambda _: None)
    data, codec = encode_h264([np.full((48, 64, 3), i * 10, np.uint8) for i in range(10)], 10)
    assert data[4:8] == b"ftyp" and codec in ("h264", "mpeg4")
