"""영상 + GPS CSV 입력 (edge/blackbox/source.py) 과 그 입력으로 도는 drive()."""

import json

import cv2
import numpy as np
import pytest

from edge.blackbox import detectors, run
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


def test_drive_video_dry_run(video, monkeypatch, tmp_path):
    monkeypatch.setattr(detectors, "Detector", FakeDetector)
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    args = run.parse_args(["--video", str(video), "--model", str(model), "--conf", "0.25", "--speed", "0",
                           "--dry-run", "--detect-every", "5", "--lane-every", "0", "--no-quality",
                           "--live-dir", str(tmp_path / "live"), "--keep-dir", str(tmp_path / "keep")])
    r = run.drive(args)
    assert r["frames"] == N
    # 0, 5, 10, 15, 20 번째 프레임만 탐지 (25 프레임째는 없음). 21~24 는 위치 없음
    assert r["detected_frames"] == 5
    assert r["skipped_no_gps"] == 4
    # 같은 자리 포트홀 5번 탐지 → 연속 프레임 합치기로 관측 1건 (계획 2.3), 그 관측의 클립 1개 (계획 1.3)
    assert r["detections"] == 5 and r["tracks"] == 1
    assert r["sent"] == 1 and r["clips"] == 1 and r["clips_skipped_overlap"] == 0
    assert r["bytes_sent"] == r["bytes_meta"] + r["bytes_snapshot"] + r["bytes_clip"] and r["bytes_clip"] > 0
    kept = list((tmp_path / "keep" / "bb-01").glob("*/*"))
    assert sum(f.suffix == ".jpg" for f in kept) == 1 and sum(f.suffix == ".mp4" for f in kept) == 1
    # 블랙박스 화면: 박스를 그린 마지막 프레임과 끝난 상태가 남는다
    live = tmp_path / "live"
    img = cv2.imread(str(live / "bb-01.jpg"))
    assert img is not None and img.shape[:2] == (48, 64)
    status = json.loads((live / "bb-01.json").read_text())
    assert status["ended"] is True and status["source"] == "v1.mp4" and status["found"] == 5


def test_video_and_segment_are_exclusive():
    with pytest.raises(SystemExit):
        run.parse_args(["--video", "a.mp4", "--segment", "x:00000001-00000002", "--model", "m.onnx"])
    with pytest.raises(SystemExit):
        run.parse_args(["--model", "m.onnx"])


def _drive(video, tmp_path, *extra):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"")
    # 클립은 test_drive_video_dry_run 에서 본다. 여기서는 합치기·주기만 (--no-clip)
    args = run.parse_args(["--video", str(video), "--model", str(model), "--conf", "0.25", "--speed", "0",
                           "--dry-run", "--lane-every", "0", "--no-clip", "--live-dir", str(tmp_path / "live"), *extra])
    return run.drive(args)


def test_drive_merges_same_pothole_into_one_observation(video, monkeypatch, tmp_path):
    """같은 자리 포트홀이 매 프레임 보이면 합쳐서 1건, --no-merge 면 탐지마다 1건 (계획 2.3)."""
    monkeypatch.setattr(detectors, "Detector", FakeDetector)
    merged = _drive(video, tmp_path, "--detect-every", "2", "--no-quality")
    assert merged["detected_frames"] == 11  # 0,2,…,20 (21~24 는 위치 없음)
    assert merged["detections"] == 11 and merged["tracks"] == 1 and merged["sent"] == 1
    plain = _drive(video, tmp_path, "--detect-every", "2", "--no-quality", "--no-merge")
    assert plain["sent"] == 11 and plain["tracks"] is None
    assert merged["bytes_sent"] < plain["bytes_sent"] / 5


def test_drive_runs_quality_every_frame_and_logs_stage_ratio(video, monkeypatch, tmp_path, capsys):
    """품질은 매 프레임, 탐지는 5프레임마다 (계획 2.2). 끝날 때 단계별 실행 비율을 출력한다."""
    monkeypatch.setattr(detectors, "Detector", FakeDetector)
    r = _drive(video, tmp_path, "--detect-every", "5")
    st = r["stages"]
    assert st["frames"] == 21  # 위치 있는 프레임
    assert st["quality"]["runs"] == 21 and st["lane"]["runs"] == 0
    # 탐지 차례는 0,5,10,15,20. 합성 프레임 일부는 밋밋해서 품질 나쁨(흐림)으로 건너뛴다
    assert st["detect"]["runs"] + r["skipped_quality"] == 5 and st["detect"]["runs"] >= 1
    assert r["quality_bad"] >= r["skipped_quality"]
    assert "단계별 실행 품질 21/21 (100%" in capsys.readouterr().out


def test_lane_wear_is_off_by_default():
    """시연 기본은 차선 마모를 끈다 (handoff 17장, plan r3.15). 켜려면 --lane-every 3."""
    assert run.parse_args(["--video", "a.mp4", "--model", "m.onnx"]).lane_every == 0
    assert run.parse_args(["--video", "a.mp4", "--model", "m.onnx", "--lane-every", "3"]).lane_every == 3
