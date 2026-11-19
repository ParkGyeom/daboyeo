"""탐지기 고르기 (edge/blackbox/detectors.py): fake 라벨 → 관측 수, onnx 기록 → replay 가 같은 결과, 잘못된 기록 거부."""

import json
from typing import ClassVar

import cv2
import numpy as np
import pytest

from edge.blackbox import detectors, run
from edge.blackbox.detect import SENT_TYPES, Detection
from edge.blackbox.detectors import FakeDetector, ReplayDetector
from edge.blackbox.source import default_gps_path

FPS = 10
N = 30  # 3초. GPS 는 전체 구간에 있다


@pytest.fixture()
def video(tmp_path):
    path = tmp_path / "v1.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    assert w.isOpened()
    for i in range(N):
        img = np.full((48, 64, 3), 120, np.uint8)
        img[:, : 2 + i * 2] = 200  # 프레임마다 다르게 (밝은 띠가 넓어진다)
        w.write(img)
    w.release()
    default_gps_path(path).write_text("t_s,lat,lon\n0,37.5,127.0\n3,37.503,127.0\n")
    return path


LABELS = """defect_id,t_start_s,t_end_s,defect_type,memo
P001,0.3,0.8,pothole,구간
C001,1.6,,crack,t_end_s 비움 → 1.6 ± 1초
L001,0.5,2.5,lane_wear,차선 마모 구간
G001,1.0,1.4,guardrail_damage,시설물: 화면에만 그리고 보내지 않는다
S001,2.0,2.2,표지판,한국어 유형도 받는다
"""


@pytest.fixture()
def labels(tmp_path):
    p = tmp_path / "v1.csv"
    p.write_text(LABELS, encoding="utf-8")
    return p


class Captured:
    """run.Sender 를 대신해 보낸 관측을 모은다 (dry-run 과 같고 내용만 남긴다)."""

    log: ClassVar[list] = []

    def __init__(self, server, dry_run=False, **kw):
        from edge.blackbox.sender import Sender

        self._s = Sender(server, dry_run=True, **kw)
        Captured.log = []

    def __getattr__(self, name):
        return getattr(self._s, name)

    def send(self, obs, snapshot, clip=None):
        Captured.log.append((obs.defect_type.value, obs.confidence, obs.bbox, obs.t_s))
        return self._s.send(obs, snapshot, clip)


def drive(video, tmp_path, *extra):
    args = run.parse_args(["--video", str(video), "--speed", "0", "--dry-run", "--no-quality",
                           "--live-dir", str(tmp_path / "live"), *extra])
    return args, run.drive(args)


# ---- fake ------------------------------------------------------------------------


def test_fake_sends_one_observation_per_sent_label(video, labels, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "Sender", Captured)
    _, r = drive(video, tmp_path, "--detector", "fake", "--labels", str(labels), "--detect-every", "3")
    assert r["sent"] == 3  # 라벨 5건 중 시설물 2건은 SENT_TYPES 밖
    assert sorted(t for t, *_ in Captured.log) == ["crack", "lane_wear", "pothole"]
    lane = [o for o in Captured.log if o[0] == "lane_wear"]
    assert lane[0][2] is None  # 차선 마모는 박스 없이


def test_fake_default_labels_path():
    assert detectors.default_labels_path("data/videos/gn_a.mp4").as_posix() == "experiments/gt_events/gn_a.csv"


def test_fake_detector_fires_each_label_once(labels):
    fake = FakeDetector(labels)
    img = np.zeros((100, 200, 3), np.uint8)
    hits = []
    for t in np.arange(0, 3.01, 0.1):
        hits += [d.type for d in fake.detect(img, frame=f"v1@{t:.2f}s", t=float(t))]
    assert sorted(hits) == sorted(["pothole", "crack", "lane_wear", "guardrail", "sign"])
    assert fake.remaining == 0
    dets = FakeDetector(labels).detect(img, "v1@2.00s", 2.0)
    assert sorted(d.type for d in dets) == ["crack", "lane_wear", "sign"]  # 포트홀·가드레일은 시작 전에 끝남
    assert all(0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 100 for x1, y1, x2, y2 in (e.bbox for e in dets))
    assert all(d.confidence == detectors.FAKE_CONFIDENCE for d in dets)


def test_fake_detector_late_frame_and_labels_before_start(labels):
    img = np.zeros((10, 10, 3), np.uint8)
    # 시작이 1.5초면 0.3~0.8초 포트홀(끝남)은 못 본 것, 1.0~1.4초 가드레일도 끝남
    fake = FakeDetector(labels)
    assert sorted(d.type for d in fake.detect(img, "a", 1.5)) == ["crack", "lane_wear"]
    # 분석 프레임이 표지판 구간(2.0~2.2)을 건너뛰어도 늦게라도 1번 낸다 (관측 수 = 라벨 수)
    assert [d.type for d in fake.detect(img, "b", 2.9)] == ["sign"]
    assert fake.detect(img, "c", 3.0) == []


def test_fake_needs_video_and_existing_labels(tmp_path):
    with pytest.raises(FileNotFoundError):
        FakeDetector(tmp_path / "none.csv")
    args = run.parse_args(["--segment", "x_y:00000001-00000002", "--detector", "fake"])
    with pytest.raises(ValueError):
        detectors.make_detector(args)


# ---- record → replay -----------------------------------------------------------------


class StubOnnx:
    """모델 없이: 밝은 띠 너비에 따라 프레임마다 다른 탐지를 낸다."""

    def __init__(self, *a, **k):
        pass

    def detect(self, image):
        width = int((image[0, :, 0] > 160).sum())
        out = [Detection("pothole", round(0.3 + width / 200, 4), (1, 1, max(2, width), 20))]
        if width % 3 == 0:
            out.append(Detection("guardrail", 0.5, (40, 5, 60, 30)))
        if width > 30:
            out.append(Detection("crack", 0.45, (10, 25, 50, 40)))
        return out


def test_record_then_replay_gives_same_observations(video, tmp_path, monkeypatch):
    monkeypatch.setattr(detectors, "Detector", StubOnnx)
    monkeypatch.setattr(run, "Sender", Captured)
    model = tmp_path / "detector_v2.onnx"
    model.write_bytes(b"")
    rec = tmp_path / "replay" / "v1.jsonl"

    _, r1 = drive(video, tmp_path, "--model", str(model), "--conf", "0.25", "--detect-every", "4",
                  "--lane-every", "0", "--record", str(rec))
    sent_onnx = list(Captured.log)
    assert r1["sent"] > 0 and len(sent_onnx) == r1["sent"]

    head, *lines = [json.loads(ln) for ln in rec.read_text().splitlines()]
    assert head["model"] == "detector_v2.onnx" and head["video"] == "v1.mp4" and head["detect_every"] == 4
    assert len(lines) == r1["detected_frames"]

    monkeypatch.setattr(detectors, "Detector", None)  # replay 는 모델을 부르지 않는다
    assert head["version"] == 2 and head["conf"] == 0.25  # 기록에 탐지 기준이 남는다
    _, r2 = drive(video, tmp_path, "--detector", "replay", "--record", str(rec), "--detect-every", "4",
                  "--lane-every", "0", "--conf", "0.25")
    assert Captured.log == sent_onnx
    assert r2["detected_frames"] == r1["detected_frames"]


V3_CONF = {"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}


def _write_record(path, **head):
    h = {"kind": "daboyeo-detections", "version": 2, "model": "m.onnx", "video": "v1.mp4", "detect_every": 5,
         "conf": V3_CONF}
    h.update(head)
    lines = [h,
             {"frame": "v1@0.00s", "t": 0.0, "dets": [["pothole", 0.8, 1, 2, 30, 40]]},
             {"frame": "v1@0.50s", "t": 0.5, "dets": []},
             {"frame": "v1@1.00s", "t": 1.0, "dets": [["crack", 0.6, 5, 6, 20, 30]]}]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_replay_matches_by_name_then_nearest_time(tmp_path):
    rep = ReplayDetector(_write_record(tmp_path / "r.jsonl"), "v1.mp4", 5)
    img = np.zeros((48, 64, 3), np.uint8)
    assert [d.type for d in rep.detect(img, "v1@0.00s", 0.0)] == ["pothole"]
    # 시작 시점을 늦춘 기기(bb-03): 이름은 없지만 기록 간격(0.5초)의 절반 안이면 가장 가까운 줄
    assert [d.type for d in rep.detect(img, "v1@1.10s", 1.1)] == ["crack"]
    assert rep.detect(img, "v1@0.40s", 0.4) == []  # 0.5초 줄 (탐지 없음)
    assert rep.detect(img, "v1@2.00s", 2.0) == []  # 기록 밖
    assert rep.name == "replay:m.onnx" and rep.frames == 3


@pytest.mark.parametrize("head, msg", [
    ({"video": "gn_b.mp4"}, "gn_b.mp4"),
    ({"detect_every": 10}, "탐지 주기"),
    ({"kind": "something"}, "탐지 기록 파일이 아니다"),
    ({"version": 1}, "다시 뜬다"),  # 기준이 적혀 있지 않은 예전 기록
])
def test_replay_rejects_mismatched_record(tmp_path, head, msg):
    with pytest.raises(ValueError, match=msg):
        ReplayDetector(_write_record(tmp_path / "r.jsonl", **head), "v1.mp4", 5)


def test_replay_rejects_broken_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayDetector(tmp_path / "none.jsonl", "v1.mp4", 5)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    with pytest.raises(ValueError):
        ReplayDetector(bad, "v1.mp4", 5)
    p = _write_record(tmp_path / "r.jsonl")
    p.write_text(p.read_text() + '{"frame": "v1@1.50s", "t": 1.5, "dets": [["pothole", 0.5]]}\n')
    with pytest.raises(ValueError, match=":5:"):
        ReplayDetector(p, "v1.mp4", 5)


def test_drive_stops_on_wrong_record(video, tmp_path):
    rec = _write_record(tmp_path / "r.jsonl", video="other.mp4")
    _, r = drive(video, tmp_path, "--detector", "replay", "--record", str(rec), "--detect-every", "5")
    assert r == {"frames": 0}
    _, r = drive(video, tmp_path, "--detector", "replay", "--detect-every", "5")  # --record 없음
    assert r == {"frames": 0}
    _, r = drive(video, tmp_path)  # onnx 인데 --model 없음
    assert r == {"frames": 0}


def test_sent_types_unchanged():
    # fake 의 관측 수 규칙은 SENT_TYPES 에 기대므로, 바뀌면 이 테스트와 handoff 12장을 같이 본다
    assert SENT_TYPES == {"pothole", "crack", "lane_wear"}


def test_replay_rejects_record_made_with_other_conf(tmp_path):
    """v2 기준(균열 0.29)으로 뜬 기록을 v3 기준(0.42)으로 재생하지 않는다 — 기록을 다시 뜬다."""
    rec = _write_record(tmp_path / "r.jsonl", conf={**V3_CONF, "crack": 0.29})
    with pytest.raises(ValueError, match="다시 뜬다"):
        ReplayDetector(rec, "v1.mp4", 5, conf=V3_CONF)
    assert ReplayDetector(_write_record(tmp_path / "ok.jsonl"), "v1.mp4", 5, conf=dict(V3_CONF)).conf == V3_CONF


def test_default_conf_is_v3_model_card():
    """블랙박스 기본 탐지 기준 = 시연 모델 v3 카드의 class_conf_draft (handoff 16장)."""
    args = run.parse_args(["--video", "a.mp4", "--model", "m.onnx"])
    assert args.conf_json == "ai/models/detector_v3.json"
    assert detectors._conf_from_args(args) == V3_CONF


def test_fake_skips_amb_labels_unless_asked(tmp_path):
    p = tmp_path / "v1.csv"
    p.write_text("defect_id,t_start_s,t_end_s,defect_type,memo\n"
                 "C1,0.0,1.0,crack,확실\nC2,0.5,1.5,crack,amb 블록 포장 이음매\nP1,1.0,2.0,pothole,\n", encoding="utf-8")
    img = np.zeros((10, 10, 3), np.uint8)
    fake = FakeDetector(p)
    assert fake.skipped_amb == 1 and len(fake.events) == 2
    assert sorted(d.type for d in fake.detect(img, "a", 1.0)) == ["crack", "pothole"]  # amb 균열 C2 는 안 냄
    assert len(FakeDetector(p, include_amb=True).events) == 3


def test_road_tiles_off_by_default():
    """조각 추론은 기본 끔 — 시연 동작이 그대로다 (handoff 26장)."""
    args = run.parse_args(["--video", "a.mp4", "--model", "m.onnx"])
    assert args.road_tiles == 0 and detectors.tiles_from_args(args) is None


def test_road_tiles_args():
    args = run.parse_args(["--video", "a.mp4", "--model", "m.onnx",
                           "--road-tiles", "2", "--tiles-bottom", "0.8", "--tile-classes", "crack"])
    assert detectors.tiles_from_args(args) == {"cols": 2, "bottom": 0.8, "classes": ["crack"]}
    for bad in (["--tile-classes", "lane"], ["--tiles-bottom", "0.3"], ["--road-tiles", "-1"]):
        with pytest.raises(SystemExit):
            run.parse_args(["--video", "a.mp4", "--road-tiles", "2", *bad])


def test_onnx_mode_passes_road_tiles_and_records_them(tmp_path, monkeypatch):
    """--road-tiles 2 → 프레임 전체 + 도로 2조각 창, 조각에서는 균열만. 기록 머리 줄에 설정이 남는다."""
    seen = {}

    class FakeDet:
        def __init__(self, model, conf, threads, windows=None, tile_classes=None):
            seen.update(windows=windows, tile_classes=tile_classes)

        def detect(self, image):
            return []

    monkeypatch.setattr(detectors, "Detector", FakeDet)
    tiles = {"cols": 2, "bottom": 0.8, "classes": ["crack"]}
    mode = detectors.OnnxMode("ai/models/detector_v3.onnx", conf=V3_CONF, tiles=tiles)
    assert mode.name == "detector_v3.onnx+tiles2"
    assert len(seen["windows"]) == 3 and seen["windows"][0] == (0.0, 0.0, 1.0, 1.0)
    assert all(w[3] == 0.8 for w in seen["windows"][1:]) and seen["tile_classes"] == ("crack",)

    rec = detectors.Recorder(mode, tmp_path / "r.jsonl", "v1.mp4", 5)
    rec.close()
    head = json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert head["tiles"] == tiles and head["model"] == "detector_v3.onnx+tiles2"

    detectors.OnnxMode("m.onnx", conf=V3_CONF)
    assert seen["windows"] is None and seen["tile_classes"] is None  # 끔이면 지금과 같다
