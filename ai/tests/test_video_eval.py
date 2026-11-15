"""영상 평가 도구 (eval_detector · interval_sweep) — 진짜 모델·영상 없이 합성 영상 + 가짜 탐지기로."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest
from daboyeo_ai.eval import eval_detector, interval_sweep
from daboyeo_ai.eval.video_events import (
    FrameRecord,
    GtEvent,
    TypeCounts,
    detector_hits,
    evaluate_video,
    group_events,
    lane_wear_hits,
    load_cache,
    load_conf_json,
    load_gt,
    match,
    normalize_type,
    scan_or_load,
    subsample,
)
from daboyeo_contracts import Detection

FPS = 10.0


def write_video(path, n_frames: int, fps: float = FPS) -> None:
    """프레임 번호를 픽셀 값에 담은 작은 mp4 (가짜 탐지기가 프레임 번호를 읽는다)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert vw.isOpened()
    for i in range(n_frames):
        img = np.zeros((48, 64, 3), np.uint8)
        img[:, :32] = (i // 16) * 16 + 8  # 왼쪽 = 16 의 몫, 오른쪽 = 나머지. 16 단계라 압축 오차에 견딘다
        img[:, 32:] = (i % 16) * 16 + 8
        vw.write(img)
    vw.release()


def frame_index(img: np.ndarray) -> int:
    return int(img[10:38, 4:28].mean() // 16) * 16 + int(img[10:38, 36:60].mean() // 16)


class ScriptedDetector:
    """프레임 번호 → [(class_id, conf)] 대본대로 Detection 을 낸다 (OnnxDetector.detect 와 같은 모양)."""

    def __init__(self, script: dict[int, list[tuple[int, float]]], conf: float = 0.0):
        self.script, self.conf, self.calls = script, conf, 0

    def detect(self, frame_bgr):
        self.calls += 1
        i = frame_index(frame_bgr)
        return [Detection.from_class_id(c, conf=s, bbox=(0.1, 0.1, 0.5, 0.5))
                for c, s in self.script.get(i, []) if s >= self.conf]


# ---- 정답 CSV ---------------------------------------------------------------------


def test_load_gt_english_and_korean(tmp_path):
    p = tmp_path / "v1.csv"
    p.write_text(
        "# 주석 줄\n"
        "defect_id,t_start_s,t_end_s,defect_type,memo\n"
        "D1,2.0,3.0,pothole,오른쪽\n"
        "D2,5.0,,균열,끝 비움\n"
        "D3,0:07,0:08.5,가드레일,mm:ss\n"
        "\n", encoding="utf-8")
    gt = load_gt(p)
    assert [g.defect_type for g in gt] == ["pothole", "crack", "guardrail_damage"]
    assert (gt[1].t_start, gt[1].t_end) == (4.0, 6.0)
    assert (gt[2].t_start, gt[2].t_end) == (7.0, 8.5)
    assert gt[0].video_id == "v1" and gt[0].memo == "오른쪽"


def test_load_gt_t_s_only_korean_header(tmp_path):
    p = tmp_path / "v2.csv"
    p.write_text("시각,유형,메모\n12.5,차선 마모,\n3,sign,표지판 있음\n", encoding="utf-8")
    gt = load_gt(p, point_half_s=0.5)
    assert [(g.defect_type, g.t_start, g.t_end) for g in gt] == [("lane_wear", 12.0, 13.0), ("sign_damage", 2.5, 3.5)]
    assert gt[0].defect_id == "v2#2"  # ID 가 없으면 영상#줄번호


def test_load_gt_errors(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("t_s,유형\n1,모르는것\n", encoding="utf-8")
    with pytest.raises(ValueError, match="모르는 결함 유형"):
        load_gt(p)
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="헤더"):
        load_gt(p)
    p.write_text("t_start_s,t_end_s,type\n5,4,pothole\n", encoding="utf-8")
    with pytest.raises(ValueError, match="t_end_s"):
        load_gt(p)


def test_normalize_type_aliases():
    assert normalize_type("Pothole") == "pothole"
    assert normalize_type("LANE-WEAR") == "lane_wear"
    assert normalize_type("표지판 파손") == "sign_damage"


def test_load_conf_json_shapes(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"pothole": 0.3, "crack": 0.2, "note": "x"}))
    assert load_conf_json(p) == {"pothole": 0.3, "crack": 0.2}
    p.write_text(json.dumps({"per_class_conf": {"guardrail": {"conf": 0.4}, "sign": 0.5}, "model": "v2"}))
    assert load_conf_json(p) == {"guardrail_damage": 0.4, "sign_damage": 0.5}
    # conf_thresholds.py 출력 모양: class_conf + 근거 (classes 에도 클래스 이름 키가 있지만 class_conf 를 먼저 본다)
    p.write_text(json.dumps({"class_conf": {"pothole": 0.22, "sign": 0.24}, "classes": {"pothole": {"conf": 0.9}}}))
    assert load_conf_json(p) == {"pothole": 0.22, "sign_damage": 0.24}
    p.write_text(json.dumps({"model": "v2"}))
    with pytest.raises(ValueError):
        load_conf_json(p)


# ---- 이벤트 묶기 · 맞추기 ------------------------------------------------------------


def test_group_events_merge_gap():
    ev = group_events("v", "pothole", [(1.0, 0.5), (1.4, 0.9), (3.0, 0.4), (1.8, 0.3)], merge_gap_s=0.5)
    assert [(e.t_start, e.t_end, e.max_conf, len(e.hits)) for e in ev] == [(1.0, 1.8, 0.9, 3), (3.0, 3.0, 0.4, 1)]
    assert group_events("v", "pothole", [], 1.0) == []


def test_match_precision_recall_and_fragmentation():
    gt = [GtEvent("v", "A", "pothole", 2.0, 3.0), GtEvent("v", "B", "pothole", 10.0, 11.0),
          GtEvent("v", "C", "crack", 2.0, 3.0)]
    preds = group_events("v", "pothole", [(2.2, .9), (2.9, .9), (3.4, .5), (20.0, .8)], merge_gap_s=0.6)
    # 이벤트: [2.2], [2.9, 3.4](tol 0.5 안), [20.0](오탐). A 찾음, B 놓침. 쪼개진 두 이벤트 모두 맞음.
    c = match(gt, preds, "pothole", tol_s=0.5, minutes=2.0)
    assert (c.n_gt, c.n_gt_found, c.n_pred, c.n_pred_ok, c.missed) == (2, 1, 3, 2, ["B"])
    assert c.recall == 0.5 and c.precision == pytest.approx(2 / 3) and c.miss_rate == 0.5
    assert c.f1 == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert c.per_min == 1.5
    # 유형이 다르면 맞지 않는다 (crack 정답 C 는 pothole 탐지로 찾지 않음)
    assert match(gt, preds, "crack", 0.5, 2.0).n_gt_found == 0


def test_tolerance_boundary():
    gt = [GtEvent("v", "A", "crack", 5.0, 5.0)]
    near = group_events("v", "crack", [(5.5, 1.0)], 1.0)
    far = group_events("v", "crack", [(5.51, 1.0)], 1.0)
    assert match(gt, near, "crack", 0.5, 1).n_gt_found == 1
    assert match(gt, far, "crack", 0.5, 1).n_gt_found == 0


def test_empty_counts_are_none():
    c = TypeCounts()
    assert c.recall is None and c.precision is None and c.f1 is None and c.per_min is None


def test_detector_hits_per_class_conf():
    recs = [FrameRecord(0, 0.0, [("pothole", 0.3), ("pothole", 0.6), ("crack", 0.3)]),
            FrameRecord(1, 0.1, [("crack", 0.25)])]
    h = detector_hits(recs, {"pothole": 0.5, "crack": 0.28})
    assert h["pothole"] == [(0.0, 0.6)] and h["crack"] == [(0.0, 0.3)]
    assert detector_hits(recs, 0.25)["crack"] == [(0.0, 0.3), (0.1, 0.25)]


def test_lane_wear_hits_vote():
    # 차선 선명(p25 1.0, 길이 1.0) 30 프레임 → 마모(길이 0.3) 이어짐 → 확정 1번 (warmup 20 뒤)
    recs = [FrameRecord(i, i * 0.1, [], (True, 1.0, 1.0)) for i in range(30)]
    recs += [FrameRecord(30 + i, (30 + i) * 0.1, [], (True, 1.0, 0.3)) for i in range(10)]
    recs += [FrameRecord(40, 4.0, [], None), FrameRecord(41, 4.1, [], (False, None, 0.0))]
    hits = lane_wear_hits(recs)
    assert len(hits) == 1 and hits[0][0] == pytest.approx(3.2)  # 마모 3 번째 프레임에서 5 중 3


# ---- 합성 영상 끝까지 ------------------------------------------------------------------

# 10 fps, 200 프레임 = 20초. 대본:
#   포트홀 정답 P1 2.0~3.0s: 프레임 20~30 매 프레임 탐지 (conf 0.6)
#   포트홀 정답 P2 10.0s(t_s): 프레임 100 하나만 탐지 (conf 0.3) → 간격 3 이면 못 봄 (100 % 3 != 0), 간격 5·10 은 봄
#   균열 정답 C1 15~16s: 탐지 없음 → 늘 놓침
#   균열 오탐: 프레임 60~62 (conf 0.9)
#   가드레일 정답 G1 6~7s: 프레임 63~66 (conf 0.5) — 시설물 위치
SCRIPT: dict[int, list[tuple[int, float]]] = {}
for _i in range(20, 31):
    SCRIPT.setdefault(_i, []).append((0, 0.6))
SCRIPT.setdefault(100, []).append((0, 0.3))
for _i in range(60, 63):
    SCRIPT.setdefault(_i, []).append((1, 0.9))
for _i in range(63, 67):
    SCRIPT.setdefault(_i, []).append((2, 0.5))

GT_CSV = ("defect_id,t_start_s,t_end_s,t_s,defect_type,memo\n"
          "P1,2.0,3.0,,pothole,\n"
          "P2,,,10.0,포트홀,한 프레임만\n"
          "C1,15,16,,crack,못 찾는 결함\n"
          "G1,6,7,,guardrail,시설물\n")


@pytest.fixture
def synth(tmp_path):
    video = tmp_path / "vid.mp4"
    write_video(video, 200)
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    (gt_dir / "vid.csv").write_text(GT_CSV, encoding="utf-8")
    return video, gt_dir


def test_synthetic_video_frame_index_roundtrip(synth):
    from daboyeo_ai.eval.video_events import iter_video_frames

    got = [(i, frame_index(f)) for i, _, f in iter_video_frames(synth[0], every=7)]
    assert got and all(i == j for i, j in got)


def test_scan_cache_and_subsample(synth, tmp_path):
    video, _ = synth
    det = ScriptedDetector(SCRIPT)
    vr = scan_or_load(video, 1, 0.25, lambda c: det, lane=False, cache_dir=tmp_path / "cache", model_name="fake",
                      log=lambda s: None)
    assert len(vr.records) == 200 and det.calls == 200 and vr.duration_s == pytest.approx(20.0)
    meta, recs = load_cache(tmp_path / "cache" / "vid.jsonl")
    assert meta["every"] == 1 and len(recs) == 200 and recs[25].dets == [("pothole", 0.6)]
    # 캐시가 맞으면 탐지기를 다시 돌리지 않는다 (간격 5 는 1 의 배수)
    vr5 = scan_or_load(video, 5, 0.3, lambda c: pytest.fail("rescan"), lane=False, cache_dir=tmp_path / "cache",
                       model_name="fake", log=lambda s: None)
    assert [r.frame for r in vr5.records] == list(range(0, 200, 5))
    assert subsample(vr.records, 10) == vr.records[::10]


def test_evaluate_video_counts(synth):
    video, gt_dir = synth
    det = ScriptedDetector(SCRIPT)
    vr = scan_or_load(video, 1, 0.25, det, lane=False, model_name="fake", log=lambda s: None)
    gt = load_gt(gt_dir / "vid.csv")
    res = evaluate_video("vid", vr.records, gt, 0.25, vr.duration_s)
    assert (res["pothole"].n_gt, res["pothole"].n_gt_found, res["pothole"].n_pred) == (2, 2, 2)
    assert (res["crack"].n_gt_found, res["crack"].n_pred, res["crack"].n_pred_ok) == (0, 1, 0)
    assert res["guardrail_damage"].n_gt_found == 1 and res["sign_damage"].n_gt == 0
    assert "lane_wear" not in res
    hi = evaluate_video("vid", vr.records, gt, 0.5, vr.duration_s)  # conf 0.5 면 P2(0.3) 를 놓친다
    assert hi["pothole"].missed == ["P2"]


def test_eval_detector_cli(synth, tmp_path, capsys):
    video, gt_dir = synth
    out = tmp_path / "out"
    rc = eval_detector.main([str(video), "--gt-dir", str(gt_dir), "--every", "1", "--confs", "0.25,0.5",
                             "--out", str(out), "--model", str(tmp_path / "fake.onnx")],
                            detector_factory=lambda c: ScriptedDetector(SCRIPT, c))
    assert rc == 0
    rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
    get = {(r["scope"], r["conf"], r["defect_type"]): r for r in rows}
    assert get[("all", "0.25", "pothole")]["recall"] == "1.000"
    assert get[("all", "0.5", "pothole")]["recall"] == "0.500"
    assert get[("all", "0.5", "pothole")]["missed_ids"] == "P2"
    tot = get[("all", "0.25", "pothole+crack")]
    assert (tot["n_gt"], tot["n_gt_found"], tot["n_pred"], tot["n_pred_ok"]) == ("3", "2", "3", "2")
    assert get[("all", "0.25", "guardrail_damage")]["group"] == "facility_presence"
    assert ("all", "vote", "lane_wear") in get  # 차선 마모 행 (영상이 검은색이라 차선 없음)
    assert get[("all", "vote", "lane_wear")]["n_pred"] == "0"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["videos"][0]["frames_analyzed"] == 200 and summary["public_testset"] is None


def test_eval_detector_skips_missing_gt(synth, tmp_path):
    video, _ = synth
    rc = eval_detector.main([str(video), "--gt-dir", str(tmp_path / "none"), "--out", str(tmp_path / "o")],
                            detector_factory=lambda c: pytest.fail("should not open"))
    assert rc == 1


def test_interval_sweep_cli(synth, tmp_path):
    video, gt_dir = synth
    out = tmp_path / "sweep"
    det = ScriptedDetector(SCRIPT)
    rc = interval_sweep.main([str(video), "--gt-dir", str(gt_dir), "--intervals", "1,3,5,10", "--no-lane",
                              "--out", str(out), "--cache-dir", str(tmp_path / "c")],
                             detector_factory=lambda c: det)
    assert rc == 0 and det.calls == 200  # 모든 프레임 한 번만
    rows = list(csv.DictReader((out / "sweep.csv").open(encoding="utf-8")))
    get = {(int(r["interval"]), r["defect_type"]): r for r in rows}
    # 포트홀: 간격 1·5·10 은 프레임 100 을 보므로 0 놓침, 간격 3 은 P2 를 놓침
    assert [get[(n, "pothole")]["n_missed"] for n in (1, 3, 5, 10)] == ["0", "1", "0", "0"]
    assert get[(3, "pothole")]["missed_ids"] == "P2"
    # 결함 합계(포트홀 2 + 균열 1): 균열은 늘 놓침
    assert get[(1, "pothole+crack+lane_wear")]["miss_rate"] == "0.333"
    assert get[(3, "pothole+crack+lane_wear")]["miss_rate"] == "0.667"
    assert [int(get[(n, "pothole")]["frames_analyzed"]) for n in (1, 3, 5, 10)] == [200, 67, 40, 20]
    assert get[(10, "pothole")]["analyzed_per_min"] == "60.0"
    assert get[(1, "guardrail_damage")]["group"] == "facility_presence"
