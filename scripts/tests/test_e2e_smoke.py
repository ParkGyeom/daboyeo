"""scripts/e2e_smoke.py 의 집계·판정 (DB·서버 없이)."""

from collections import Counter

from scripts.e2e_smoke import (
    Label,
    ac6_summary,
    cross_video_defects,
    device_ids,
    facility_tracks,
    judge,
    load_labels,
    match_label,
    min_track_distance_m,
    parse_blackbox_output,
    prefix_line,
)

LOG = """\
[bb-01] 탐지기 replay:detector_v3.onnx+tiles2, 기록 r.jsonl
[bb-01]  gn_b@5.00s: crack 0.55 (탐지 3회 합침, 0.7초) → 결함 1a2b3c4d (새 결함)
[bb-02]  gn_b@5.17s: crack 0.51 (탐지 2회 합침, 0.3초) → 결함 1a2b3c4d
[bb-01]  gn_b@99.2s: pothole 0.40 (탐지 1회 합침, 0.0초) → 결함 9f9f9f9f (새 결함)
[bb-01] 주행 종료: 시민에게 보여줄 결함 2건
[bb-02] 주행 종료: 시민에게 보여줄 결함 1건
[bb-01] 끝. 탐지한 프레임 10, 탐지 4회 → 트랙 2개, 보낸 관측 2건 (1.0KB: 메타 0.1 · 사진 0.9 · 클립 0개 0.0), 전송 실패 0, 위치 없음 0
[bb-02] 끝. 탐지한 프레임 10, 탐지 2회 → 트랙 1개, 보낸 관측 1건 (1.0KB: 메타 0.1 · 사진 0.9 · 클립 0개 0.0), 전송 실패 1, 위치 없음 0
[bb-01] 보내지 않은 탐지 guardrail 25 · sign 136 (트랙, 합치기 1초)
[bb-02] 보내지 않은 탐지 없음 (트랙, 합치기 1초)
"""


def test_parse_blackbox_output():
    d = parse_blackbox_output(LOG)
    assert d["bb-01"]["sent"] == 2 and d["bb-01"]["failed"] == 0 and d["bb-01"]["ended"]
    assert d["bb-01"]["types"] == Counter({"crack": 1, "pothole": 1})
    assert d["bb-02"]["sent"] == 1 and d["bb-02"]["failed"] == 1 and d["bb-02"]["findings"] == 1
    assert d["bb-01"]["unsent"] == Counter({"guardrail": 25, "sign": 136})
    assert d["bb-02"]["unsent"] == Counter()


def test_facility_tracks_first_device_per_video():
    d = parse_blackbox_output(LOG + "[bb-04] 보내지 않은 탐지 sign 3 (트랙, 합치기 1초)\n")
    assert facility_tracks(d, ["bb-01", "bb-04"]) == (Counter({"guardrail": 25, "sign": 139}), [])
    assert facility_tracks(d, ["bb-01", "bb-05"]) == (Counter({"guardrail": 25, "sign": 136}), ["bb-05"])


CSV = """# 주석
defect_id,t_start_s,t_end_s,defect_type,memo
C1,5.0,8.0,crack,균열
C2,9.0,14.5,crack,균열
P1,99.0,99.5,pothole,포트홀
C3,104.5,105.5,crack,amb 작은 균열
"""


def test_load_and_match_labels():
    labels = load_labels(CSV)
    assert [lb.id for lb in labels] == ["C1", "C2", "P1", "C3"]
    assert labels[3].amb and not labels[0].amb
    assert match_label("crack", 8.4, labels) == "C1"  # 끝 + 0.5 안
    assert match_label("crack", 8.7, labels) == "C2"  # 두 라벨 창에 걸치면 가운데가 가까운 쪽
    assert match_label("pothole", 6.0, labels) is None  # 유형이 다르면 안 맞음
    assert match_label("crack", 50.0, labels) is None


def test_ac6_summary():
    labels = [Label("C1", 5, 8, "crack", False), Label("C2", 20, 22, "crack", False),
              Label("P1", 99, 99.5, "pothole", False), Label("A1", 50, 51, "crack", True)]
    obs = [
        ("bb-01", "crack", 6.0, "d1"), ("bb-02", "crack", 6.2, "d1"), ("bb-03", "crack", 6.1, "d1"),  # 묶임
        ("bb-01", "crack", 21.0, "d2"), ("bb-02", "crack", 21.0, "d3"),  # 나뉨
        ("bb-01", "pothole", 99.2, "d4"),  # 1대만
        ("bb-02", "crack", 50.5, "d5"),  # amb
        ("bb-03", "crack", 70.0, "d6"),  # 헛것
    ]
    s = ac6_summary(obs, labels)
    assert (s["seen"], s["common"], s["merged"]) == (3, 2, 1)
    assert s["ratio"] == 0.5
    assert s["split"] == {"C2": 2}
    assert (s["unmatched_obs"], s["amb_obs"]) == (1, 1)


def test_device_ids_and_prefix():
    assert device_ids(2) == [["bb-01", "bb-02", "bb-03"], ["bb-04", "bb-05", "bb-06"]]
    assert prefix_line("  f: crack 0.5\n", "bb-04") == "[bb-04]  f: crack 0.5\n"
    assert prefix_line("[bb-04] 끝.\n", "bb-04") == "[bb-04] 끝.\n"


def test_cross_video_defects():
    labels = {"a": [Label("A1", 0, 2, "crack", False)], "b": [Label("B1", 10, 12, "crack", False)]}
    obs = {"a": [("bb-01", "crack", 1.0, "d1"), ("bb-02", "crack", 1.0, "d2")],
           "b": [("bb-04", "crack", 11.0, "d1"), ("bb-05", "crack", 50.0, "d3")]}
    assert cross_video_defects(obs, labels) == {"d1": {"a": ["A1"], "b": ["B1"]}}


def test_min_track_distance():
    a = [(37.5000, 127.0), (37.5010, 127.0)]
    b = [(37.5013, 127.0), (37.6, 127.0)]
    assert 30 < min_track_distance_m(a, b) < 37  # 위도 0.0003도 ≈ 33 m


def _devices(sent, failed=0, ended=True):
    return {f"bb-{i:02d}": {"sent": n, "failed": failed, "ended": ended, "findings": 0, "types": Counter()}
            for i, n in enumerate(sent, start=1)}


def _ok(rows):
    return all(r[4] for r in rows)


def test_judge_pass_and_failures():
    exp = ["bb-01", "bb-02", "bb-03", "bb-04", "bb-05", "bb-06"]
    six = [2, 2, 2, 1, 1, 1]
    ac6 = {"a": {"seen": 5, "labels": 10, "common": 4, "merged": 3, "ratio": 0.75},
           "b": {"seen": 2, "labels": 3, "common": 2, "merged": 1, "ratio": 0.5}}
    road = Counter({"pothole": 1, "crack": 3})
    fac = Counter({"guardrail": 2, "sign": 5})
    assert _ok(judge(_devices(six), exp, 9, road, fac, ac6, 0.5))  # 4/6 = 67%
    assert not _ok(judge(_devices(six), exp, 9, road, fac, ac6, 0.7))
    assert not _ok(judge(_devices(six), exp, 8, road, fac, ac6, 0.5))  # AC-4: 저장 수가 다름
    assert not _ok(judge(_devices(six, failed=1), exp, 9, road, fac, ac6, 0.5))  # 전송 실패
    assert not _ok(judge(_devices(six[:5]), exp, 8, road, fac, ac6, 0.5))  # 기기 하나가 끝나지 않음
    assert not _ok(judge(_devices(six), exp, 9, Counter({"crack": 3}), fac, ac6, 0.5))  # AC-1 포트홀 0
    assert not _ok(judge(_devices(six), exp, 9, road + Counter({"sign_damage": 1}), fac, ac6, 0.5))
    none = {"a": {**ac6["a"], "common": 0, "merged": 0, "ratio": None}}
    assert not _ok(judge(_devices(six), exp, 9, road, fac, none, 0.5))
    cross = {"d1": {"a": ["A1"], "b": ["B1"]}}
    assert _ok(judge(_devices(six), exp, 9, road, fac, ac6, 0.5, cross, tracks_overlap=True))  # 겹치면 판정 안 함
    assert not _ok(judge(_devices(six), exp, 9, road, fac, ac6, 0.5, cross, tracks_overlap=False))
    assert not _ok(judge(_devices(six), exp, 9, road, fac, ac6, 0.5, facility_missing=["bb-04"]))
