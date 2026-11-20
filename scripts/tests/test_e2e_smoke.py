"""scripts/e2e_smoke.py 의 집계·판정 (DB·서버 없이)."""

from collections import Counter

from scripts.e2e_smoke import (
    Label,
    ac6_summary,
    judge,
    load_labels,
    match_label,
    parse_blackbox_output,
    record_events,
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
"""


def test_parse_blackbox_output():
    d = parse_blackbox_output(LOG)
    assert d["bb-01"]["sent"] == 2 and d["bb-01"]["failed"] == 0 and d["bb-01"]["ended"]
    assert d["bb-01"]["types"] == Counter({"crack": 1, "pothole": 1})
    assert d["bb-02"]["sent"] == 1 and d["bb-02"]["failed"] == 1 and d["bb-02"]["findings"] == 1


def test_record_events_groups_within_gap():
    lines = [
        '{"kind": "daboyeo-detections", "detect_every": 10}',
        '{"frame": "a", "t": 0.0, "dets": [["sign", 0.5, 0, 0, 1, 1], ["sign", 0.3, 2, 2, 3, 3]]}',
        '{"frame": "b", "t": 0.5, "dets": [["sign", 0.5, 0, 0, 1, 1]]}',
        '{"frame": "c", "t": 3.0, "dets": [["sign", 0.5, 0, 0, 1, 1], ["guardrail", 0.4, 0, 0, 1, 1]]}',
        '{"frame": "d", "t": 3.3, "dets": []}',
    ]
    assert record_events(lines, gap_s=1.0) == Counter({"sign": 2, "guardrail": 1})


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


def _devices(sent, failed=0, ended=True):
    return {f"bb-0{i}": {"sent": n, "failed": failed, "ended": ended, "findings": 0, "types": Counter()}
            for i, n in enumerate(sent, start=1)}


def _ok(rows):
    return all(r[4] for r in rows)


def test_judge_pass_and_failures():
    exp = ["bb-01", "bb-02", "bb-03"]
    ac6 = {"seen": 5, "labels": 10, "common": 4, "merged": 4, "ratio": 1.0}
    road = Counter({"pothole": 1, "crack": 3})
    fac = Counter({"guardrail": 2, "sign": 5})
    assert _ok(judge(_devices([3, 3, 3]), exp, 9, road, fac, ac6, 0.8))
    assert not _ok(judge(_devices([3, 3, 3]), exp, 8, road, fac, ac6, 0.8))  # AC-4: 저장 수가 다름
    assert not _ok(judge(_devices([3, 3, 3], failed=1), exp, 9, road, fac, ac6, 0.8))  # 전송 실패
    assert not _ok(judge(_devices([3, 3]), exp, 6, road, fac, ac6, 0.8))  # 기기 하나가 끝나지 않음
    assert not _ok(judge(_devices([3, 3, 3]), exp, 9, Counter({"crack": 3}), fac, ac6, 0.8))  # AC-1 포트홀 0
    assert not _ok(judge(_devices([3, 3, 3]), exp, 9, road + Counter({"sign_damage": 1}), fac, ac6, 0.8))
    assert not _ok(judge(_devices([3, 3, 3]), exp, 9, road, fac, {**ac6, "merged": 3, "ratio": 0.75}, 0.8))
    assert not _ok(judge(_devices([3, 3, 3]), exp, 9, road, fac, {**ac6, "common": 0, "ratio": None}, 0.8))
