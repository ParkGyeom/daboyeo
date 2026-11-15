"""중복 정리 (B3 완료 기준). DB가 필요하다."""

from server import db
from server.tests.conftest import event, needs_db, north, post

pytestmark = needs_db


def _defects():
    with db.connect() as conn:
        return conn.execute(f"SELECT {db.DEFECT_COLS} FROM defects d ORDER BY d.id").fetchall()


def test_same_place_same_type_three_devices_become_one(client):
    lat = 37.55
    ids = []
    for dev, dm, conf in [("bb-01", 0, 0.5), ("bb-02", 6, 0.9), ("bb-03", -5, 0.7)]:
        r = post(client, event(lat=north(lat, dm), device=dev, confidence=conf)).json()
        ids.append((r["defect_id"], r["is_new_defect"]))
    assert ids[0][1] is True and ids[1][1] is False and ids[2][1] is False
    assert len({i for i, _ in ids}) == 1
    (d,) = _defects()
    assert d["observation_count"] == 3 and d["device_count"] == 3
    assert abs(d["max_confidence"] - 0.9) < 1e-6
    # 위치는 세 관측의 평균 (북쪽으로 +1/3 m)
    assert abs((d["lat"] - lat) * 111_320 - 1 / 3) < 0.05
    # 대표 사진은 신뢰도가 가장 높은 관측의 사진
    with db.connect() as conn:
        best = conn.execute("SELECT snapshot_path FROM observations ORDER BY confidence DESC LIMIT 1").fetchone()
    assert d["best_snapshot_path"] == best["snapshot_path"]


def test_same_device_twice_counts_one_device(client):
    post(client, event(device="bb-01"))
    post(client, event(lat=north(37.55, 8), device="bb-01", at="2026-11-14T10:00:01+09:00"))
    (d,) = _defects()
    assert d["observation_count"] == 2 and d["device_count"] == 1
    assert d["first_seen"] < d["last_seen"]


def test_different_type_not_merged(client):
    post(client, event(type="pothole"))
    post(client, event(type="crack"))
    assert len(_defects()) == 2


def test_outside_distance_not_merged(client):
    post(client, event(lat=37.55))
    post(client, event(lat=north(37.55, 25)))  # 포트홀 기준 20m 밖
    assert len(_defects()) == 2


def test_lane_wear_uses_wider_distance(client):
    post(client, event(type="lane_wear", lat=37.55))
    post(client, event(type="lane_wear", lat=north(37.55, 25)))  # 차선 마모 기준 30m 안
    assert len(_defects()) == 1


def test_new_defect_gets_severity_reward_jurisdiction(client):
    post(client, event(type="pothole", lat=37.5154, lon=126.9521))  # 서울
    post(client, event(type="crack", lat=36.7850, lon=127.1736))     # 서울 밖
    seoul, outside = _defects()
    assert (seoul["severity"], seoul["reward_eligible"], seoul["jurisdiction"]) == ("빠른 보수", True, "서울특별시")
    assert (outside["severity"], outside["reward_eligible"]) == ("정기 보수", False)
    assert seoul["status"] == "미신고"
