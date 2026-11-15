"""중복 정리 (계획 2.4): 같은 지점·같은 유형은 결함 하나, 차량 수는 서로 다른 차량 수. DB가 필요하다."""

from server import db
from server.tests.conftest import meta, needs_db, north, observe, post, start

pytestmark = needs_db
LAT = 37.4979


def _defects():
    with db.connect() as conn:
        return conn.execute(
            "SELECT defect_type::text AS t, observation_count, vehicle_count, confidence, risk_level::text AS risk,"
            " reward_eligible_road FROM defects ORDER BY first_seen_at"
        ).fetchall()


def test_three_vehicles_same_place_become_one(client):
    ids = set()
    for i, label in enumerate(["bb-01", "bb-02", "bb-03"]):
        t = start(client, label)
        ids.add(observe(client, t, lat=north(LAT, i * 5), confidence=0.5 + i * 0.1)["defect_id"])
    assert len(ids) == 1
    (d,) = _defects()
    assert (d["observation_count"], d["vehicle_count"]) == (3, 3)
    assert abs(d["confidence"] - 0.7) < 1e-6  # 대표 신뢰도는 최고값


def test_same_vehicle_twice_counts_one_vehicle(client):
    t = start(client)
    observe(client, t, t_s=1)
    observe(client, t, t_s=2)
    (d,) = _defects()
    assert (d["observation_count"], d["vehicle_count"]) == (2, 1)


def test_different_type_or_far_not_merged(client):
    t = start(client)
    a = observe(client, t)["defect_id"]
    b = observe(client, t, type="crack")["defect_id"]
    c = observe(client, t, lat=north(LAT, 50))["defect_id"]
    assert len({a, b, c}) == 3


def test_lane_wear_uses_wider_distance(client):
    t = start(client)
    a = observe(client, t, type="lane_wear")["defect_id"]
    b = observe(client, t, type="lane_wear", lat=north(LAT, 25))["defect_id"]  # 점 결함 기준(20m)보다 멀다
    assert a == b


def test_new_defect_gets_risk_and_reward(client):
    t = start(client)
    observe(client, t)                                         # 강남역: 서울
    observe(client, t, type="crack", lat=36.78, lon=127.17)    # 서울 밖
    p, c = _defects()
    assert (p["t"], p["risk"], p["reward_eligible_road"]) == ("pothole", "urgent", True)
    assert (c["t"], c["risk"], c["reward_eligible_road"]) == ("crack", "routine", False)


def test_resend_same_client_id_is_not_saved_twice(client):
    t = start(client)
    m = meta(t)
    first = post(client, m)
    assert first.status_code == 201 and first.json()["duplicate"] is False
    again = post(client, m)
    assert again.status_code == 201 and again.json()["duplicate"] is True
    assert again.json()["observation_id"] == first.json()["observation_id"]
    (d,) = _defects()
    assert d["observation_count"] == 1
