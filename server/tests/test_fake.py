import random
import uuid
from datetime import datetime, timedelta, timezone

from server.fake import make_plan, observation, offset_m


def test_plan_every_vehicle_sees_every_defect():
    plan = make_plan(37.5, 127.0, defects=4, devices=3, gps_noise=5, radius=300,
                     types=["pothole", "lane_wear"], rng=random.Random(1))
    assert list(plan) == ["bb-01", "bb-02", "bb-03"]
    assert all(len(items) == 4 for items in plan.values())
    started = datetime(2026, 11, 20, tzinfo=timezone(timedelta(hours=9)))
    for items in plan.values():
        for it in items:
            obs = observation(it, uuid.uuid4(), uuid.uuid4(), started)  # contracts 검증을 통과한다
            assert (obs.bbox is None) == (it["defect_type"] == "lane_wear")


def test_offset_is_about_right():
    lat, lon = offset_m(37.5, 127.0, 100, 100)
    assert abs((lat - 37.5) * 111_320 - 100) < 0.1
