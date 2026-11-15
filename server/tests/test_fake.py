import random

from server.fake import make_events, offset_m
from server.schema import DetectionEvent


def test_events_follow_schema_and_counts():
    evs = make_events(37.55, 127.07, defects=4, devices=3, gps_noise=7, radius=300,
                      types=["pothole", "crack"], rng=random.Random(1))
    assert len(evs) == 12
    for ev, photo in evs:
        DetectionEvent.model_validate(ev)
        assert photo.startswith(b"\xff\xd8")
    assert {e["device_id"] for e, _ in evs} == {"bb-01", "bb-02", "bb-03"}


def test_offset_is_about_right():
    import math
    lat, lon = offset_m(37.55, 127.07, 100, 0)
    assert abs((lat - 37.55) * 111_320 - 100) < 0.01
    lat2, lon2 = offset_m(37.55, 127.07, 0, 100)
    assert abs((lon2 - 127.07) * 111_320 * math.cos(math.radians(37.55)) - 100) < 0.01
