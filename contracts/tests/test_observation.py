from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from daboyeo_contracts import DefectType, ObservationIn, RISK_BY_TYPE


def make(**kw):
    base = dict(
        trip_id=uuid4(),
        vehicle_id=uuid4(),
        defect_type="pothole",
        confidence=0.9,
        bbox=(0.1, 0.6, 0.3, 0.8),
        lat=37.55,
        lon=126.92,
        t_s=12.5,
        observed_at=datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc),
    )
    return ObservationIn(**(base | kw))


def test_valid_observation_gets_client_id():
    a, b = make(), make()
    assert a.defect_type is DefectType.POTHOLE
    assert a.client_observation_id != b.client_observation_id


def test_lane_wear_without_bbox():
    assert make(defect_type="lane_wear", bbox=None).bbox is None


@pytest.mark.parametrize("field,value", [("confidence", 1.2), ("lat", 91), ("t_s", -1), ("defect_type", "tree")])
def test_rejects_invalid(field, value):
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_every_defect_type_has_risk():
    assert set(RISK_BY_TYPE) == set(DefectType)
