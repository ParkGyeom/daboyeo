import pytest
from pydantic import ValidationError

from daboyeo_contracts import CLASS_NAMES, DefectType, Detection


def test_class_order_is_fixed():
    # 바뀌면 기존 모델과 호환이 깨진다 — 의도한 변경이면 이 테스트도 같이 고친다
    assert CLASS_NAMES == ("pothole", "crack", "guardrail", "sign")


def test_from_class_id_maps_name_and_defect_type():
    d = Detection.from_class_id(2, conf=0.8, bbox=(0.1, 0.2, 0.3, 0.4))
    assert d.class_name == "guardrail"
    assert d.defect_type is DefectType.GUARDRAIL_DAMAGE
    assert d.area == pytest.approx(0.04)


def test_rejects_mismatched_id_and_name():
    with pytest.raises(ValidationError, match="class_id 0"):
        Detection(class_id=0, class_name="crack", conf=0.5, bbox=(0, 0, 1, 1))


@pytest.mark.parametrize(
    "bbox",
    [(0.5, 0.1, 0.4, 0.2), (0.1, 0.1, 1.5, 0.2), (0.1, 0.3, 0.2, 0.3)],
)
def test_rejects_bad_bbox(bbox):
    with pytest.raises(ValidationError):
        Detection.from_class_id(0, conf=0.5, bbox=bbox)


def test_damage_only_for_facilities():
    Detection.from_class_id(3, conf=0.5, bbox=(0, 0, 1, 1), damage="suspected", damage_conf=0.7)
    with pytest.raises(ValidationError, match="damage"):
        Detection.from_class_id(0, conf=0.5, bbox=(0, 0, 1, 1), damage="suspected")


def test_json_round_trip():
    d = Detection.from_class_id(1, conf=0.66, bbox=(0.1, 0.5, 0.4, 0.9))
    assert Detection.model_validate_json(d.model_dump_json()) == d
