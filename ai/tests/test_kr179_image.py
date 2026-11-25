"""kr179_image — Wilson 구간, 촬영 영상 이름, 구성·기저 비율, 크기별 재현율, 겹침 부분집합을 가짜 데이터로 시험한다."""

import numpy as np
import pytest
from daboyeo_ai.eval.kr179_image import (
    clip_of,
    composition,
    descriptor,
    precision_at,
    recall_by_size,
    report,
    subsets,
    wilson,
)

CONF = {"pothole": 0.31, "crack": 0.42}
NAME = "ro_V3F_HY_{:04d}_20201217_101520_N_CH1_Seoul_Sun_Mainroad_Day_65208"


def _im(name, gts=(), preds=()):
    return {"name": name, "w": 640, "h": 360, "gts": [{"cls": c, "box": b} for c, b in gts],
            "preds": [{"cls": c, "score": s, "box": b} for c, s, b in preds]}


def test_wilson_known_values():
    assert wilson(0, 0) is None
    assert wilson(5, 10) == [0.237, 0.763]
    assert wilson(10, 10) == [0.722, 1.0]


def test_clip_of_ignores_random_id_and_channel():
    assert clip_of(NAME.format(1)) == clip_of(NAME.format(2).replace("CH1", "CH0")) == "20201217_101520"
    with pytest.raises(ValueError):
        clip_of("rdd_Japan_000001")


def test_descriptor_is_normalized_correlation():
    g = np.random.default_rng(0).integers(0, 200, (360, 640), dtype=np.uint8)
    assert float(descriptor(g) @ descriptor(g)) == pytest.approx(1.0, abs=1e-3)
    assert abs(float(descriptor(g) @ descriptor(np.random.default_rng(1).integers(0, 200, (360, 640), dtype=np.uint8)))) < 0.2


def test_report_base_rate_and_false_alarm():
    b = [100, 100, 140, 140]
    ims = [_im(NAME.format(1), [("pothole", b)], [("pothole", 0.5, b)]),
           _im(NAME.format(2), [("pothole", b), ("crack", b)], [("pothole", 0.30, b)]),  # 기준 아래 → 놓침
           _im(NAME.format(3), [], [("pothole", 0.9, b)]),
           _im(NAME.format(4))]
    assert composition(ims) == {"images": 4, "pothole_images": 2, "crack_images": 1, "both": 1, "neither": 2,
                                "base_rate": {"pothole": 0.5, "crack": 0.25}}
    i = report(ims, CONF)["pothole"]["image"]
    assert (i["recall"], i["precision"], i["base_rate"]) == (0.5, 0.5, 0.5)
    assert (i["false_alarm_images"], i["negative_images"], i["false_alarm_rate"]) == (1, 2, 0.5)


def test_precision_at_follows_base_rate():
    assert precision_at(0.5, 0.5, 0.3) == 0.3  # 아무 사진에나 경보 → 정밀도 = 기저 비율
    assert precision_at(0.6, 0.0, 0.1) == 1.0
    assert precision_at(0.0, 0.0, 0.1) is None


def test_recall_by_size_uses_long_side_640():
    small, big = [0, 0, 10, 10], [100, 100, 200, 200]
    r = recall_by_size([_im("a", [("pothole", small), ("pothole", big)], [("pothole", 0.5, big)])], CONF)
    assert (r["<16px"]["boxes"], r["<16px"]["found"], r[">=32px"]["boxes"], r[">=32px"]["found"]) == (1, 0, 1, 1)
    assert (r[">=32px"]["images"], r[">=32px"]["hit_images"], r["<16px"]["images"]) == (1, 1, 0)
    im = {**_im("a", [("pothole", [0, 0, 40, 40])]), "w": 1280, "h": 720}  # 640 으로 줄이면 20 px
    assert recall_by_size([im], CONF)["16~32px"]["boxes"] == 1


def test_subsets_drop_shared_clips_and_near_dups():
    a, b = NAME.format(1), NAME.format(2).replace("101520", "111111")
    s = subsets([_im(a), _im(b)], {"20201217_101520"}, {a: 0.5, b: 0.95})
    names = {k: [im["name"] for im in v] for k, v in s.items()}
    assert names == {"all": [a, b], "no_shared_clip": [b], "no_near_dup": [a], "near_dup_only": [b]}
