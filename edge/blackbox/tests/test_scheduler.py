"""edge/blackbox/scheduler.py: 유형별 분석 주기 (PRD 8.3, 계획 2.2)."""

import pytest

from edge.blackbox.scheduler import Scheduler


def test_quality_every_frame_detect_n_lane_m():
    s = Scheduler(detect_every=5, lane_every=15)
    steps = [s.plan(i) for i in range(30)]
    assert all(st.quality for st in steps)
    assert [i for i, st in enumerate(steps) if st.detect] == [0, 5, 10, 15, 20, 25]
    assert [i for i, st in enumerate(steps) if st.lane] == [0, 15]


def test_quality_always_before_detect_even_with_longer_quality_period():
    s = Scheduler(detect_every=3, lane_every=0, quality_every=4)
    steps = [s.plan(i) for i in range(12)]
    assert [i for i, st in enumerate(steps) if st.quality] == [0, 3, 4, 6, 8, 9]
    assert not any(st.lane for st in steps)


def test_no_quality_and_stage_ratio():
    s = Scheduler(detect_every=2, lane_every=0, quality=False)
    for i in range(10):
        st = s.plan(i)
        assert not st.quality
        if st.detect:
            with s.run("detect"):
                pass
    stats = s.stats()
    assert stats["frames"] == 10 and stats["detect"]["runs"] == 5 and stats["detect"]["ratio"] == 0.5
    assert "탐지 5/10 (50%" in s.summary()


def test_bad_period():
    with pytest.raises(ValueError):
        Scheduler(detect_every=0)
