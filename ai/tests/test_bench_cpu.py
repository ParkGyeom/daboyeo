"""bench_cpu: 계산·로그 요약만 시험한다 (시간 측정은 장비마다 달라 CI 에서 재지 않는다)."""

import json

import pytest
from daboyeo_ai.eval.bench_cpu import (
    compute_s_per_video_min,
    interval_columns,
    main,
    read_log,
    stats_ms,
    summarize_log,
)

# 계획 4장 측정 로그 형식. 30 fps, 6 프레임(0.2 초). quality 는 매 프레임, detect 는 3 프레임마다.
FIXTURE = """
{"frame":0,"t":0.0,"stage":"quality","ran":true,"ms":2.0,"result":"good"}
{"frame":0,"t":0.0,"stage":"detect","ran":true,"ms":100.0,"n":2}
{"frame":1,"t":0.0333,"stage":"quality","ran":true,"ms":2.0,"result":"good"}
{"frame":1,"t":0.0333,"stage":"detect","ran":false}
{"frame":2,"t":0.0667,"stage":"quality","ran":true,"ms":2.0,"result":"good"}
{"frame":2,"t":0.0667,"stage":"detect","ran":false}

{"frame":3,"t":0.1,"stage":"quality","ran":true,"ms":2.0,"result":"blur"}
{"frame":3,"t":0.1,"stage":"detect","ran":true,"ms":120.0,"n":0}
{"frame":4,"t":0.1333,"stage":"quality","ran":true,"ms":2.0,"result":"good"}
{"frame":4,"t":0.1333,"stage":"detect","ran":false}
{"frame":5,"t":0.1667,"stage":"quality","ran":true,"ms":2.0,"result":"good"}
{"frame":5,"t":0.1667,"stage":"detect","ran":false}
"""


def test_stats_ms():
    s = stats_ms([10, 10, 10, 30])
    assert s["n"] == 4 and s["median_ms"] == 10 and s["mean_ms"] == 15
    assert s["fps"] == pytest.approx(1000 / 15)
    assert stats_ms([])["fps"] is None


def test_compute_per_video_minute():
    # 30 fps × 60 s = 1800 프레임. 100 ms 를 매 프레임 → 180 s (실시간의 3배)
    assert compute_s_per_video_min(100, 1) == pytest.approx(180)
    assert compute_s_per_video_min(100, 3) == pytest.approx(60)
    assert compute_s_per_video_min(100, 10) == pytest.approx(18)
    # 나누어떨어지지 않으면 올림 (0, 7, 14, ... 번째 프레임을 분석)
    assert compute_s_per_video_min(1000, 7) == pytest.approx(258)
    with pytest.raises(ValueError):
        compute_s_per_video_min(100, 0)
    cols = interval_columns(100, intervals=(1, 5))
    assert cols == {"s_per_min_i1": 180.0, "rt_i1": 3.0, "s_per_min_i5": 36.0, "rt_i5": 0.6}
    assert interval_columns(None, intervals=(3,)) == {"s_per_min_i3": None, "rt_i3": None}


def test_read_log_skips_blank_and_rejects_bad():
    rows = read_log(FIXTURE.splitlines())
    assert len(rows) == 12
    with pytest.raises(ValueError):
        read_log(['{"t": 1.0, "ms": 3}'])


def test_summarize_log():
    out = {r["stage"]: r for r in summarize_log(read_log(FIXTURE.splitlines()))}
    det, q, total = out["detect"], out["quality"], out["total"]
    assert det["n_frames"] == 6 and det["n_ran"] == 2 and det["ran_ratio"] == pytest.approx(0.333)
    assert det["median_ms"] == 110 and det["mean_ms"] == 110
    assert det["fps"] == pytest.approx(2 / 0.22, abs=0.01)
    # 영상 길이 = t 범위 0.1667 + 간격 0.0333 = 0.2 초 → 계산 0.22 초 ÷ (0.2/60) 분
    assert det["video_s"] == pytest.approx(0.2)
    assert det["s_per_video_min"] == pytest.approx(0.22 / (0.2 / 60), rel=1e-3)
    assert q["n_ran"] == 6 and q["median_ms"] == 2
    assert total["n_ran"] == 8 and total["frames_per_s"] == pytest.approx(6 / 0.232, abs=0.01)


def test_summarize_log_with_video_fps_and_missing_ran():
    rows = [{"frame": i, "stage": "detect", "ms": 50.0} for i in range(0, 30, 5)]  # ran 없음 → ms 있으면 실행
    rows += [{"frame": i, "stage": "detect"} for i in range(30) if i % 5]
    out = {r["stage"]: r for r in summarize_log(rows, video_fps=30)}
    assert out["detect"]["n_ran"] == 6 and out["detect"]["video_s"] == 1.0
    assert out["detect"]["s_per_video_min"] == pytest.approx(0.3 * 60)
    with pytest.raises(ValueError):
        summarize_log(rows)  # t 도 fps 도 없음


def test_log_cli_before_after(tmp_path):
    before = tmp_path / "before.jsonl"
    before.write_text("\n".join(json.dumps({"frame": i, "t": i / 30, "stage": "detect", "ran": True, "ms": 100.0})
                                for i in range(30)))
    after = tmp_path / "after.jsonl"
    after.write_text(FIXTURE)
    out = tmp_path / "log.csv"
    assert main(["log", f"전={before}", str(after), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("label,stage,n_frames,n_ran")
    labels = {ln.split(",")[0] for ln in lines[1:]}
    assert labels == {"전", "after"}
