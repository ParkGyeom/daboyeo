"""유형별 분석 주기 (PRD 8.3, 계획 2.2). 프레임마다 어떤 단계를 돌릴지 정하고, 단계별 실행 횟수·시간을 센다.

  품질 판별: 매 프레임 (quality_every). 탐지·차선 마모를 돌리는 프레임에서는 늘 먼저 잰다
  결함 탐지: detect_every 프레임마다 (품질 양호일 때만)
  차선 마모: lane_every 프레임마다 (품질 양호일 때만, 0 이면 끔). 변화가 느려 탐지보다 긴 주기를 권한다

  with sched.run("detect"):   # 횟수와 걸린 시간을 센다
      dets = detector.detect(image)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

STAGES = ("quality", "detect", "lane")
LABELS = {"quality": "품질", "detect": "탐지", "lane": "차선"}


@dataclass(frozen=True)
class Steps:
    quality: bool
    detect: bool
    lane: bool

    @property
    def any(self) -> bool:
        return self.quality or self.detect or self.lane


class Scheduler:
    def __init__(self, detect_every: int = 1, lane_every: int = 0, quality_every: int = 1, quality: bool = True):
        if detect_every < 1 or lane_every < 0 or quality_every < 1:
            raise ValueError("주기는 detect·quality 1 이상, lane 0 이상")
        self.detect_every, self.lane_every, self.quality_every = detect_every, lane_every, quality_every
        self.quality_on = quality
        self.frames = 0  # plan 을 부른 프레임 수 (위치가 있는 프레임)
        self.count = dict.fromkeys(STAGES, 0)
        self.seconds = dict.fromkeys(STAGES, 0.0)

    def plan(self, i: int) -> Steps:
        """i 번째 프레임(영상 처음부터 센 번호)에서 돌릴 단계."""
        self.frames += 1
        detect = i % self.detect_every == 0
        lane = self.lane_every > 0 and i % self.lane_every == 0
        quality = self.quality_on and (i % self.quality_every == 0 or detect or lane)
        return Steps(quality, detect, lane)

    @contextmanager
    def run(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] += time.perf_counter() - t0
            self.count[stage] += 1

    def stats(self) -> dict:
        out = {"frames": self.frames}
        for s in STAGES:
            n = self.count[s]
            out[s] = {"runs": n, "ratio": n / self.frames if self.frames else 0.0,
                      "ms_avg": self.seconds[s] / n * 1000 if n else 0.0, "s_total": self.seconds[s]}
        return out

    def summary(self) -> str:
        """로그 한 줄: 단계별 실행 비율과 평균 시간."""
        st = self.stats()
        parts = [f"{LABELS[s]} {st[s]['runs']}/{self.frames} ({st[s]['ratio']:.0%}, {st[s]['ms_avg']:.1f}ms)"
                 for s in STAGES]
        return "단계별 실행 " + " · ".join(parts)
