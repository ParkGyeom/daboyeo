"""차선 마모 개선 후보 (handoff §17-6) — 시연 클립에서 `lane_wear_demo` 도구로 비교하려고 만든 변형들.

`lane_wear.py` 의 기본 동작은 바꾸지 않는다. 여기서 고른 것만 박겸 확인 뒤 `lane_wear.py` 로 옮긴다.

1. 정지 판정 `road_motion`: 노면 영역(작게 줄인 회색)을 약 0.5초 전 프레임과 비교한 차이의 중앙값.
   차가 서 있으면 노면 무늬가 그대로라 작다 (지나가는 사람은 중앙값에 거의 안 들어간다). 작으면 판정·투표에 안 넣는다.
2. 횡단보도 `crosswalk_drop`: 행마다 선분이 지나는 x 를 모아, 같은 선 조각(간격 < same·d)은 하나로 묶고
   간격 < k·d 로 이어진 서로 다른 선이 min_lines 개 이상이면 그 선분들을 뺀다 (d = 그 행의 차선 폭 상한).
   차선 사이 간격(차로 폭)은 d 의 10배 이상이지만, 횡단보도 줄무늬 간격은 2~4배다.
3. 점수 `VariantTracker(mode=...)`: paint_len 을 빼거나 같은 주행의 보통 paint_len 으로 나눈다.
   + 차선 찾은 비율 보호 `min_found`: 최근 측정 프레임 중 차선 찾은 비율이 낮으면 판정하지 않는다.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from daboyeo_ai.lane_wear import HORIZON_Y, LaneWearMeter, LaneWearResult, LaneWearTracker, WearJudgement

MOTION_LAG_S = 0.5
MOTION_SIZE = (160, 90)
MOTION_ROAD = (0.55, 0.9, 0.2, 0.8)  # 노면 영역 (y0, y1, x0, x1 비율)


def road_small(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame_bgr, MOTION_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    y0, y1, x0, x1 = MOTION_ROAD
    return g[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].astype(np.int16)


def road_motion(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(np.abs(a - b)))


class MotionProbe:
    """프레임마다 부르면 약 lag_s 전 프레임과의 노면 차이를 돌려준다 (처음 lag_s 동안은 None)."""

    def __init__(self, lag_s: float = MOTION_LAG_S):
        self.lag_s = lag_s
        self._buf: deque[tuple[float, np.ndarray]] = deque()

    def update(self, frame_bgr: np.ndarray, t: float) -> float | None:
        cur = road_small(frame_bgr)
        while len(self._buf) >= 2 and self._buf[1][0] <= t - self.lag_s:
            self._buf.popleft()
        out = road_motion(cur, self._buf[0][1]) if self._buf and self._buf[0][0] <= t - self.lag_s else None
        self._buf.append((t, cur))
        return out


def stripe_w(y: float, w: int, h: int, max_stripe: float = 0.035) -> float:
    return max(2.0, max_stripe * w * (y - HORIZON_Y * h) / ((1 - HORIZON_Y) * h))


def crosswalk_drop(segs, w: int, h: int, k: float = 4.0, min_lines: int = 3, same: float = 1.5,
                   nrows: int = 12) -> set[int]:
    """segs: 프레임(또는 작업 해상도) 좌표 (x1,y1,x2,y2). 횡단보도 줄무늬로 볼 선분 번호."""
    if len(segs) < min_lines:
        return set()
    ys = [min(s[1], s[3]) for s in segs] + [max(s[1], s[3]) for s in segs]
    drop: set[int] = set()
    for y in np.linspace(min(ys), max(ys), nrows):
        xs = sorted((xa + (y - ya) * (xb - xa) / (yb - ya), i) for i, (xa, ya, xb, yb) in enumerate(segs)
                    if ya != yb and min(ya, yb) <= y <= max(ya, yb))
        if len(xs) < min_lines:
            continue
        d = stripe_w(y, w, h)
        lines = [[xs[0]]]
        for x, i in xs[1:]:
            if x - lines[-1][-1][0] < same * d:
                lines[-1].append((x, i))
            else:
                lines.append([(x, i)])
        run = [lines[0]]
        for ln in lines[1:] + [None]:
            if ln is not None and ln[0][0] - run[-1][-1][0] < k * d:
                run.append(ln)
                continue
            if len(run) >= min_lines:
                drop.update(i for r in run for _, i in r)
            run = [ln] if ln is not None else []
    return drop


def measure_without_crosswalk(meter: LaneWearMeter, frame_bgr: np.ndarray, parts: dict | None = None,
                              **kw) -> tuple[LaneWearResult, list, int]:
    """횡단보도 줄무늬 선분을 뺀 뒤 측정. (결과, 원본 좌표 선분, 뺀 선분 수)"""
    p = parts if parts is not None else meter._segments(frame_bgr)
    (x0, y0), (sw, sh) = p["origin"], p["size"]
    full = [(xa + x0, ya + y0, xb + x0, yb + y0) for xa, ya, xb, yb in p["segs"]]
    drop = crosswalk_drop(full, sw, sh, **kw)
    if drop:
        keep = [i for i in range(len(full)) if i not in drop]
        p = p | {key: [p[key][i] for i in keep] for key in ("segs", "lane_vals", "road_vals", "row_vals")}
    r, segs = meter._finish(p)
    return r, segs, len(drop)


MODES = ("p25_x_len", "p25_only", "p25_x_rel_len", "p25_x_rel_len_cap", "p25_x_sqrt_rel_len")


class VariantTracker(LaneWearTracker):
    """LaneWearTracker 와 같은 투표. 점수 식(mode)과 차선 찾은 비율 보호(min_found)만 다르다.

    - p25_x_len: 지금 식 (contrast_p25 ÷ 기준) × paint_len
    - p25_only: contrast_p25 ÷ 기준
    - p25_x_rel_len: (contrast_p25 ÷ 기준) × (paint_len ÷ 주행 중앙 paint_len)
    - p25_x_rel_len_cap: 위와 같되 길이 비율을 1 로 자름 (길면 더하지 않음, 짧을 때만 깎음)
    - p25_x_sqrt_rel_len: 길이 비율의 제곱근
    min_found: 최근 found_window 개 측정(차선 못 찾은 프레임 포함) 중 차선 찾은 비율이 이보다 낮으면 판정·투표 안 함.
    """

    def __init__(self, mode: str = "p25_x_len", min_found: float = 0.0, found_window: int = 100, **kw):
        super().__init__(**kw)
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.min_found = min_found
        self._found: deque[bool] = deque(maxlen=found_window)
        self._paint: deque[float] = deque(maxlen=self.window)

    def score(self, r: LaneWearResult, baseline: float) -> float:
        rel = float(r.contrast_p25) / max(baseline, 1e-3)
        if self.mode == "p25_x_len":
            return rel * r.paint_len
        if self.mode == "p25_only":
            return rel
        lr = r.paint_len / max(float(np.median(self._paint)) if self._paint else r.paint_len, 1e-3)
        if self.mode == "p25_x_rel_len_cap":
            lr = min(1.0, lr)
        elif self.mode == "p25_x_sqrt_rel_len":
            lr = float(np.sqrt(lr))
        return rel * lr

    def update(self, r: LaneWearResult, t: float | None = None) -> WearJudgement:
        self._found.append(bool(r.lane_found and r.contrast_p25 is not None))
        if self._found[-1] and self.min_found > 0 and len(self._found) >= 20 \
                and sum(self._found) / len(self._found) < self.min_found:
            self._recent.append(float(r.contrast_p25))   # 기준은 계속 쌓되 판정하지 않는다
            self._paint.append(float(r.paint_len))
            return WearJudgement(lane_found=True, ready=False, baseline=self.baseline, score=None, is_worn=False,
                                 confidence=0.0, **self._vote_fields())
        j = super().update(r, t)
        if j.lane_found:
            self._paint.append(float(r.paint_len))
        return j
