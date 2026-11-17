"""연속 프레임 합치기 (계획 2.3). 같은 결함이 여러 프레임에 보여도 서버로는 관측 1건만 보낸다.

같은 유형의 박스가 gap_s 초 안에 다시 나오고 앞 박스를 '이어 가면' 같은 트랙으로 본다.
트랙이 끝나면(gap_s 초 동안 안 보이거나, max_age_s 초를 넘기거나, 주행이 끝나면)
그 트랙에서 신뢰도가 가장 높은 프레임 1장으로 관측 1건을 만든다.

'이어 간다'의 기준: IoU > iou_min (계획의 0.3) 이거나, 아래 움직임 문(gate)을 통과하면.
IoU 만 쓰지 않는 이유: 노면 결함은 땅에 붙어 있어서, 차가 결함 길이(L, 포트홀 0.3~0.6 m)보다 많이
나아가면 앞뒤 탐지 박스가 세로로 전혀 겹치지 않는다. 핀홀 카메라에서 거리 Z 의 땅 위 점은
지평선 아래 f·h/Z 에 찍히므로, 박스 [Z, Z+L] 와 d 만큼 다가간 뒤의 박스 [Z-d, Z+L-d] 는 d < L 일 때만 겹친다.
50 km/h · 30 fps · 10 프레임마다 탐지면 d ≈ 4.6 m 라 IoU 는 거의 늘 0 이다 (수치는 docs/plan.md 2.3).
그래서 박스가 아래(또는 옆)로 내려가며 커지는 움직임을 허용한다:
  새 박스 중심이 앞 박스 중심에서 가로 ±gate·S, 세로 -0.5·S ~ +gate·S 안 (S = 두 박스 중 가장 긴 변).
옆으로 나란한 두 결함은 거리와 상관없이 '간격 ÷ 결함 폭' 비율로 보이므로, 폭의 gate 배(2.5)보다 멀리
떨어진 결함은 따로 남는다. 더 가까운 같은 유형 결함이 하나로 합쳐져도 서버가 20 m 안의 같은 유형을
어차피 한 결함으로 묶으므로(계획 2.4) 결함 수는 달라지지 않는다.

max_age_s: 긴 균열처럼 몇 초 동안 계속 보이는 것이 관측 1건으로 뭉개지지 않게 트랙 길이를 자른다.
3 초는 50 km/h 로 약 42 m, 서버 묶기 반경(20 m)의 두 배 정도다.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

Box = tuple[float, float, float, float]

IOU_MIN = 0.3
GAP_S = 1.0  # 이만큼 안 보이면 트랙이 끝난 것으로 본다 (10 프레임마다 탐지 = 0.33 초, 두 번 놓쳐도 이어진다)
MAX_AGE_S = 3.0
GATE = 2.5  # 움직임 문의 크기 (박스 긴 변의 배수)
UP_SLACK = 0.5  # 위로 움직이는 것은 박스 흔들림 정도만 허용


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def follow_score(prev: Box, new: Box, iou_min: float = IOU_MIN, gate: float = GATE) -> float | None:
    """new 가 prev 를 이어 가면 점수(클수록 잘 맞음), 아니면 None. IoU 로 맞은 것(>1)을 문으로 맞은 것(≤1)보다 앞에 둔다."""
    o = iou(prev, new)
    if o > iou_min:
        return 1.0 + o
    if gate <= 0:
        return None
    side = max(prev[2] - prev[0], prev[3] - prev[1], new[2] - new[0], new[3] - new[1])
    if side <= 0:
        return None
    dx = abs((new[0] + new[2]) - (prev[0] + prev[2])) / 2
    dy = ((new[1] + new[3]) - (prev[1] + prev[3])) / 2
    reach = gate * side
    if dx > reach or dy > reach or dy < -UP_SLACK * side:
        return None
    return 1.0 - max(dx, dy, 0.0) / reach


@dataclass
class Track:
    type: str
    start_t: float
    last_t: float
    last_bbox: Box
    best: Any  # 가장 신뢰도 높은 탐지 (type·confidence·bbox 를 가진 것)
    best_t: float
    payload: Any = None  # 그 탐지가 나온 프레임에서 보낼 때 필요한 것 (이미지, 위치 등)
    hits: int = 1
    confidences: list[float] = field(default_factory=list)


class TrackMerger:
    """주행(기기) 하나에 하나. 탐지할 때마다 update, 매 프레임 expire, 주행이 끝나면 flush.

    세 함수 모두 '끝난 트랙' 목록을 돌려준다. 끝난 트랙 하나가 관측 1건이다 (track.best, track.payload).
    """

    def __init__(self, iou_min: float = IOU_MIN, gap_s: float = GAP_S, max_age_s: float = MAX_AGE_S,
                 gate: float = GATE):
        self.iou_min, self.gap_s, self.max_age_s, self.gate = iou_min, gap_s, max_age_s, gate
        self.open: list[Track] = []
        self.detections = 0  # 받은 탐지 수 (프레임마다 센 것)
        self.closed = 0  # 내보낸 트랙 수 (= 관측 수)

    def _ended(self, tr: Track, t: float) -> bool:
        dt = t - tr.last_t
        return dt < 0 or dt > self.gap_s or t - tr.start_t > self.max_age_s

    def expire(self, t: float) -> list[Track]:
        """시각 t 에서 끝난 것으로 볼 트랙을 닫는다. 시각이 거꾸로 가면(구간이 바뀜) 모두 닫는다."""
        done = [tr for tr in self.open if self._ended(tr, t)]
        if done:
            self.open = [tr for tr in self.open if tr not in done]
            self.closed += len(done)
        return done

    def update(self, dets: Iterable, t: float, payload: Any = None) -> list[Track]:
        """시각 t 프레임의 탐지들을 트랙에 붙인다. 붙을 트랙이 없으면 새 트랙. 돌려주는 것은 이번에 끝난 트랙."""
        done = self.expire(t)
        dets = list(dets)
        self.detections += len(dets)
        pairs = []
        for di, d in enumerate(dets):
            for ti, tr in enumerate(self.open):
                if tr.type == d.type and tr.last_t < t:
                    s = follow_score(tr.last_bbox, d.bbox, self.iou_min, self.gate)
                    if s is not None:
                        pairs.append((s, di, ti))
        used_d, used_t = set(), set()
        for _, di, ti in sorted(pairs, key=lambda p: p[0], reverse=True):
            if di in used_d or ti in used_t:
                continue
            used_d.add(di)
            used_t.add(ti)
            tr, d = self.open[ti], dets[di]
            tr.last_t, tr.last_bbox, tr.hits = t, tuple(d.bbox), tr.hits + 1
            tr.confidences.append(d.confidence)
            if d.confidence > tr.best.confidence:
                tr.best, tr.best_t, tr.payload = d, t, payload
        for di, d in enumerate(dets):
            if di not in used_d:
                self.open.append(Track(d.type, t, t, tuple(d.bbox), d, t, payload, confidences=[d.confidence]))
        return done

    def flush(self) -> list[Track]:
        """주행이 끝날 때 열린 트랙을 모두 닫는다."""
        done, self.open = self.open, []
        self.closed += len(done)
        return done
