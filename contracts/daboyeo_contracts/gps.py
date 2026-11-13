"""GPS CSV 계약 (FR-4.4).

형식 하나만 쓴다:

    t_s,lat,lon[,speed_kmh]
    0.0,37.55012,126.92341
    1.0,37.55020,126.92355

`t_s` 는 영상 시작 기준 초. 재구성 경로(tools/gps), BDD100K, 실제 블랙박스 GPS 모두
이 형식으로 바꿔서 넣는다.
"""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("t_s", "lat", "lon")
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 사이 거리(m)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass(frozen=True, slots=True)
class GpsPoint:
    t_s: float
    lat: float
    lon: float
    speed_kmh: float | None = None


class GpsTrack:
    """시각 순으로 정렬된 GPS 점들. `at(t)` 로 임의 시각의 좌표를 선형 보간한다."""

    def __init__(self, points: list[GpsPoint]):
        if len(points) < 2:
            raise ValueError("GPS track needs at least 2 points")
        for p in points:
            if not (-90 <= p.lat <= 90 and -180 <= p.lon <= 180):
                raise ValueError(f"invalid coordinate at t_s={p.t_s}: ({p.lat}, {p.lon})")
        for a, b in zip(points, points[1:]):
            if b.t_s <= a.t_s:
                raise ValueError(f"t_s must strictly increase: {a.t_s} -> {b.t_s}")
        self.points = points
        self._times = [p.t_s for p in points]

    @property
    def start_s(self) -> float:
        return self._times[0]

    @property
    def end_s(self) -> float:
        return self._times[-1]

    def at(self, t_s: float, *, clamp: bool = False) -> tuple[float, float]:
        """시각 t_s 의 (lat, lon). 범위 밖이면 clamp=True 일 때 끝점, 아니면 ValueError."""
        if not (self.start_s <= t_s <= self.end_s):
            if not clamp:
                raise ValueError(f"t_s={t_s} outside track [{self.start_s}, {self.end_s}]")
            t_s = min(max(t_s, self.start_s), self.end_s)
        i = bisect_right(self._times, t_s)
        if i >= len(self.points):
            p = self.points[-1]
            return p.lat, p.lon
        a, b = self.points[i - 1], self.points[i]
        r = (t_s - a.t_s) / (b.t_s - a.t_s)
        return a.lat + (b.lat - a.lat) * r, a.lon + (b.lon - a.lon) * r

    def length_m(self) -> float:
        return sum(
            haversine_m(a.lat, a.lon, b.lat, b.lon) for a, b in zip(self.points, self.points[1:])
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> GpsTrack:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}, need {REQUIRED_COLUMNS}")
            points = []
            for row in reader:
                speed = row.get("speed_kmh")
                points.append(
                    GpsPoint(
                        t_s=float(row["t_s"]),
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        speed_kmh=float(speed) if speed not in (None, "") else None,
                    )
                )
        return cls(points)

    def to_csv(self, path: str | Path) -> None:
        has_speed = any(p.speed_kmh is not None for p in self.points)
        cols = [*REQUIRED_COLUMNS, *(["speed_kmh"] if has_speed else [])]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for p in self.points:
                row = [f"{p.t_s:.3f}", f"{p.lat:.7f}", f"{p.lon:.7f}"]
                if has_speed:
                    row.append("" if p.speed_kmh is None else f"{p.speed_kmh:.1f}")
                w.writerow(row)
