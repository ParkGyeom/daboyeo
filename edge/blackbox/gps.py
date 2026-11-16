"""로드마크 GPS 파일 읽기. 형식 세 가지를 모두 읽는다 (docs/DATA.md 3장).

  A: 시각, 위도, 경도, 속도(노트), 방위각, 값 3개
  B: 번호, 시각, 위도, 경도, 속도(노트), 방위각, 빈 값
  C: 시각, 위도, 경도, 속도(노트), 방위각, null 4개

시각은 세 형식 모두 HHMMSS.ss(시분초)다. 날짜는 없다.
좌표가 한국 범위를 벗어나면 위치 없음(None)으로 본다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KOREA_LAT = (33.0, 39.0)
KOREA_LON = (124.0, 132.0)
KNOT_TO_KMH = 1.852


@dataclass(frozen=True)
class GpsFix:
    time_s: float | None  # 그날 0시부터 지난 초. 시각을 읽지 못하면 None
    lat: float
    lon: float
    speed_kmh: float | None
    heading: float | None


def in_korea(lat: float, lon: float) -> bool:
    return KOREA_LAT[0] <= lat <= KOREA_LAT[1] and KOREA_LON[0] <= lon <= KOREA_LON[1]


def hhmmss_to_seconds(v: float) -> float | None:
    hh, rest = divmod(v, 10000)
    mm, ss = divmod(rest, 100)
    if hh >= 24 or mm >= 60 or ss >= 60 or v < 0:
        return None
    return hh * 3600 + mm * 60 + ss


def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s.lower() == "null":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def detect_format(values: list[str]) -> str:
    """셋째·넷째 값이 한국 좌표면 B, null 이 있으면 C, 아니면 A."""
    lat, lon = (_num(values[2]), _num(values[3])) if len(values) >= 4 else (None, None)
    if lat is not None and lon is not None and in_korea(lat, lon):
        return "B"
    if any(v.strip().lower() == "null" for v in values):
        return "C"
    return "A"


def parse_line(line: str) -> GpsFix | None:
    """한 줄을 읽는다. 좌표가 없거나 한국 범위 밖이면 None."""
    values = line.strip().split(",")
    if len(values) < 5:
        return None
    off = 1 if detect_format(values) == "B" else 0
    t, lat, lon, speed, heading = (_num(values[off + i]) if off + i < len(values) else None for i in range(5))
    if lat is None or lon is None or not in_korea(lat, lon):
        return None
    return GpsFix(
        time_s=hhmmss_to_seconds(t) if t is not None else None,
        lat=lat,
        lon=lon,
        speed_kmh=speed * KNOT_TO_KMH if speed is not None else None,
        heading=heading,
    )


def read_gps(path: str | Path) -> GpsFix | None:
    """프레임 하나의 GPS 파일(한 줄)을 읽는다. 파일이 없거나 읽지 못하면 None."""
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace").strip()
    return parse_line(text.splitlines()[0]) if text else None
