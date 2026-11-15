"""프레임 열과 구간 읽기 (docs/DATA.md 3장).

구간은 "<이름 앞부분>:<시작 번호>-<끝 번호>" 로 준다.
예: 0401_light_rainy_smooth:03000071-03000182
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edge.blackbox.gps import GpsFix, read_gps

_SEGMENT = re.compile(r"^(?P<prefix>[A-Za-z0-9_]+):(?P<start>\d{8})-(?P<end>\d{8})$")


@dataclass(frozen=True)
class Frame:
    name: str
    image_path: Path
    gps_path: Path


def parse_segment(text: str) -> tuple[str, int, int]:
    m = _SEGMENT.match(text.strip())
    if not m:
        raise ValueError(f"구간 형식이 틀렸다: {text!r} (예: 0401_light_rainy_smooth:03000071-03000182)")
    start, end = int(m["start"]), int(m["end"])
    if start > end:
        raise ValueError(f"시작 번호가 끝 번호보다 크다: {text!r}")
    return m["prefix"], start, end


def list_frames(root: str | Path, segment: str) -> list[Frame]:
    """구간에 속한 프레임을 번호 순으로. 이미지가 없는 번호는 건너뛴다."""
    root = Path(root)
    prefix, start, end = parse_segment(segment)
    out = []
    for n in range(start, end + 1):
        name = f"{prefix}_{n:08d}"
        img = root / "IMAGE" / f"{name}.png"
        if img.exists():
            out.append(Frame(name, img, root / "GPS" / f"{name}.csv"))
    return out


def load_image(frame: Frame) -> np.ndarray:
    img = cv2.imread(str(frame.image_path))
    if img is None:
        raise OSError(f"이미지를 읽지 못했다: {frame.image_path}")
    return img


def load_gps(frame: Frame) -> GpsFix | None:
    return read_gps(frame.gps_path)
