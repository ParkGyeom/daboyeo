"""가상 블랙박스 입력. 주행 영상(시연 메인) 또는 로드마크 프레임 구간(보조)을 같은 모양으로 내준다.

- 영상: OpenCV VideoCapture 로 읽고, 위치는 GPS CSV(`t_s,lat,lon[,speed_kmh]`, contracts 의 GpsTrack)를
  영상 시각으로 보간한다 (PRD FR-4.4). GPS CSV 를 따로 주지 않으면 영상 옆의 `<영상 이름>.gps.csv`.
- 로드마크: 프레임마다 실제 GPS 파일이 있는 사진 열 (edge/blackbox/frames.py).

어느 쪽이든 SourceFrame 을 순서대로 내주고, 이미지는 load() 를 부를 때만 읽는다.
영상은 분석하지 않는 프레임을 디코딩하지 않는다(grab 만 한다). load() 는 그 프레임 차례에만 부를 수 있다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from daboyeo_contracts import GpsTrack

from edge.blackbox.frames import list_frames, load_gps, load_image
from edge.blackbox.gps import GpsFix


@dataclass
class SourceFrame:
    name: str
    fix: GpsFix | None  # time_s: 영상은 영상 시작 기준 초, 로드마크는 그날 0시 기준 초
    load: Callable[[], np.ndarray]  # BGR 이미지
    # 영상 시각(영상 시작 기준 초). GPS 경로 밖이라 fix 가 없어도 있다 — 그런 프레임도 실제 간격만큼만 기다리게.
    # 로드마크는 None (프레임 시각이 GPS 에만 있다)
    time_s: float | None = None


def default_gps_path(video: str | Path) -> Path:
    video = Path(video)
    return video.with_name(f"{video.stem}.gps.csv")


def video_fix(track: GpsTrack, t_s: float) -> GpsFix | None:
    """영상 시각 t_s 의 위치. GPS 경로 범위 밖이면 None (그 프레임은 건너뛴다)."""
    try:
        lat, lon = track.at(t_s)
    except ValueError:
        return None
    return GpsFix(time_s=t_s, lat=lat, lon=lon, speed_kmh=None, heading=None)


class VideoSource:
    def __init__(self, video: str | Path, gps: str | Path | None = None, start_offset: int = 0):
        self.video = Path(video)
        if not self.video.exists():
            raise FileNotFoundError(f"영상이 없다: {self.video}")
        gps_path = Path(gps) if gps else default_gps_path(self.video)
        if not gps_path.exists():
            raise FileNotFoundError(f"GPS CSV 가 없다: {gps_path} (--gps 로 준다)")
        self.track = GpsTrack.from_csv(gps_path)
        self.start_offset = start_offset
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise OSError(f"영상을 열지 못했다: {self.video}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_offset)
        cap.release()

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[SourceFrame]:
        cap = cv2.VideoCapture(str(self.video))
        try:
            i = 0
            while cap.grab():
                if i >= self.start_offset:
                    t = i / self.fps

                    def load(cap=cap, i=i) -> np.ndarray:
                        ok, img = cap.retrieve()
                        if not ok or img is None:
                            raise OSError(f"{self.video} 프레임 {i} 을 읽지 못했다")
                        return img

                    yield SourceFrame(f"{self.video.stem}@{t:.2f}s", video_fix(self.track, t), load, time_s=t)
                i += 1
        finally:
            cap.release()


class RoadmarkSource:
    def __init__(self, root: str | Path, segment: str, start_offset: int = 0):
        self.frames = list_frames(root, segment)[start_offset:]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[SourceFrame]:
        for f in self.frames:
            yield SourceFrame(f.name, load_gps(f), lambda f=f: load_image(f))


def open_source(args) -> VideoSource | RoadmarkSource:
    if args.video:
        return VideoSource(args.video, args.gps, args.start_offset)
    return RoadmarkSource(args.frames, args.segment, args.start_offset)
