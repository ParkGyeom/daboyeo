"""결함 구간 클립 (계획 1.3, 2장 '클립': 탐지 전후 2초, 480p, H.264 mp4).

블랙박스는 원본을 계속 덮어쓰며 녹화한다. 결함을 찾으면 그 시각 앞뒤 2초를 원본에서 잘라 480p H.264 로 따로
저장하고 서버로 보낸다 (PRD FR-1.5). 가상 블랙박스는 영상 파일이 원본이라, 파일에서 그 구간을 다시 읽어 자른다.
실제 블랙박스라면 뒤쪽 2초가 녹화될 때까지 기다렸다 보내야 하지만, 파일 재생에서는 바로 자를 수 있다.

같은 결함을 연달아 찾으면 구간이 겹친다. 탐지 시각이 앞 클립 안에 있으면 클립을 새로 만들지 않고(앞 클립에 담겨 있다),
앞 클립 뒤라도 구간이 겹치면 앞 클립이 끝난 다음 프레임부터 자른다. 같은 프레임을 두 번 보내지 않는다.
사진·메타는 탐지마다 보낸다.

인코딩은 ffmpeg(libx264, CRF 28)가 있으면 그것을, 없으면 OpenCV(avc1, 안 되면 mp4v)를 쓴다.
로드마크 구간(사진 열)은 클립을 만들지 않는다.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

CLIP_PRE_S = 2.0
CLIP_POST_S = 2.0
CLIP_HEIGHT = 480
CLIP_FPS = 30.0  # 원본이 60fps 여도 클립은 30fps 로 줄인다 (일반 블랙박스 녹화 fps)
CLIP_CRF = 28


def clip_size(w: int, h: int) -> tuple[int, int]:
    """긴 변 비율을 지키며 높이 480 (이미 작으면 그대로). H.264 를 위해 짝수로 맞춘다."""
    if h > CLIP_HEIGHT:
        w, h = w * CLIP_HEIGHT / h, CLIP_HEIGHT
    return max(2, int(round(w / 2)) * 2), max(2, int(round(h / 2)) * 2)


def encode_h264(frames: list[np.ndarray], fps: float) -> tuple[bytes, str]:
    """BGR 프레임들(같은 크기)을 mp4 로. (바이트, 코덱 이름)."""
    if not frames:
        raise ValueError("프레임이 없다")
    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "clip.mp4"
        if ffmpeg:
            cmd = [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CLIP_CRF),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
            p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True)
            if p.returncode == 0 and out.exists():
                return out.read_bytes(), "h264"
        for fourcc, name in (("avc1", "h264"), ("mp4v", "mpeg4")):
            wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if not wr.isOpened():
                continue
            for f in frames:
                wr.write(f)
            wr.release()
            if out.exists() and out.stat().st_size > 0:
                return out.read_bytes(), name
    raise OSError("mp4 를 만들지 못했다 (ffmpeg 도 OpenCV 인코더도 없다)")


class ClipCutter:
    """영상 파일 하나에서 결함 구간 클립을 자른다. 기기(주행) 하나에 하나."""

    def __init__(self, video: str | Path, pre_s: float = CLIP_PRE_S, post_s: float = CLIP_POST_S):
        self.video = Path(video)
        self.pre_s, self.post_s = pre_s, post_s
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            raise OSError(f"영상을 열지 못했다: {self.video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.step = max(1, round(self.fps / CLIP_FPS))
        self.last_frame = -1  # 마지막으로 자른 클립의 끝 프레임 번호
        self.codec = ""
        self.made = 0
        self.skipped_overlap = 0

    def cut(self, t_s: float) -> bytes | None:
        """t_s 앞뒤 구간을 mp4 로. t_s 가 앞 클립 안이면 None (앞 클립에 담겨 있다)."""
        step = self.step
        if int(round(t_s * self.fps)) <= self.last_frame:
            self.skipped_overlap += 1
            return None
        first = max(0, int(round((t_s - self.pre_s) * self.fps)))
        if self.last_frame >= 0:
            first = max(first, self.last_frame + step)
        last = int(round((t_s + self.post_s) * self.fps))
        if self.total > 0:
            last = min(self.total - 1, last)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        size = None
        for i in range(first, last + 1):
            if not self.cap.grab():
                break
            if (i - first) % step:
                continue
            ok, img = self.cap.retrieve()
            if not ok or img is None:
                break
            if size is None:
                size = clip_size(img.shape[1], img.shape[0])
            frames.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
        if not frames:
            return None
        data, self.codec = encode_h264(frames, self.fps / step)
        self.last_frame = first + (len(frames) - 1) * step
        self.made += 1
        return data

    def close(self) -> None:
        self.cap.release()
