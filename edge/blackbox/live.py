"""블랙박스 화면. 가상 블랙박스가 보고 있는 프레임에 탐지 박스를 그려 파일로 내놓는다.

화면(ui/views/blackbox.py)이 이 파일을 계속 다시 읽어 보여준다. 서버를 거치지 않는다(시연은 노트북 한 대).

  server_data/live/<기기>.jpg   박스를 그린 최신 프레임
  server_data/live/<기기>.json  기기 상태 (영상, 시각, 이번 프레임 탐지, 보낸 수, 끝났는지)

초당 fps 장까지만 쓴다. 탐지 박스는 다음 탐지가 나올 때까지 hold_s 초 동안 남겨 둔다
(탐지는 N 프레임마다라서, 그 사이 프레임에서도 박스가 보이게).

박스 라벨은 한국어("표지판 0.31")다. OpenCV 기본 글꼴은 한글을 못 그리므로 Pillow 로 라벨 조각(글자+바탕)만
만들어 붙이고, 같은 라벨은 캐시해 다시 쓴다. 한글 글꼴(FONT_PATHS)이 없으면(CI·Linux) 영어 라벨("sign 0.31")로 그린다.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "server_data" / "live"
# 이 시간 넘게 새로 쓰지 않은 기기는 블랙박스 화면에서 뺀다. 강제로 끈 주행은 '주행 끝'을 남기지 못하므로
# 지난 주행의 마지막 프레임이 다음 날 화면을 켜자마자 뜨지 않게 한다. 막 끝난 주행은 이 시간 동안 보인다
HIDE_AFTER_S = 600

# BGR. 서버로 보내지 않는 시설물(guardrail·sign)도 화면에는 그린다
COLORS = {
    "pothole": (40, 40, 230),
    "crack": (0, 150, 255),
    "lane_wear": (0, 220, 255),
    "guardrail": (230, 160, 40),
    "sign": (80, 200, 80),
}
# 화면에 보이는 이름. ui/views/blackbox.py 범례·탐지 목록도 이것을 쓴다
LABELS = {"pothole": "포트홀", "crack": "균열", "lane_wear": "차선 마모", "guardrail": "가드레일", "sign": "표지판"}
# 한글 글꼴. 앞에서부터 처음 있는 것을 쓴다 (macOS, Linux Noto CJK·나눔)
FONT_PATHS = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


@lru_cache(maxsize=8)
def _font(size: int):
    """한글 글꼴(크기별로 한 번만 연다). 없으면 None → 영어 라벨."""
    from PIL import ImageFont

    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return None


@lru_cache(maxsize=1024)
def _label_patch(text: str, color: tuple[int, int, int], size: int) -> np.ndarray | None:
    """라벨 조각(BGR): color 바탕에 검은 글자. 유형 5개 × 신뢰도 100가지라 캐시가 거의 다 맞는다."""
    font = _font(size)
    if font is None:
        return None
    from PIL import Image, ImageDraw

    pad = max(2, size // 5)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    img = Image.new("RGB", (w, h), color[::-1])  # BGR → RGB
    ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=(0, 0, 0))
    return np.asarray(img)[:, :, ::-1].copy()


def _paste(out: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """patch 를 (x, y) 왼쪽 위에 붙인다. 프레임 밖으로 나가는 부분은 자른다."""
    H, W = out.shape[:2]
    h, w = patch.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 < x1 and y0 < y1:
        out[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]


def is_current(status: dict, now: float, hide_after_s: float = HIDE_AFTER_S) -> bool:
    """블랙박스 화면에 보일 기기인가: 최근 hide_after_s 초 안에 상태를 쓴 기기만."""
    updated = status.get("updated_at")
    return isinstance(updated, (int, float)) and now - updated <= hide_after_s


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)  # 화면이 반쯤 쓴 파일을 읽지 않게


def draw(image: np.ndarray, dets, sent_types: set[str]) -> np.ndarray:
    """dets 의 박스를 그린 사본. 서버로 보내지 않는 유형은 얇은 선."""
    out = image.copy()
    scale = max(1.0, image.shape[1] / 960)
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        color = COLORS.get(d.type, (255, 255, 255))
        thick = int(round((3 if d.type in sent_types else 1) * scale))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        patch = _label_patch(f"{LABELS.get(d.type, d.type)} {d.confidence:.2f}", color, int(round(16 * scale)))
        if patch is not None:  # 박스 위쪽에 붙인다. 프레임 위로 넘치면 박스 안쪽으로
            _paste(out, patch, x1, max(y1 - patch.shape[0], 0))
            continue
        label = f"{d.type} {d.confidence:.2f}"
        fs = 0.6 * scale
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(scale)))
        ty = max(y1, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), max(1, int(scale)))
    return out


class LiveView:
    def __init__(self, device_id: str, source: str, sent_types: set[str], out_dir: str | Path = LIVE_DIR,
                 fps: float = 5.0, hold_s: float = 1.0, width: int = 960):
        self.device_id, self.source, self.sent_types = device_id, source, set(sent_types)
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.jpg = self.dir / f"{device_id}.jpg"
        self.json = self.dir / f"{device_id}.json"
        for p in (self.jpg, self.json):
            p.unlink(missing_ok=True)  # 이전 주행 화면이 남지 않게
        self.period = 1.0 / fps if fps > 0 else 0.0
        self.hold_s, self.width = hold_s, width
        self.last_t: float | None = None
        self.shown: list[tuple[float, object]] = []  # (탐지 시각, Detection)
        self.found = 0

    def due(self, t: float) -> bool:
        """이 프레임을 화면에 내보낼 차례인가 (영상 시각 기준)."""
        return self.last_t is None or t - self.last_t >= self.period or t < self.last_t

    def add(self, dets, t: float, replace: bool = True) -> None:
        """새 탐지 결과. replace 면 이전 박스를 지운다 (탐지기는 매번 프레임 전체를 보므로)."""
        if replace:
            self.shown = [(st, d) for st, d in self.shown if d.type == "lane_wear"]
        self.shown += [(t, d) for d in dets]
        self.found += sum(1 for d in dets if d.type in self.sent_types)

    def update(self, image: np.ndarray, t: float, name: str, note: str = "", sent: int = 0) -> None:
        self.last_t = t
        self.shown = [(st, d) for st, d in self.shown if t - st <= self.hold_s and t >= st]
        dets = [d for _, d in self.shown]
        out = draw(image, dets, self.sent_types)
        if out.shape[1] > self.width:
            h = round(out.shape[0] * self.width / out.shape[1])
            out = cv2.resize(out, (self.width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            _write_atomic(self.jpg, buf.tobytes())
        self._status(name=name, t=t, note=note, sent=sent, ended=False,
                     boxes=[{"type": d.type, "confidence": round(d.confidence, 3)} for d in dets])

    def finish(self, sent: int) -> None:
        self._status(name=None, t=self.last_t, note="주행 끝", sent=sent, ended=True, boxes=[])

    def _status(self, **kw) -> None:
        data = {"device_id": self.device_id, "source": self.source, "found": self.found,
                "updated_at": time.time(), **kw}
        _write_atomic(self.json, json.dumps(data, ensure_ascii=False).encode())
