"""블랙박스 화면. 가상 블랙박스가 보고 있는 프레임에 탐지 박스를 그려 파일로 내놓는다.

화면(ui/views/blackbox.py)이 이 파일을 계속 다시 읽어 보여준다. 서버를 거치지 않는다(시연은 노트북 한 대).

  server_data/live/<기기>.jpg   박스를 그린 최신 프레임
  server_data/live/<기기>.json  기기 상태 (영상, 시각, 이번 프레임 탐지, 보낸 수, 끝났는지)

초당 fps 장까지만 쓴다. 탐지 박스는 다음 탐지가 나올 때까지 hold_s 초 동안 남겨 둔다
(탐지는 N 프레임마다라서, 그 사이 프레임에서도 박스가 보이게).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "server_data" / "live"

# BGR. 서버로 보내지 않는 시설물(guardrail·sign)도 화면에는 그린다
COLORS = {
    "pothole": (40, 40, 230),
    "crack": (0, 150, 255),
    "lane_wear": (0, 220, 255),
    "guardrail": (230, 160, 40),
    "sign": (80, 200, 80),
}


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
