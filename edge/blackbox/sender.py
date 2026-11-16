"""관측을 서버로 보내고 전송 용량을 센다 (계획 1.2·1.3).

주행 하나 = 서버의 trip 하나. start_trip → send(ObservationIn, 사진) … → end_trip 순서로 부른다.
dry_run 이면 보내지 않고 용량만 센다 (주행·차량 id 는 임의로 만든다).
"""

from __future__ import annotations

import uuid
from datetime import datetime

import cv2
import httpx
import numpy as np
from daboyeo_contracts import ObservationIn

SNAPSHOT_MAX_SIDE = 640
SNAPSHOT_QUALITY = 80


def crop_snapshot(image: np.ndarray, bbox: tuple[int, int, int, int], margin: float = 0.5) -> bytes:
    """결함 주변을 여유를 두고 잘라 긴 변 640픽셀 이하 JPEG 로 만든다. 프레임 전체를 보내지 않는 것이 통신량 절감의 핵심이다."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin) + 16, int((y2 - y1) * margin) + 16
    crop = image[max(0, y1 - my): min(h, y2 + my), max(0, x1 - mx): min(w, x2 + mx)]
    ch, cw = crop.shape[:2]
    scale = SNAPSHOT_MAX_SIDE / max(ch, cw)
    if scale < 1:
        crop = cv2.resize(crop, (max(1, round(cw * scale)), max(1, round(ch * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, SNAPSHOT_QUALITY])
    if not ok:
        raise ValueError("JPEG 인코딩 실패")
    return buf.tobytes()


class Sender:
    def __init__(self, server: str, dry_run: bool = False, timeout: float = 10.0):
        self.dry_run = dry_run
        self.client = None if dry_run else httpx.Client(base_url=server, timeout=timeout)
        self.trip_id: uuid.UUID | None = None
        self.vehicle_id: uuid.UUID | None = None
        self.sent = 0
        self.failed = 0
        self.bytes_sent = 0

    def start_trip(self, vehicle_label: str, started_at: datetime) -> bool:
        """주행을 연다. 서버에 닿지 않거나 없는 차량이면 False (그러면 관측을 보내지 않는다)."""
        if self.dry_run:
            self.trip_id, self.vehicle_id = uuid.uuid4(), uuid.uuid4()
            return True
        try:
            r = self.client.post("/api/v1/trips",
                                 json={"vehicle_label": vehicle_label, "started_at": started_at.isoformat()})
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"  주행 시작 실패: {e}")
            return False
        data = r.json()
        self.trip_id, self.vehicle_id = uuid.UUID(data["trip_id"]), uuid.UUID(data["vehicle_id"])
        return True

    def send(self, obs: ObservationIn, snapshot: bytes | None) -> dict | None:
        meta = obs.model_dump_json()
        size = len(meta.encode("utf-8")) + (len(snapshot) if snapshot else 0)
        if self.dry_run:
            self.sent += 1
            self.bytes_sent += size
            return None
        files = {"snapshot": ("snapshot.jpg", snapshot, "image/jpeg")} if snapshot else None
        try:
            r = self.client.post("/api/v1/observations", data={"meta": meta}, files=files)
            r.raise_for_status()
        except httpx.HTTPError as e:
            # 서버가 잠깐 멈춰도 주행(재생)은 계속한다
            self.failed += 1
            print(f"  전송 실패: {e}")
            return None
        self.sent += 1
        self.bytes_sent += size
        return r.json()

    def end_trip(self) -> dict | None:
        """주행을 닫는다. 시민 화면에 '주행이 끝났어요' 알림과 확인 목록이 생긴다. 실패해도 주행 결과는 그대로 둔다."""
        if self.dry_run or self.trip_id is None:
            return None
        try:
            r = self.client.post(f"/api/v1/trips/{self.trip_id}/end")
            r.raise_for_status()
        except httpx.HTTPError as e:
            print(f"  주행 종료 알림 실패: {e}")
            return None
        return r.json()

    def close(self) -> None:
        if self.client:
            self.client.close()
