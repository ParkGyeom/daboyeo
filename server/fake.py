"""가짜 탐지 결과 생성기. 모델과 데이터 없이 서버를 시험한다 (docs/DESIGN.md 6장).

지정한 좌표 주변에 결함을 만들고, 기기마다 GPS 오차를 더해 그 결함을 본 것처럼 서버로 보낸다.

예: 결함 5개를 기기 3대가 보고, GPS 오차 표준편차 7m
  .venv/bin/python -m server.fake --defects 5 --devices 3 --gps-noise 7
"""

from __future__ import annotations

import argparse
import io
import math
import random
from datetime import datetime, timedelta, timezone

import httpx
from PIL import Image, ImageDraw

KST = timezone(timedelta(hours=9))
FRAME_W, FRAME_H = 1920, 1080
COLORS = {"pothole": (200, 60, 60), "crack": (60, 120, 200), "lane_wear": (200, 180, 60)}


def snapshot_jpeg(kind: str, confidence: float) -> bytes:
    """단색 사진. 유형별로 색이 다르다."""
    img = Image.new("RGB", (320, 240), COLORS[kind])
    ImageDraw.Draw(img).text((10, 10), f"{kind} {confidence:.2f}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def offset_m(lat: float, lon: float, north: float, east: float) -> tuple[float, float]:
    """(lat, lon)에서 북쪽 north m, 동쪽 east m 떨어진 점."""
    return lat + north / 111_320, lon + east / (111_320 * math.cos(math.radians(lat)))


def make_events(
    lat: float, lon: float, defects: int, devices: int, gps_noise: float, radius: float,
    types: list[str], rng: random.Random, device_prefix: str = "bb-", start: datetime | None = None,
) -> list[tuple[dict, bytes]]:
    """(event, 사진) 목록. 결함마다 모든 기기가 한 번씩 본다."""
    start = start or datetime.now(KST)
    out = []
    for i in range(defects):
        r, a = radius * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        dlat, dlon = offset_m(lat, lon, r * math.cos(a), r * math.sin(a))
        kind = rng.choice(types)
        for d in range(devices):
            plat, plon = offset_m(dlat, dlon, rng.gauss(0, gps_noise), rng.gauss(0, gps_noise))
            conf = round(rng.uniform(0.45, 0.95), 3)
            x1, y1 = rng.randint(0, FRAME_W - 400), rng.randint(0, FRAME_H - 300)
            event = {
                "device_id": f"{device_prefix}{d + 1:02d}",
                "detected_at": (start + timedelta(seconds=i * 10 + d)).isoformat(timespec="seconds"),
                "lat": round(plat, 7), "lon": round(plon, 7),
                "type": kind, "confidence": conf,
                "bbox": [x1, y1, x1 + rng.randint(80, 400), y1 + rng.randint(60, 300)],
                "frame_w": FRAME_W, "frame_h": FRAME_H, "quality_ok": True,
            }
            out.append((event, snapshot_jpeg(kind, conf)))
    return out


def send(client: httpx.Client, event: dict, photo: bytes) -> dict:
    import json

    r = client.post("/api/detections", data={"event": json.dumps(event)}, files={"snapshot": ("s.jpg", photo, "image/jpeg")})
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--lat", type=float, default=37.5503, help="중심 위도 (기본: 서울 성동구 부근)")
    ap.add_argument("--lon", type=float, default=127.0731)
    ap.add_argument("--defects", type=int, default=5, help="만들 결함 수")
    ap.add_argument("--devices", type=int, default=3, help="결함을 보는 기기 수 (bb-01, bb-02, ...)")
    ap.add_argument("--gps-noise", type=float, default=7.0, help="GPS 오차 표준편차(m)")
    ap.add_argument("--radius", type=float, default=500.0, help="결함을 흩뿌릴 반지름(m)")
    ap.add_argument("--types", nargs="+", default=["pothole", "crack", "lane_wear"], choices=list(COLORS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--end-drive", action="store_true",
                    help="다 보낸 뒤 기기마다 주행 종료를 알린다 (시민 화면에 '주행이 끝났어요' 알림이 뜬다)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start = datetime.now(KST).replace(microsecond=0)  # detected_at 이 초 단위라 맞춘다
    events = make_events(args.lat, args.lon, args.defects, args.devices, args.gps_noise, args.radius, args.types, rng,
                         start=start)
    with httpx.Client(base_url=args.server, timeout=10) as client:
        for ev, photo in events:
            send(client, ev, photo)
        if args.end_drive:
            for d in range(args.devices):
                r = client.post("/api/drives", json={"device_id": f"bb-{d + 1:02d}", "started_at": start.isoformat()})
                r.raise_for_status()
    print(f"탐지 결과 {len(events)}건을 보냈다 (결함 {args.defects}개 x 기기 {args.devices}대, GPS 오차 {args.gps_noise}m)"
          + (f", 주행 종료 {args.devices}대" if args.end_drive else ""))


if __name__ == "__main__":
    main()
