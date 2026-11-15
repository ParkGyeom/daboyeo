"""가짜 관측 생성기. 모델과 영상 없이 서버·화면을 시험한다.

지정한 좌표 주변에 결함을 만들고, 차량(bb-01, bb-02, …)마다 주행을 열어 GPS 오차를 더해 그 결함을 본 것처럼
ObservationIn 으로 보낸다. --end-drive 를 주면 주행을 닫아 시민의 확인 목록과 '주행이 끝났어요' 알림이 생긴다.

예: 결함 5개를 차량 3대가 보고, GPS 오차 표준편차 7m
  uv run python -m server.fake --defects 5 --devices 3 --gps-noise 7 --end-drive
"""

from __future__ import annotations

import argparse
import io
import math
import random
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
from daboyeo_contracts import DefectType, ObservationIn
from PIL import Image, ImageDraw

KST = timezone(timedelta(hours=9))
COLORS = {"pothole": (200, 60, 60), "crack": (60, 120, 200), "lane_wear": (200, 180, 60),
          "guardrail_damage": (60, 160, 90), "sign_damage": (140, 90, 180)}


def snapshot_jpeg(kind: str, confidence: float) -> bytes:
    """단색 사진. 유형별로 색이 다르다."""
    img = Image.new("RGB", (320, 240), COLORS.get(kind, (128, 128, 128)))
    ImageDraw.Draw(img).text((10, 10), f"{kind} {confidence:.2f}", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def offset_m(lat: float, lon: float, north: float, east: float) -> tuple[float, float]:
    """(lat, lon)에서 북쪽 north m, 동쪽 east m 떨어진 점."""
    return lat + north / 111_320, lon + east / (111_320 * math.cos(math.radians(lat)))


def make_plan(lat: float, lon: float, defects: int, devices: int, gps_noise: float, radius: float,
              types: list[str], rng: random.Random) -> dict[str, list[dict]]:
    """차량 이름 → 그 차량이 보낼 관측(주행 id 를 빼고). 결함마다 모든 차량이 한 번씩 본다."""
    plan: dict[str, list[dict]] = {f"bb-{d + 1:02d}": [] for d in range(devices)}
    for i in range(defects):
        r, a = radius * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        dlat, dlon = offset_m(lat, lon, r * math.cos(a), r * math.sin(a))
        kind = rng.choice(types)
        for d, label in enumerate(plan):
            plat, plon = offset_m(dlat, dlon, rng.gauss(0, gps_noise), rng.gauss(0, gps_noise))
            x1, y1 = rng.uniform(0, 0.7), rng.uniform(0.3, 0.7)
            plan[label].append({
                "defect_type": kind, "confidence": round(rng.uniform(0.45, 0.95), 3),
                "bbox": None if kind == "lane_wear" else
                        (round(x1, 4), round(y1, 4), round(x1 + rng.uniform(0.05, 0.25), 4), round(y1 + rng.uniform(0.05, 0.25), 4)),
                "lat": round(plat, 7), "lon": round(plon, 7), "t_s": float(i * 10 + d),
            })
    return plan


def observation(item: dict, trip_id: UUID, vehicle_id: UUID, started: datetime) -> ObservationIn:
    return ObservationIn(trip_id=trip_id, vehicle_id=vehicle_id, defect_type=DefectType(item["defect_type"]),
                         confidence=item["confidence"], bbox=item["bbox"], lat=item["lat"], lon=item["lon"],
                         t_s=item["t_s"], observed_at=started + timedelta(seconds=item["t_s"]))


def send_observation(client: httpx.Client, obs: ObservationIn, photo: bytes | None) -> dict:
    files = {"snapshot": ("s.jpg", photo, "image/jpeg")} if photo is not None else None
    r = client.post("/api/v1/observations", data={"meta": obs.model_dump_json()}, files=files)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--lat", type=float, default=37.4979, help="중심 위도 (기본: 강남역)")
    ap.add_argument("--lon", type=float, default=127.0276)
    ap.add_argument("--defects", type=int, default=5, help="만들 결함 수")
    ap.add_argument("--devices", type=int, default=3, choices=range(1, 4), help="결함을 보는 차량 수 (bb-01~03)")
    ap.add_argument("--gps-noise", type=float, default=7.0, help="GPS 오차 표준편차(m)")
    ap.add_argument("--radius", type=float, default=500.0, help="결함을 흩뿌릴 반지름(m)")
    ap.add_argument("--types", nargs="+", default=["pothole", "crack", "lane_wear"], choices=list(COLORS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--end-drive", action="store_true",
                    help="다 보낸 뒤 주행을 닫는다 (시민 확인 목록과 '주행이 끝났어요' 알림이 생긴다)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    plan = make_plan(args.lat, args.lon, args.defects, args.devices, args.gps_noise, args.radius, args.types, rng)
    started = datetime.now(KST).replace(microsecond=0)
    sent = 0
    with httpx.Client(base_url=args.server, timeout=10) as client:
        for label, items in plan.items():
            r = client.post("/api/v1/trips", json={"vehicle_label": label, "started_at": started.isoformat()})
            r.raise_for_status()
            trip = r.json()
            for it in items:
                send_observation(client, observation(it, trip["trip_id"], trip["vehicle_id"], started),
                                 snapshot_jpeg(it["defect_type"], it["confidence"]))
                sent += 1
            if args.end_drive:
                client.post(f"/api/v1/trips/{trip['trip_id']}/end").raise_for_status()
    print(f"관측 {sent}건을 보냈다 (결함 {args.defects}개 x 차량 {args.devices}대, GPS 오차 {args.gps_noise}m)"
          + (f", 주행 종료 {args.devices}대" if args.end_drive else ""))


if __name__ == "__main__":
    main()
