"""가상 블랙박스 (docs/DESIGN.md 5장). 한 프로세스가 차량 한 대다.

주행 영상(시연 메인) 또는 로드마크 구간(보조)의 프레임을 순서대로 읽고, GPS 를 붙이고, 분석 주기에 따라
추론 모듈을 불러, 결함 주변을 잘라 서버로 보낸다. 프레임 사이에는 실제 시간 간격만큼 기다린다.

  # 영상 + GPS CSV (t_s,lat,lon[,speed_kmh]). --gps 를 빼면 영상 옆의 <영상 이름>.gps.csv
  uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --gps data/gps/v1.gps.csv \\
      --model ai/models/detector_v2.onnx --detect-every 10 --speed 4
  # 로드마크 구간 (프레임마다 실제 GPS)
  uv run python -m edge.blackbox.run --segment 0401_light_rainy_smooth:03000071-03000182 \\
      --model ai/models/detector_v2.onnx --speed 4

품질 판별(B6): 매 프레임 탐지 전에 재고, 나쁘면 탐지를 건너뛴다 (--no-quality 로 끈다).
탐지 신뢰도: 기본은 클래스별 기준(--conf-json, 박겸 초안). --conf 를 주면 숫자 하나로 모든 클래스에 쓴다.
차선 마모: --lane-every 프레임마다 daboyeo_ai 의 LaneWearMeter 로 재고 LaneWearTracker 로 판정한다.
같은 주행의 평소 차선과 비교해 연속 판정 5장 중 3장이 마모일 때 그 프레임 위치로 1건 보내고,
마모 구간이 끝날 때까지 다시 보내지 않는다 (docs/ai-handoff.md 8장. 기준값은 박겸이 11/22까지 확정).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from daboyeo_ai.inference import load_class_conf
from daboyeo_contracts import DefectType, ObservationIn

from edge.blackbox.quality import measure
from edge.blackbox.detect import SENT_TYPES, Detection, Detector, LaneWear
from edge.blackbox.live import LIVE_DIR, LiveView
from edge.blackbox.sender import Sender, crop_snapshot
from edge.blackbox.source import open_source

KST = timezone(timedelta(hours=9))
DEFAULT_INTERVAL_S = 1.0  # 시각을 읽지 못한 프레임 사이 간격
MAX_INTERVAL_S = 3.0  # 구간 중간이 끊겨 있어도 이 이상 기다리지 않는다


def add_gps_noise(lat: float, lon: float, sigma_m: float, rng: random.Random) -> tuple[float, float]:
    if sigma_m <= 0:
        return lat, lon
    north, east = rng.gauss(0, sigma_m), rng.gauss(0, sigma_m)
    return lat + north / 111_320, lon + east / (111_320 * math.cos(math.radians(lat)))


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return image
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


class Pacer:
    """프레임 사이에 기다릴 시간(실제 시간, 초)을 정한다.

    GPS 시각 차이만큼 기다리되 MAX_INTERVAL_S 를 넘지 않는다. 시각이 없는 프레임은 DEFAULT_INTERVAL_S 를 기다리고,
    그만큼은 다음 시각 있는 프레임의 간격에서 뺀다. 빼지 않으면 같은 시간을 두 번 기다리게 된다.
    """

    def __init__(self):
        self.prev: float | None = None
        self.waited = 0.0
        self.started = False

    def next(self, time_s: float | None) -> float:
        first, self.started = not self.started, True
        if time_s is None:
            if first:
                return 0.0
            self.waited += DEFAULT_INTERVAL_S
            return DEFAULT_INTERVAL_S
        wait = 0.0
        if self.prev is not None:
            gap = time_s - self.prev
            if gap < 0:  # 자정을 넘긴 경우
                gap += 86400
            wait = max(0.0, min(gap, MAX_INTERVAL_S) - self.waited)
        self.prev, self.waited = time_s, 0.0
        return wait


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="주행 영상 파일 (mp4 등)")
    src.add_argument("--segment", help="로드마크 구간 <이름 앞부분>:<시작 번호>-<끝 번호>")
    ap.add_argument("--gps", default=None, help="영상의 GPS CSV. 없으면 영상 옆의 <영상 이름>.gps.csv")
    ap.add_argument("--frames", default="data/roadmark/VS4", help="로드마크 프레임 폴더 (IMAGE/, GPS/ 가 들어 있다)")
    ap.add_argument("--device-id", default="bb-01")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="ONNX 파일")
    ap.add_argument("--conf", type=float, default=None, help="탐지 신뢰도 기준 하나 (주면 --conf-json 대신 쓴다)")
    ap.add_argument("--conf-json", default="experiments/conf_thresholds.json",
                    help="클래스별 탐지 신뢰도 기준 파일 (class_conf 키)")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime 스레드 수 (PRD: CPU 1~2코어)")
    ap.add_argument("--detect-every", type=int, default=1, help="결함 탐지 주기(프레임)")
    ap.add_argument("--lane-every", type=int, default=3, help="차선 마모 주기(프레임). 0이면 재지 않는다")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속. 0이면 기다리지 않는다")
    ap.add_argument("--gps-noise", type=float, default=0.0, help="GPS 에 더할 오차의 표준편차(m)")
    ap.add_argument("--brightness", type=float, default=1.0, help="밝기 배율")
    ap.add_argument("--start-offset", type=int, default=0, help="건너뛸 앞쪽 프레임 수")
    ap.add_argument("--seed", type=int, default=None, help="GPS 오차 난수 씨앗")
    ap.add_argument("--dry-run", action="store_true", help="서버로 보내지 않고 화면에만 출력")
    ap.add_argument("--no-quality", action="store_true", help="품질 판별을 끈다 (실험용)")
    ap.add_argument("--live-fps", type=float, default=5.0,
                    help="블랙박스 화면에 초당 몇 장 내보낼지. 0이면 화면을 쓰지 않는다")
    ap.add_argument("--live-dir", default=str(LIVE_DIR), help="블랙박스 화면 파일을 둘 폴더 (화면 ui/ 가 읽는다)")
    args = ap.parse_args(argv)
    if args.detect_every < 1 or args.lane_every < 0:
        ap.error("주기는 1 이상이어야 한다")
    if args.speed < 0:
        ap.error("--speed 는 0 이상이어야 한다")
    return args


def drive(args: argparse.Namespace) -> dict:
    """구간 하나를 재생하고 집계를 돌려준다. 실험 스크립트(experiments/)도 이 함수를 쓴다."""
    try:
        frames = open_source(args)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return {"frames": 0}
    if len(frames) == 0:
        print(f"프레임이 없다: {args.video or f'{args.segment} ({args.frames})'}", file=sys.stderr)
        return {"frames": 0}

    conf = args.conf if args.conf is not None else load_class_conf(args.conf_json)
    detector = Detector(args.model, conf=conf, threads=args.threads)
    sender = Sender(args.server, dry_run=args.dry_run)
    rng = random.Random(args.seed)
    started = datetime.now(KST).replace(microsecond=0)
    if not sender.start_trip(args.device_id, started):
        print(f"[{args.device_id}] 서버에 주행을 열지 못했다 ({args.server}). --dry-run 이면 서버 없이 돈다", file=sys.stderr)
        sender.close()
        return {"frames": 0}
    first_t: float | None = None
    pacer = Pacer()
    detect_s = 0.0
    detected_frames = skipped_no_gps = skipped_quality = 0

    lane_meter = LaneWear() if args.lane_every > 0 else None
    live = (LiveView(args.device_id, Path(args.video).name if args.video else args.segment, SENT_TYPES,
                     out_dir=args.live_dir, fps=args.live_fps) if args.live_fps > 0 else None)

    def send(d: Detection, image, fix, i: int, name: str) -> None:
        x1, y1, x2, y2 = d.bbox
        if x2 <= x1 or y2 <= y1:
            return
        h, w = image.shape[:2]
        # t_s: 주행(영상) 시작부터 초. 영상은 영상 시각, 로드마크는 첫 프레임 GPS 시각 기준
        offset = (fix.time_s - first_t) if (fix.time_s is not None and first_t is not None) else float(i)
        if offset < 0:
            offset += 86400
        lat, lon = add_gps_noise(fix.lat, fix.lon, args.gps_noise, rng)
        # 차선 마모는 구간이라 박스를 보내지 않는다 (handoff 8장). 사진은 측정 영역으로 자른다
        bbox = None if d.type == "lane_wear" else (
            round(x1 / w, 4), round(y1 / h, 4), round(min(1.0, x2 / w), 4), round(min(1.0, y2 / h), 4))
        obs = ObservationIn(
            trip_id=sender.trip_id, vehicle_id=sender.vehicle_id, defect_type=DefectType(d.type),
            confidence=round(d.confidence, 4), bbox=bbox, lat=round(lat, 7), lon=round(lon, 7),
            t_s=round(offset, 3), observed_at=started + timedelta(seconds=offset),
        )
        res = sender.send(obs, crop_snapshot(image, d.bbox))
        tag = "" if res is None else f" → 결함 {str(res['defect_id'])[:8]}{' (새 결함)' if res['is_new_defect'] else ''}"
        print(f"  {name}: {d.type} {d.confidence:.2f}{tag}")

    print(f"[{args.device_id}] {args.video or args.segment} 프레임 {len(frames)}장, 탐지 주기 {args.detect_every}, "
          f"차선 주기 {args.lane_every}, 배속 {args.speed}{', 보내지 않음' if args.dry_run else ''}")
    try:
        for i, frame in enumerate(frames):
            fix = frame.fix

            # 실제 시간 간격만큼 기다린다
            t = fix.time_s if fix else None
            if t is not None and first_t is None:
                first_t = t
            wait = pacer.next(t)
            if args.speed > 0 and wait > 0:
                time.sleep(wait / args.speed)

            if fix is None:
                skipped_no_gps += 1
                print(f"  {frame.name}: 위치 없음, 건너뜀")
                continue
            ft = fix.time_s if fix.time_s is not None else float(i)
            run_detect = i % args.detect_every == 0
            run_lane = args.lane_every > 0 and i % args.lane_every == 0
            show = live is not None and live.due(ft)
            if not (run_detect or run_lane or show):
                continue

            image = adjust_brightness(frame.load(), args.brightness)
            note = ""
            if (run_detect or run_lane) and not args.no_quality:
                q = measure(image)
                if not q.ok:
                    skipped_quality += 1
                    run_detect = run_lane = False
                    note = f"품질 나쁨({q.reason}), 탐지 건너뜀"
                    print(f"  {frame.name}: {note} "
                          f"[밝기 {q.brightness:.0f}, 선명도 {q.sharpness:.0f}, 대비 {q.contrast:.0f}]")
            if run_detect:
                t0 = time.perf_counter()
                dets = detector.detect(image)
                detect_s += time.perf_counter() - t0
                detected_frames += 1
                if live:
                    live.add(dets, ft)
                for d in dets:
                    if d.type in SENT_TYPES:
                        send(d, image, fix, i, frame.name)

            if run_lane:
                lw = lane_meter.check(image, t=fix.time_s)
                if lw:
                    if live:
                        live.add([lw], ft, replace=False)
                    send(lw, image, fix, i, frame.name)
            if show:
                live.update(image, ft, frame.name, note=note, sent=sender.sent)
    except KeyboardInterrupt:
        print("중단")
    finally:
        # 주행 중에는 알리지 않고 끝난 뒤에 한 번만 알린다 (주행을 닫으면 시민의 확인 목록이 생긴다)
        end = sender.end_trip()
        if end is not None:
            print(f"[{args.device_id}] 주행 종료: 시민에게 보여줄 결함 {end['finding_count']}건")
        if live:
            live.finish(sender.sent)
        sender.close()

    avg = detect_s / detected_frames * 1000 if detected_frames else 0
    print(f"[{args.device_id}] 끝. 탐지한 프레임 {detected_frames}, 보낸 결과 {sender.sent}건 "
          f"({sender.bytes_sent / 1024:.1f}KB), 전송 실패 {sender.failed}, 위치 없음 {skipped_no_gps}, "
          f"품질 나쁨 {skipped_quality}, "
          f"탐지 평균 {avg:.0f}ms/장")
    return {
        "frames": len(frames),
        "detected_frames": detected_frames,
        "sent": sender.sent,
        "bytes_sent": sender.bytes_sent,
        "failed": sender.failed,
        "skipped_no_gps": skipped_no_gps,
        "skipped_quality": skipped_quality,
        "detect_ms_avg": avg,
    }


def main(argv=None) -> int:
    result = drive(parse_args(argv))
    return 1 if result.get("frames", 0) == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
