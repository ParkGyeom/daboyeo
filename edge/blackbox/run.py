"""가상 블랙박스 (docs/DESIGN.md 5장). 한 프로세스가 차량 한 대다.

주행 영상(시연 메인) 또는 로드마크 구간(보조)의 프레임을 순서대로 읽고, GPS 를 붙이고, 분석 주기에 따라
추론 모듈을 불러, 결함 주변을 잘라 서버로 보낸다. 프레임 사이에는 실제 시간 간격만큼 기다린다.

  # 영상 + GPS CSV (t_s,lat,lon[,speed_kmh]). --gps 를 빼면 영상 옆의 <영상 이름>.gps.csv
  uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --gps data/gps/v1.gps.csv \\
      --model ai/models/detector_v2.onnx --detect-every 10 --speed 4
  # 로드마크 구간 (프레임마다 실제 GPS)
  uv run python -m edge.blackbox.run --segment 0401_light_rainy_smooth:03000071-03000182 \\
      --model ai/models/detector_v2.onnx --speed 4
  # 탐지기 고르기 (edge/blackbox/detectors.py): 기본 onnx. fake 는 정답 라벨 CSV, replay 는 onnx 로 남긴 기록
  ... --detector fake --labels experiments/gt_events/v1.csv      # 1.1 걷는 골격 (라벨 1개 = 관측 1건)
  ... --model ai/models/detector_v3.onnx --record data/replay/v1.jsonl   # onnx 로 돌리며 기록
  ... --detector replay --record data/replay/v1.jsonl            # 추론 없이 기록대로 (모델 파일 필요 없음)

분석 주기(계획 2.2, scheduler.py): 품질 판별은 매 프레임(--quality-every), 탐지는 --detect-every,
차선 마모는 --lane-every 프레임마다. 품질이 나쁜 프레임은 탐지·차선을 건너뛴다 (--no-quality 로 끈다).
끝날 때 단계별 실행 비율과 평균 시간을 출력한다.
연속 프레임 합치기(계획 2.3, track_merge.py): 같은 결함이 여러 프레임에 보여도 트랙 하나로 묶고,
트랙이 끝날 때(--merge-gap 초 동안 안 보이거나 주행이 끝날 때) 가장 신뢰도 높은 프레임으로 관측 1건만 보낸다.
--no-merge 면 예전처럼 탐지마다 보낸다 (합치기 전후 비교용).
탐지 신뢰도: 기본은 시연 모델 v3 의 클래스별 기준(--conf-json ai/models/detector_v3.json: 포트홀 0.31 · 균열 0.42 ·
가드레일 0.21 · 표지판 0.21, handoff 16장). --conf 를 주면 숫자 하나로 모든 클래스에 쓴다.
도로 조각 추론(handoff 26장, 기본 끔): --road-tiles 2 --tiles-bottom 0.8 --tile-classes crack 이면 프레임 전체에 더해
도로 부분(높이 0.45~0.8)을 가로 2조각으로 한 번씩 더 돌려 작은 균열을 찾는다 (gn_b 균열 정답 9곳 중 3 → 7곳).
모델을 3번 돌려 탐지 시간이 약 3배다. 앞유리 반사가 심한 영상에서는 헛것이 생기니 끈다. 기록(--record) 머리 줄에 설정이 남는다.
차선 마모: 기본은 끈다 (--lane-every 0). 시연 클립 gn_b 에서 확정 58건 중 54건이 헛것(25건은 신호 대기 중
횡단보도), gn_a 는 젖은 노면이라 차선을 2.8% 프레임에서만 찾았다 (docs/ai-handoff.md 17장). 개선(정지 판정·횡단보도
거르기)을 해도 헛것 84% 라 시연에서는 끈다 (확정, handoff 17-6). 실험용으로 켤 때는 --lane-every 3. 켜면 --lane-every 프레임마다 daboyeo_ai 의 LaneWearMeter 로 재고
LaneWearTracker 로 판정한다. 같은 주행의 평소 차선과 비교해 연속 판정 5장 중 3장이 마모일 때 그 프레임 위치로
1건 보내고, 마모 구간이 끝날 때까지 다시 보내지 않는다 (docs/ai-handoff.md 8장).
클립(계획 1.3): 영상 입력이면 결함마다 전후 2초를 480p H.264 로 잘라 사진과 함께 보낸다 (edge/blackbox/clipper.py,
앞 클립과 겹치면 다시 자르지 않는다). 보낸 사진·클립은 --keep-dir/<기기>/<주행>/ 에도 남는다. --no-clip 으로 끈다.
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

from daboyeo_contracts import CLASS_NAMES, DefectType, ObservationIn

from edge.blackbox.clipper import ClipCutter
from edge.blackbox.detect import SENT_TYPES, Detection, LaneWear
from edge.blackbox.detectors import DETECTORS, make_detector
from edge.blackbox.live import LIVE_DIR, LiveView
from edge.blackbox.quality import measure
from edge.blackbox.scheduler import Scheduler
from edge.blackbox.sender import Sender, crop_snapshot
from edge.blackbox.source import open_source
from edge.blackbox.track_merge import GAP_S, TrackMerger

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
    ap.add_argument("--detector", choices=DETECTORS, default="onnx",
                    help="onnx: 모델 추론, fake: 정답 라벨 CSV 를 탐지처럼, replay: --record 기록을 추론 없이")
    ap.add_argument("--model", default=None, help="ONNX 파일 (--detector onnx 일 때)")
    ap.add_argument("--labels", default=None,
                    help="fake 의 정답 라벨 CSV. 없으면 experiments/gt_events/<영상 이름>.csv")
    ap.add_argument("--include-amb", action="store_true",
                    help="fake 가 memo 가 amb 로 시작하는 애매한 라벨도 낸다 (기본은 확실한 정답만, handoff 18-4)")
    ap.add_argument("--record", default=None,
                    help="탐지 기록 JSONL. onnx·fake 는 여기에 쓰고, replay 는 여기서 읽는다")
    ap.add_argument("--conf", type=float, default=None, help="탐지 신뢰도 기준 하나 (주면 --conf-json 대신 쓴다)")
    ap.add_argument("--conf-json", default="ai/models/detector_v3.json",
                    help="클래스별 탐지 신뢰도 기준 파일 (class_conf 또는 class_conf_draft 키). 기본은 시연 모델 v3")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime 스레드 수 (PRD: CPU 1~2코어)")
    ap.add_argument("--road-tiles", type=int, default=0,
                    help="도로 조각 추론: 도로 부분을 가로 N 조각으로 더 돌려 작은 균열을 찾는다 (handoff 26장). "
                         "기본 0 = 끔. 시연 후보 2 (프레임 전체 + 2조각 = 모델 3번, 그만큼 느려진다)")
    ap.add_argument("--tiles-bottom", type=float, default=0.8,
                    help="조각의 아래 끝 (프레임 높이 비율). 0.8 이면 보닛·앞유리 아래쪽은 뺀다")
    ap.add_argument("--tile-classes", type=lambda s: [c for c in s.split(",") if c], default=["crack"],
                    help="조각에서 받을 클래스 (쉼표로). 나머지는 프레임 전체 결과만 쓴다. 기본 crack")
    ap.add_argument("--detect-every", type=int, default=1, help="결함 탐지 주기(프레임)")
    ap.add_argument("--lane-every", type=int, default=0,
                    help="차선 마모 주기(프레임). 기본 0 = 재지 않음 (시연에서 끔 확정, handoff 17-6). 실험용 3")
    ap.add_argument("--quality-every", type=int, default=1,
                    help="품질 판별 주기(프레임). 탐지·차선 프레임에서는 주기와 상관없이 먼저 잰다")
    ap.add_argument("--merge-gap", type=float, default=GAP_S,
                    help="이만큼(초) 안 보이면 결함 트랙이 끝난 것으로 보고 관측 1건을 보낸다")
    ap.add_argument("--no-merge", action="store_true", help="연속 프레임을 합치지 않고 탐지마다 보낸다 (비교용)")
    ap.add_argument("--speed", type=float, default=1.0, help="재생 배속. 0이면 기다리지 않는다")
    ap.add_argument("--gps-noise", type=float, default=0.0, help="GPS 에 더할 오차의 표준편차(m)")
    ap.add_argument("--brightness", type=float, default=1.0, help="밝기 배율")
    ap.add_argument("--start-offset", type=int, default=0, help="건너뛸 앞쪽 프레임 수")
    ap.add_argument("--seed", type=int, default=None, help="GPS 오차 난수 씨앗")
    ap.add_argument("--dry-run", action="store_true", help="서버로 보내지 않고 화면에만 출력")
    ap.add_argument("--no-clip", action="store_true", help="결함 구간 클립을 만들지 않는다 (사진·메타만)")
    ap.add_argument("--keep-dir", default="server_data/blackbox",
                    help="보낸 사진·클립을 남길 폴더 (<폴더>/<기기>/<주행>/). 빈 문자열이면 남기지 않는다")
    ap.add_argument("--no-quality", action="store_true", help="품질 판별을 끈다 (실험용)")
    ap.add_argument("--live-fps", type=float, default=5.0,
                    help="블랙박스 화면에 초당 몇 장 내보낼지. 0이면 화면을 쓰지 않는다")
    ap.add_argument("--live-dir", default=str(LIVE_DIR), help="블랙박스 화면 파일을 둘 폴더 (화면 ui/ 가 읽는다)")
    args = ap.parse_args(argv)
    if args.detect_every < 1 or args.quality_every < 1 or args.lane_every < 0:
        ap.error("주기는 1 이상이어야 한다")
    if args.speed < 0:
        ap.error("--speed 는 0 이상이어야 한다")
    if args.road_tiles < 0 or not 0.45 < args.tiles_bottom <= 1:
        ap.error("--road-tiles 는 0 이상, --tiles-bottom 은 0.45 초과 1 이하")
    unknown = set(args.tile_classes) - set(CLASS_NAMES)
    if unknown or not args.tile_classes:
        ap.error(f"--tile-classes 는 {', '.join(CLASS_NAMES)} 중에서 (받은 값 {args.tile_classes})")
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

    try:
        detector = make_detector(args)
    except (OSError, ValueError, KeyError) as e:
        print(e, file=sys.stderr)
        return {"frames": 0}
    if args.detector == "fake" and args.lane_every > 0:
        args.lane_every = 0  # 차선 마모도 라벨에서 나온다
    print(f"[{args.device_id}] 탐지기 {detector.name}" + (f", 기록 {args.record}" if args.record else ""))
    sender = Sender(args.server, dry_run=args.dry_run,
                    keep_dir=Path(args.keep_dir) / args.device_id if args.keep_dir else None)
    rng = random.Random(args.seed)
    started = datetime.now(KST).replace(microsecond=0)
    if not sender.start_trip(args.device_id, started):
        print(f"[{args.device_id}] 서버에 주행을 열지 못했다 ({args.server}). --dry-run 이면 서버 없이 돈다", file=sys.stderr)
        sender.close()
        return {"frames": 0}
    clipper = ClipCutter(args.video) if args.video and not args.no_clip else None
    first_t: float | None = None
    pacer = Pacer()
    sched = Scheduler(args.detect_every, args.lane_every, args.quality_every, quality=not args.no_quality)
    merger = None if args.no_merge else TrackMerger(gap_s=args.merge_gap)
    skipped_no_gps = skipped_quality = quality_bad = 0

    lane_meter = LaneWear() if args.lane_every > 0 else None
    live = (LiveView(args.device_id, Path(args.video).name if args.video else args.segment, SENT_TYPES,
                     out_dir=args.live_dir, fps=args.live_fps) if args.live_fps > 0 else None)

    def send(d: Detection, image, fix, i: int, name: str, note: str = "") -> None:
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
        # 클립 시각은 영상 시각(fix.time_s). 차선 마모도 결함 구간이라 같이 자른다
        clip = clipper.cut(fix.time_s) if clipper is not None and fix.time_s is not None else None
        res = sender.send(obs, crop_snapshot(image, d.bbox), clip)
        tag = "" if res is None else f" → 결함 {str(res['defect_id'])[:8]}{' (새 결함)' if res['is_new_defect'] else ''}"
        print(f"  {name}: {d.type} {d.confidence:.2f}{note}{tag}")

    def send_tracks(tracks) -> None:
        """끝난 트랙마다 가장 신뢰도 높은 프레임으로 관측 1건."""
        for tr in tracks:
            image, fix, i, name = tr.payload
            send(tr.best, image, fix, i, name, note=f" (탐지 {tr.hits}회 합침, {tr.last_t - tr.start_t:.1f}초)")

    print(f"[{args.device_id}] {args.video or args.segment} 프레임 {len(frames)}장, 탐지 주기 {args.detect_every}, "
          f"차선 주기 {args.lane_every}, 품질 주기 {args.quality_every}, "
          f"{'합치지 않음' if merger is None else f'합치기 {args.merge_gap:g}초'}, 배속 {args.speed}"
          f"{', 보내지 않음' if args.dry_run else ''}")
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
            if merger:
                send_tracks(merger.expire(ft))
            steps = sched.plan(i)
            run_detect, run_lane = steps.detect, steps.lane
            show = live is not None and live.due(ft)
            if not (steps.any or show):
                continue

            image = adjust_brightness(frame.load(), args.brightness)
            note = ""
            if steps.quality:
                with sched.run("quality"):
                    q = measure(image)
                if not q.ok:
                    quality_bad += 1
                    if run_detect or run_lane:
                        skipped_quality += 1
                        run_detect = run_lane = False
                        print(f"  {frame.name}: 품질 나쁨({q.reason}), 탐지 건너뜀 "
                              f"[밝기 {q.brightness:.0f}, 선명도 {q.sharpness:.0f}, 대비 {q.contrast:.0f}]")
                    note = f"품질 나쁨({q.reason}), 탐지 건너뜀"
            if run_detect:
                with sched.run("detect"):
                    dets = detector.detect(image, frame=frame.name, t=ft)
                if live:
                    live.add(dets, ft)
                sent_dets = [d for d in dets if d.type in SENT_TYPES]
                if merger:
                    send_tracks(merger.update(sent_dets, ft, payload=(image, fix, i, frame.name)))
                else:
                    for d in sent_dets:
                        send(d, image, fix, i, frame.name)

            if run_lane:
                # 차선 마모는 LaneWearTracker 가 이미 마모 구간마다 1건만 내므로 합치지 않는다
                with sched.run("lane"):
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
        if merger:
            send_tracks(merger.flush())  # 열린 트랙을 모두 내보낸 뒤 주행을 닫는다
        # 주행 중에는 알리지 않고 끝난 뒤에 한 번만 알린다 (주행을 닫으면 시민의 확인 목록이 생긴다)
        end = sender.end_trip()
        if end is not None:
            print(f"[{args.device_id}] 주행 종료: 시민에게 보여줄 결함 {end['finding_count']}건")
        if live:
            live.finish(sender.sent)
        sender.close()
        if clipper is not None:
            clipper.close()

    stages = sched.stats()
    merged = (f"탐지 {merger.detections}회 → 트랙 {merger.closed}개" if merger else "합치지 않음")
    print(f"[{args.device_id}] 끝. 탐지한 프레임 {stages['detect']['runs']}, {merged}, 보낸 관측 {sender.sent}건 "
          f"({sender.bytes_sent / 1024:.1f}KB: 메타 {sender.bytes_meta / 1024:.1f} · 사진 {sender.bytes_snapshot / 1024:.1f} · "
          f"클립 {sender.clips}개 {sender.bytes_clip / 1024:.1f}), 전송 실패 {sender.failed}, 위치 없음 {skipped_no_gps}, "
          f"품질 나쁨 {quality_bad}프레임 (그중 분석 건너뜀 {skipped_quality}), 탐지 평균 {stages['detect']['ms_avg']:.0f}ms/장")
    print(f"[{args.device_id}] {sched.summary()}")
    return {
        "frames": len(frames),
        "trip_id": None if args.dry_run else str(sender.trip_id),
        "detected_frames": stages["detect"]["runs"],
        "sent": sender.sent,
        "bytes_sent": sender.bytes_sent,
        "bytes_meta": sender.bytes_meta,
        "bytes_snapshot": sender.bytes_snapshot,
        "bytes_clip": sender.bytes_clip,
        "clips": sender.clips,
        "clip_codec": clipper.codec if clipper is not None else "",
        "clips_skipped_overlap": clipper.skipped_overlap if clipper is not None else 0,
        "failed": sender.failed,
        "skipped_no_gps": skipped_no_gps,
        "skipped_quality": skipped_quality,
        "quality_bad": quality_bad,
        "detect_ms_avg": stages["detect"]["ms_avg"],
        "detections": merger.detections if merger else None,
        "tracks": merger.closed if merger else None,
        "stages": stages,
    }


def main(argv=None) -> int:
    result = drive(parse_args(argv))
    return 1 if result.get("frames", 0) == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
