"""통신량 절감률과 중복 정리 비율 (계획 4.5, PRD 12장, AC-5·AC-6).

가상 블랙박스(edge.blackbox.run 의 drive)로 영상 하나를 기기 여러 대가 지나간 것처럼 차례로 재생하고, 블랙박스가
보내는 바이트(메타 JSON + 결함 사진 + 결함 구간 클립)를 기준값 두 가지와 비교한다 (계획 2장 '통신량 기준값').

  (a) 내려받은 영상 파일 크기 (재생한 시간만큼 비례). 유튜브가 다시 압축해 원본 블랙박스보다 작다 → 절감률이 낮게 나온다
  (b) 일반 블랙박스 1080p 녹화 비트레이트(가정값, 기본 10 Mbps) × 재생한 시간

서버로 보내면(--server, 기본) 그 주행들의 관측이 서버에서 결함 몇 건으로 묶였는지도 센다 (중복 정리 비율 =
1 − 결함 수 ÷ 관측 수). 기기 조건은 edge/blackbox/variants.py 의 VARIANTS (bb-02·03 은 GPS 오차 7m 등).
--dry-run 이면 서버 없이 통신량만 잰다.

탐지기:
  --model <onnx>      실제 탐지 (나머지 인자는 edge.blackbox.run 으로 그대로 넘긴다. 예: --detect-every 10)
  --fake-every <초>   모델 없이 잴 때. 영상 시작 --fake-first 초부터 <초>마다 결함 1개가 --fake-visible 초 동안 화면
                      아래쪽에 보인다고 치고, 그동안 분석하는 프레임마다 탐지로 낸다 (블랙박스 합치기 전의 실제 탐지와 같은
                      모양). 품질 판별·차선 마모는 끈다. 결과는 이 가정에 좌우된다 — 결과에 가정을 함께 적는다.

  # 통신량만 (서버 없이)
  uv run python experiments/bandwidth.py --video data/videos/gn_a.mp4 --gps data/gps/gn_a.gps.csv \\
      --fake-every 60 --devices 1 --dry-run --detect-every 10
  # 3대 + 서버 (scripts/demo.sh up 뒤. 측정 뒤 scripts/demo.sh reset)
  uv run python experiments/bandwidth.py --video data/videos/gn_a_2050-2510.mp4 --gps <구간 GPS> \\
      --model ai/models/detector_v2.onnx --devices 3 --detect-every 10 --json experiments/bandwidth_gn_a.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.blackbox import run  # noqa: E402
from edge.blackbox.detect import Detection  # noqa: E402
from edge.blackbox.source import VideoSource  # noqa: E402
from edge.blackbox.variants import VARIANTS, run_args  # noqa: E402

BLACKBOX_MBPS = 10.0


class _Clock:
    t: float | None = None  # 지금 재생 중인 프레임의 영상 시각


def _timed_open_source(open_source):
    """프레임을 내줄 때 그 시각을 _Clock 에 적는다 (가짜 탐지기가 시각을 알게)."""

    def wrapped(args):
        src = open_source(args)

        class Timed:
            def __len__(self):
                return len(src)

            def __iter__(self):
                for f in src:
                    _Clock.t = f.fix.time_s if f.fix else None
                    yield f

        return Timed()

    return wrapped


class ScheduledDetector:
    """first 초부터 every 초마다 결함 1개가 visible 초 동안 보인다. 유형은 포트홀·균열을 번갈아."""

    name = "fake-schedule"

    def __init__(self, every: float, visible: float, first: float):
        self.every, self.visible, self.first = every, visible, first

    def __call__(self, *a, **k):  # run.Detector(...) 자리에 들어간다
        return self

    def detect(self, image, **kw):
        t = kw.get("t", _Clock.t)
        if t is None or t < self.first:
            return []
        k, r = divmod(t - self.first, self.every)
        if r >= self.visible:
            return []
        h, w = image.shape[:2]
        box = (int(w * 0.40), int(h * 0.70), int(w * 0.55), int(h * 0.82))
        return [Detection("pothole" if int(k) % 2 == 0 else "crack", 0.8, box)]


def server_counts(trip_ids: list[str]) -> dict:
    """그 주행들의 관측이 서버에서 결함 몇 건으로 묶였는지 (DB 를 직접 읽는다. 서버와 같은 DATABASE_URL)."""
    from server import db

    with db.connect() as conn:
        r = conn.execute(
            """SELECT count(*) AS observations, count(DISTINCT defect_id) AS defects,
                      coalesce(sum(coalesce(snapshot_bytes, 0) + coalesce(clip_bytes, 0)), 0)::bigint AS stored_bytes,
                      count(clip_path) AS clips
               FROM observations WHERE trip_id = ANY(%s::uuid[])""", (trip_ids,)).fetchone()
        multi = conn.execute(
            """SELECT count(*) AS n FROM (SELECT defect_id FROM observations WHERE trip_id = ANY(%s::uuid[])
               GROUP BY defect_id HAVING count(DISTINCT vehicle_id) >= 2) x""", (trip_ids,)).fetchone()["n"]
        obs = conn.execute(
            """SELECT v.label AS vehicle, o.defect_type::text AS type, o.confidence, ST_Y(o.location::geometry) AS lat,
                      ST_X(o.location::geometry) AS lon, o.t_s
               FROM observations o JOIN vehicles v ON v.id = o.vehicle_id
               WHERE o.trip_id = ANY(%s::uuid[]) ORDER BY o.t_s, v.label""", (trip_ids,)).fetchall()
    # 영상 시각 순 (기기를 차례로 돌려도 같은 곳을 함께 지나간 것처럼). experiments/dedup_distance.py --replay 가 반경을 바꿔 다시 묶는다
    return {**r, "defects_seen_by_2plus_vehicles": multi, "observations_list": obs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--gps", default=None)
    ap.add_argument("--devices", type=int, default=3, choices=range(1, len(VARIANTS) + 1))
    ap.add_argument("--bitrate-mbps", type=float, default=BLACKBOX_MBPS, help="기준값 (b) 의 블랙박스 비트레이트 (가정값)")
    ap.add_argument("--source-bytes", type=int, default=None,
                    help="기준값 (a) 파일 크기를 직접 줄 때 (기본: --video 파일 크기)")
    ap.add_argument("--fake-every", type=float, default=None, help="모델 없이: 몇 초마다 결함 1개")
    ap.add_argument("--fake-visible", type=float, default=2.0, help="가짜 결함이 화면에 보이는 시간(초)")
    ap.add_argument("--fake-first", type=float, default=5.0, help="첫 가짜 결함 시각(초)")
    ap.add_argument("--note", default="", help="결과 JSON 에 함께 남길 입력 설명 (GPS 출처·모델 버전 등)")
    ap.add_argument("--json", default=None, help="결과를 저장할 파일")
    ap.add_argument("--verbose", action="store_true", help="블랙박스 출력을 그대로 보인다")
    a, rest = ap.parse_known_args(argv)
    if a.fake_every is None and "--model" not in rest and "--detector" not in rest:
        ap.error("--model <onnx> 또는 --fake-every <초> 를 준다")

    if a.fake_every is not None:
        det = ScheduledDetector(a.fake_every, a.fake_visible, a.fake_first)
        run.open_source = _timed_open_source(run.open_source)
        if hasattr(run, "make_detector"):  # --detector 를 고르는 run.py
            run.make_detector = lambda args: det
        else:
            run.Detector = det
        rest = [*rest, "--no-quality", "--lane-every", "0"]
        if "--model" not in rest:
            rest += ["--model", "fake"]
    base = ["--video", a.video, *(["--gps", a.gps] if a.gps else []), "--speed", "0", "--live-fps", "0", *rest]

    src = VideoSource(a.video, a.gps)
    duration = len(src) / src.fps
    file_bytes = a.source_bytes or Path(a.video).stat().st_size
    started = datetime.now().isoformat(timespec="seconds")

    devices = []
    for v in VARIANTS[: a.devices]:
        args = run.parse_args([*base, *run_args(v)])
        out = io.StringIO()
        with contextlib.redirect_stdout(sys.stdout if a.verbose else out):
            r = run.drive(args)
        if r.get("frames", 0) == 0:
            print(out.getvalue()[-2000:], file=sys.stderr)
            print(f"{v['device_id']}: 재생하지 못했다", file=sys.stderr)
            return 1
        r["device_id"] = v["device_id"]
        devices.append(r)
        print(f"{v['device_id']}: 보낸 관측 {r['sent']}건, {r['bytes_sent'] / 1e6:.2f}MB "
              f"(메타 {r['bytes_meta'] / 1e3:.1f}KB · 사진 {r['bytes_snapshot'] / 1e6:.2f}MB · "
              f"클립 {r['clips']}개 {r['bytes_clip'] / 1e6:.2f}MB, 겹쳐서 안 자른 탐지 {r['clips_skipped_overlap']}), "
              f"전송 실패 {r['failed']}", flush=True)

    per_device = sum(d["bytes_sent"] for d in devices) / len(devices)
    base_a = file_bytes  # 기기 한 대가 영상 한 번 = 파일 하나
    base_b = a.bitrate_mbps * 1e6 / 8 * duration
    result = {
        "measured_at": started,
        "video": Path(a.video).name,
        "video_duration_s": round(duration, 1),
        "video_fps": round(src.fps, 2),
        "gps": Path(a.gps).name if a.gps else "(영상 옆 .gps.csv)",
        "fake": ({"every": a.fake_every, "visible": a.fake_visible, "first": a.fake_first}
                 if a.fake_every is not None else None),
        "detector": (f"fake: {a.fake_first:g}초부터 {a.fake_every:g}초마다 결함 1개, {a.fake_visible:g}초 보임"
                     if a.fake_every is not None else " ".join(x for x in rest if x.endswith(".onnx")) or "?"),
        "run_args": rest,
        "note": a.note,
        "devices": devices,
        "clip_codec": devices[0].get("clip_codec", ""),
        "bytes_sent_per_device": round(per_device),
        "baseline_a_file_bytes": base_a,
        "baseline_b_bytes": round(base_b),
        "bitrate_mbps_b": a.bitrate_mbps,
        "reduction_a": 1 - per_device / base_a,
        "reduction_b": 1 - per_device / base_b,
    }
    dry = "--dry-run" in rest
    if not dry:
        trip_ids = [d["trip_id"] for d in devices if d.get("trip_id")]
        s = server_counts(trip_ids)
        result["server"] = s
        if s["observations"]:
            result["dedup_ratio"] = 1 - s["defects"] / s["observations"]

    print()
    print("| 입력 | 재생 시간 | 기기 | 보낸 관측(1대 평균) | 보낸 용량(1대 평균) | (a) 파일 | 절감률 (a) | (b) "
          f"{a.bitrate_mbps:g} Mbps | 절감률 (b) |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| {result['video']} | {duration:.0f}초 | {len(devices)} | "
          f"{sum(d['sent'] for d in devices) / len(devices):.1f}건 | {per_device / 1e6:.2f}MB | "
          f"{base_a / 1e6:.1f}MB | {result['reduction_a'] * 100:.2f}% | {base_b / 1e6:.0f}MB | "
          f"{result['reduction_b'] * 100:.2f}% |")
    if "server" in result:
        s = result["server"]
        print(f"\n서버: 관측 {s['observations']}건 → 결함 {s['defects']}건 "
              f"(중복 정리 비율 {result.get('dedup_ratio', 0) * 100:.1f}%, 2대 이상이 본 결함 "
              f"{s['defects_seen_by_2plus_vehicles']}건), 서버가 받은 사진·클립 {s['stored_bytes'] / 1e6:.2f}MB")
    if a.json:
        Path(a.json).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\n저장: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
