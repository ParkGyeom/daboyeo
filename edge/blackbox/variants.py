"""여러 차량 변형 (docs/DESIGN.md 5장, B7). 같은 구간을 기기 여러 대가 지나간 것처럼 동시에 재생한다.

기기마다 GPS 오차, 밝기, 시작 시점을 달리한다. 검증하는 것은 GPS 오차가 있어도 서버가 같은 결함을
하나로 묶는지다. 영상 하나에 3대까지 둔다. 시연용 시민은 6명(bb-01~06)이라 영상 2개째는
--first-device 4 로 bb-04~06 이 같은 조건(원본 · GPS 오차 7 m · 7 m + 2프레임 늦게)으로 지나간다.

  .venv/bin/python -m edge.blackbox.variants --model runs/<모델>.onnx --devices 3 --speed 4
  .venv/bin/python -m edge.blackbox.variants --video data/videos/v1.mp4 --model ai/models/detector_v2.onnx --devices 3
  (--video 도 --segment 도 없으면 로드마크 서울 우천 구간. 나머지 인자는 edge.blackbox.run 으로 그대로 넘긴다)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading

DEFAULT_SEGMENT = "0401_light_rainy_smooth:03000071-03000182"

# 기기별 조건. 1대째는 원본 그대로다.
VARIANTS: list[dict] = [
    {"device_id": "bb-01"},
    {"device_id": "bb-02", "gps_noise": 7.0, "brightness": 0.85, "seed": 2},
    {"device_id": "bb-03", "gps_noise": 7.0, "start_offset": 2, "seed": 3},
]

# macOS 가 여러 프로세스에서 동시에 네트워크 캐시를 만들 때 찍는 줄. 동작과 관계없다
_NOISE = ("cfurl_cache_response", "NSURLStorageURLCacheDB")


def run_args(variant: dict) -> list[str]:
    out = ["--device-id", variant["device_id"]]
    for key in ("gps_noise", "brightness", "start_offset", "seed"):
        if key in variant:
            out += [f"--{key.replace('_', '-')}", str(variant[key])]
    return out


def _pump(proc: subprocess.Popen, lock: threading.Lock, device_id: str) -> None:
    for line in proc.stdout:
        if not any(n in line for n in _NOISE):
            with lock:
                try:
                    sys.stdout.write(line if line.startswith("[") else f"[{device_id}]{line}")
                    sys.stdout.flush()
                except BrokenPipeError:  # 출력을 받는 쪽이 먼저 닫혀도 기기들은 끝까지 돈다
                    pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", type=int, default=3, choices=range(1, len(VARIANTS) + 1))
    ap.add_argument("--segment", default=DEFAULT_SEGMENT)
    ap.add_argument("--first-device", type=int, default=1, choices=range(1, 5),
                    help="첫 기기 번호. 4 면 bb-04 부터 (영상 2개째, scripts/demo.sh both)")
    ap.add_argument("--video", default=None, help="주면 로드마크 구간 대신 이 영상을 재생한다 (--gps 는 그대로 넘긴다)")
    args, rest = ap.parse_known_args(argv)
    source = ["--video", args.video] if args.video else ["--segment", args.segment]

    procs = []
    lock = threading.Lock()
    for i, v in enumerate(VARIANTS[: args.devices]):
        v = {**v, "device_id": f"bb-{args.first_device + i:02d}"}
        cmd = [sys.executable, "-u", "-m", "edge.blackbox.run", *source, *run_args(v), *rest]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=_pump, args=(p, lock, v["device_id"]), daemon=True)
        t.start()
        procs.append((p, t))
    try:
        codes = [p.wait() for p, _ in procs]
    except KeyboardInterrupt:
        for p, _ in procs:
            p.terminate()
        codes = [p.wait() for p, _ in procs]
    for _, t in procs:
        t.join(timeout=2)
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())
