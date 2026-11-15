"""처리 속도 측정 (A8, docs/DESIGN.md 8장).

가상 블랙박스와 같은 경로(프레임 읽기 → 품질 판별 → 탐지 → 사진 자르기)로 구간 하나를 처리하는 시간을 잰다.
ONNX Runtime 스레드 수(1, 2, 제한 없음)와 탐지 주기(1, 2, 3, 5)를 바꿔 가며 초당 처리 프레임 수를 낸다.
서버로는 보내지 않는다(--dry-run). 차선 마모는 끈다(탐지만 잰다).

다른 무거운 작업(학습 등)이 돌고 있으면 값이 틀어진다. 끝난 뒤에 잰다.
macOS 에는 taskset 이 없어서 CPU 제한은 ONNX Runtime 스레드 수로 대신한다. 프레임 읽기(PNG 디코딩)는 이 제한을 받지 않는다.

  .venv/bin/python experiments/speed.py --model runs/<모델>.onnx
"""

from __future__ import annotations

import argparse
import contextlib
import io
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.blackbox.run import drive, parse_args  # noqa: E402

SEGMENT = "0401_light_rainy_smooth:03000071-03000182"


def measure(model: str, threads: int | None, every: int, repeats: int) -> tuple[list[float], float, int]:
    fps, det_ms, frames = [], [], 0
    for _ in range(repeats):
        argv = ["--segment", SEGMENT, "--model", model, "--speed", "0", "--dry-run", "--lane-every", "0",
                "--detect-every", str(every)]
        argv += ["--threads", str(threads or 0)]  # 0 은 ONNX Runtime 기본값(제한 없음)
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            r = drive(parse_args(argv))
        dt = time.perf_counter() - t0
        frames = r["frames"]
        fps.append(frames / dt)
        det_ms.append(r["detect_ms_avg"])
    return fps, statistics.median(det_ms), frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    # 첫 실행은 디스크 캐시·모델 적재 때문에 느리다. 한 번 버린다
    measure(a.model, None, 1, 1)
    print(f"구간 {SEGMENT}, 반복 {a.repeats}회 중앙값 (최소~최대)")
    print("| 스레드 | 탐지 주기 | 초당 처리 프레임 | 탐지 1회 |")
    print("|---|---|---|---|")
    for threads in (1, 2, None):
        for every in (1, 2, 3, 5):
            fps, det, n = measure(a.model, threads, every, a.repeats)
            print(f"| {threads or '제한 없음'} | {every} | {statistics.median(fps):.1f} ({min(fps):.1f}~{max(fps):.1f}) "
                  f"| {det:.0f}ms |", flush=True)
    print(f"프레임 {n}장. 로드마크 원본은 초당 약 1장이다")


if __name__ == "__main__":
    main()
