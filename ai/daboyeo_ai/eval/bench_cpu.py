"""CPU 1·2코어 처리 시간 측정 (계획 4.2, AC-2 → `experiments/bench_cpu.csv`).

두 가지로 쓴다.

1) `run` — 단계별 프레임당 시간을 직접 잰다 (탐지 `OnnxDetector.detect`, 차선 마모 `measure + tracker.update`).
   threads × 해상도마다 median·p95 ms, FPS, 그리고 30 fps 영상 1분을 N 프레임마다 분석할 때 드는 계산 시간
   (순수 계산: 프레임당 평균 ms × 분석 프레임 수. 실제 주기 스케줄러는 정현우 담당 `scheduler.py`).

    uv run python -m daboyeo_ai.eval.bench_cpu run --model ai/models/detector_v2.onnx \\
        --src data/lane_calib/frames --out experiments/bench_cpu.csv

2) `log` — 정현우의 가상 블랙박스 측정 로그 JSONL (계획 4장 `{"frame","t","stage","ran","ms"}`) 을 요약한다.
   주기 적용 전/후 로그를 하나씩 주면 단계별 실행 비율·ms·FPS·영상 1분당 계산 시간을 나란히 낸다.

    uv run python -m daboyeo_ai.eval.bench_cpu log before=logs/no_interval.jsonl after=logs/interval.jsonl \\
        --out experiments/bench_cpu_log.csv

CPU 제한: ONNX Runtime `intra_op_num_threads`(=threads)·`inter_op_num_threads`=1, `OMP_NUM_THREADS`,
`cv2.setNumThreads`. Linux 에서는 `os.sched_setaffinity` 로 실제 코어도 고정한다 (macOS·Windows 는 스레드 수
제한만). 개발용 Mac(M4 Pro 등)은 목표 노트북·블랙박스 CPU 보다 훨씬 빠르다 — 공식 수치는 노트북에서 같은 명령으로 잰다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

VIDEO_FPS = 30.0
INTERVALS = (1, 3, 5, 10, 15)
RESOLUTIONS = ((640, 360), (1280, 720))
NOTE = ("참고값: 측정 장비가 목표 노트북·블랙박스 CPU 보다 빠를 수 있다. "
        "공식 수치는 노트북에서 같은 명령으로 잰다. 주기별 값은 프레임당 평균 ms 로 계산한 것이다.")


# ---------------------------------------------------------------- 계산 (테스트 대상, 시간 측정 없음)

def stats_ms(ms: Sequence[float]) -> dict:
    """프레임당 ms 목록 → n, median, p95, mean, fps(=1000/mean)."""
    a = np.asarray(ms, dtype=float)
    if a.size == 0:
        return {"n": 0, "median_ms": None, "p95_ms": None, "mean_ms": None, "fps": None}
    mean = float(a.mean())
    return {"n": int(a.size), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": mean, "fps": 1000.0 / mean if mean > 0 else None}


def compute_s_per_video_min(mean_ms: float, interval: int, video_fps: float = VIDEO_FPS) -> float:
    """video_fps 영상 1분을 `interval` 프레임마다 한 번 분석할 때 드는 계산 시간(초).

    1분 = 60 × video_fps 프레임, 그중 ceil(…/interval) 프레임을 분석. 60 초 이하면 실시간으로 따라간다.
    """
    if interval < 1:
        raise ValueError("interval 은 1 이상")
    n = -(-round(60 * video_fps) // interval)
    return n * mean_ms / 1000.0


def interval_columns(mean_ms: float | None, intervals: Iterable[int] = INTERVALS,
                     video_fps: float = VIDEO_FPS) -> dict:
    """주기마다 `s_per_min_iN` (영상 1분당 계산 초)와 `rt_iN` (= 계산 초 ÷ 60, 1 이하면 실시간 가능)."""
    out = {}
    for k in intervals:
        s = None if mean_ms is None else compute_s_per_video_min(mean_ms, k, video_fps)
        out[f"s_per_min_i{k}"] = None if s is None else round(s, 2)
        out[f"rt_i{k}"] = None if s is None else round(s / 60.0, 3)
    return out


def read_log(lines: Iterable[str]) -> list[dict]:
    """측정 로그 JSONL 줄 → dict 목록. 빈 줄은 건너뛰고 `frame`·`stage` 가 없으면 오류."""
    rows = []
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "frame" not in r or "stage" not in r:
            raise ValueError(f"{i}번째 줄: frame·stage 필드가 필요하다: {line}")
        rows.append(r)
    return rows


def summarize_log(rows: Sequence[dict], video_fps: float | None = None) -> list[dict]:
    """측정 로그 → 단계별 요약 (+ 모든 단계 합 `total`).

    - `ran` 이 없으면 `ms` 가 있을 때 실행한 것으로 본다.
    - 영상 길이: video_fps 를 주면 (서로 다른 frame 수 ÷ fps), 아니면 t 범위 + 프레임 간격 중앙값.
    - fps = 실행 횟수 ÷ 그 단계 계산 초 (그 단계만 돌렸을 때의 처리 속도),
      s_per_video_min = 그 단계 계산 초 ÷ 영상 분, frames_per_s = 영상 프레임 수 ÷ 모든 단계 계산 초 (total 만).
    """
    if not rows:
        return []
    frames = sorted({int(r["frame"]) for r in rows})
    if video_fps:
        duration_s = len(frames) / video_fps
    else:
        ts = sorted({float(r["t"]) for r in rows if r.get("t") is not None})
        if len(ts) < 2:
            raise ValueError("영상 길이를 알 수 없다: t 가 2개 이상이거나 --video-fps 가 필요하다")
        duration_s = ts[-1] - ts[0] + float(np.median(np.diff(ts)))
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stage[str(r["stage"])].append(r)

    def ran(r: dict) -> bool:
        return bool(r.get("ran", r.get("ms") is not None))

    out = []
    all_ms: list[float] = []
    for stage in sorted(by_stage):
        rs = by_stage[stage]
        ms = [float(r["ms"]) for r in rs if ran(r) and r.get("ms") is not None]
        all_ms += ms
        n_frames = len({int(r["frame"]) for r in rs})
        out.append(_log_row(stage, n_frames, ms, duration_s))
    total = _log_row("total", len(frames), all_ms, duration_s)
    total_s = sum(all_ms) / 1000.0
    total["frames_per_s"] = round(len(frames) / total_s, 2) if total_s > 0 else None
    out.append(total)
    return out


def _log_row(stage: str, n_frames: int, ms: list[float], duration_s: float) -> dict:
    s = stats_ms(ms)
    compute_s = sum(ms) / 1000.0
    return {
        "stage": stage, "n_frames": n_frames, "n_ran": len(ms),
        "ran_ratio": round(len(ms) / n_frames, 3) if n_frames else None,
        "median_ms": _r(s["median_ms"]), "p95_ms": _r(s["p95_ms"]), "mean_ms": _r(s["mean_ms"]),
        "fps": round(len(ms) / compute_s, 2) if compute_s > 0 else None,
        "video_s": round(duration_s, 2),
        "s_per_video_min": round(compute_s / (duration_s / 60.0), 2) if duration_s > 0 else None,
        "frames_per_s": None,
    }


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------- 장비 정보·CPU 제한

def machine_info() -> dict:
    import cv2

    try:
        import onnxruntime as ort
        ort_ver = ort.__version__
    except ImportError:  # pragma: no cover
        ort_ver = None
    return {
        "cpu": _cpu_name(), "physical_cores": _physical_cores(), "logical_cores": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}", "machine": platform.machine(),
        "python": platform.python_version(), "onnxruntime": ort_ver, "opencv": cv2.__version__,
        "numpy": np.__version__,
        # 측정 시작 때 부하 (다른 프로그램이 CPU 를 쓰고 있었는지)
        "load_avg_1m": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
    }


def _cpu_name() -> str:
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Windows":
            import winreg  # type: ignore[import-not-found]

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:  # noqa: BLE001 — 장비 이름을 못 읽어도 측정은 한다
        pass
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    try:
        if platform.system() == "Darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True))
        if platform.system() == "Linux":
            pairs = set()
            phys = core = None
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    pairs.add((phys, core))
            return len(pairs) or None
    except Exception:  # noqa: BLE001
        pass
    return None


def limit_cpu(threads: int) -> str:
    """이 프로세스가 쓸 CPU 를 줄인다. 돌려주는 값 = 어떤 방법을 썼는지 (CSV `cpu_limit` 열)."""
    import cv2

    os.environ["OMP_NUM_THREADS"] = str(threads)
    cv2.setNumThreads(threads)
    how = f"ort_threads={threads},cv2={threads}"
    if hasattr(os, "sched_setaffinity"):
        try:
            cpus = sorted(os.sched_getaffinity(0))[:threads]
            os.sched_setaffinity(0, cpus)
            how += f",affinity={cpus}"
        except OSError:
            pass
    return how


def _restore_affinity(all_cpus: set[int] | None) -> None:
    if all_cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, all_cpus)
        except OSError:
            pass


# ---------------------------------------------------------------- 측정

def load_frames(src: Path, max_frames: int, every_s: float) -> list[np.ndarray]:
    from daboyeo_ai.eval.lane_wear_calib import iter_frames

    frames = []
    for _, _, _, img in iter_frames(src, every_s=every_s):
        frames.append(img)
        if len(frames) >= max_frames:
            break
    if not frames:
        raise SystemExit(f"프레임이 없다: {src}")
    return frames


def _time_each(fn, frames: Sequence[np.ndarray], warmup: int, repeat: int) -> list[float]:
    for i in range(warmup):
        fn(frames[i % len(frames)])
    ms = []
    for _ in range(repeat):
        for f in frames:
            t0 = time.perf_counter()
            fn(f)
            ms.append((time.perf_counter() - t0) * 1000.0)
    return ms


def bench(model: Path, frames_src: Sequence[np.ndarray], threads_list: Sequence[int],
          resolutions: Sequence[tuple[int, int]], warmup: int = 10, repeat: int = 1,
          intervals: Sequence[int] = INTERVALS, video_fps: float = VIDEO_FPS, log=print) -> list[dict]:
    import cv2

    from daboyeo_ai.inference.onnx_detector import OnnxDetector
    from daboyeo_ai.lane_wear import LaneWearMeter, LaneWearTracker

    all_cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    rows = []
    for w, h in resolutions:
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if f.shape[1::-1] != (w, h) else f
                  for f in frames_src]
        # 차선 마모: 항상 1스레드 (OpenCV 만 쓰고 탐지와 같은 코어에서 차례로 돈다)
        how = limit_cpu(1)
        meter, tracker = LaneWearMeter(), LaneWearTracker()
        lane_ms = _time_each(lambda f, m=meter, tr=tracker: tr.update(m.measure(f)), frames, warmup, repeat)
        _restore_affinity(all_cpus)
        lane_how = how.replace("ort_threads=1,", "")  # 차선 마모는 ONNX Runtime 을 쓰지 않는다
        rows.append(_bench_row("lane_wear", 1, w, h, lane_ms, lane_how, intervals, video_fps))
        log(_fmt(rows[-1]))
        for n in threads_list:
            how = limit_cpu(n)
            det = OnnxDetector(model, threads=n)
            det_ms = _time_each(det.detect, frames, warmup, repeat)
            _restore_affinity(all_cpus)
            rows.append(_bench_row("detect", n, w, h, det_ms, how, intervals, video_fps))
            log(_fmt(rows[-1]))
            # 한 프레임에 둘 다 돌린 경우 (같은 프레임 순서로 잰 값을 더함)
            k = min(len(det_ms), len(lane_ms))
            both = [a + b for a, b in zip(det_ms[:k], lane_ms[:k])]
            rows.append(_bench_row("detect+lane_wear", n, w, h, both, how, intervals, video_fps))
            log(_fmt(rows[-1]))
            del det
    return rows


def _bench_row(stage, threads, w, h, ms, how, intervals, video_fps) -> dict:
    s = stats_ms(ms)
    return {"stage": stage, "threads": threads, "width": w, "height": h, "n": s["n"],
            "median_ms": _r(s["median_ms"]), "p95_ms": _r(s["p95_ms"]), "mean_ms": _r(s["mean_ms"]),
            "fps": _r(s["fps"], 1), **interval_columns(s["mean_ms"], intervals, video_fps), "cpu_limit": how}


def _fmt(r: dict) -> str:
    return (f"{r['stage']:>16} t={r['threads']} {r['width']}x{r['height']}: median {r['median_ms']} ms, "
            f"p95 {r['p95_ms']} ms, {r['fps']} fps, 1분당 {r['s_per_min_i1']} s(매 프레임)")


def write_csv(rows: Sequence[dict], out: Path, extra: dict | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    fields = list(rows[0]) + list(extra)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, **extra})


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="단계별 프레임당 시간을 직접 잰다")
    r.add_argument("--model", type=Path, required=True, help="탐지 ONNX (예: ai/models/detector_v2.onnx)")
    r.add_argument("--src", type=Path, required=True, help="프레임 폴더 또는 영상 파일")
    r.add_argument("--out", type=Path, default=Path("experiments/bench_cpu.csv"))
    r.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    r.add_argument("--res", nargs="+", default=[f"{w}x{h}" for w, h in RESOLUTIONS], help="예: 640x360 1280x720")
    r.add_argument("--max-frames", type=int, default=300)
    r.add_argument("--every-s", type=float, default=1.0, help="영상일 때 몇 초마다 프레임을 뽑을지")
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--video-fps", type=float, default=VIDEO_FPS)
    r.add_argument("--intervals", type=int, nargs="+", default=list(INTERVALS))

    lg = sub.add_parser("log", help="가상 블랙박스 측정 로그 JSONL 요약 (주기 적용 전/후)")
    lg.add_argument("logs", nargs="+", help="LABEL=path.jsonl 또는 path.jsonl (라벨=파일 이름)")
    lg.add_argument("--out", type=Path, default=Path("experiments/bench_cpu_log.csv"))
    lg.add_argument("--video-fps", type=float, default=None, help="주면 영상 길이 = 프레임 수 ÷ fps")

    a = ap.parse_args(argv)
    if a.cmd == "log":
        return _main_log(a)

    os.environ.setdefault("OMP_NUM_THREADS", "1")  # onnxruntime import 전에
    res = [tuple(int(v) for v in s.lower().split("x")) for s in a.res]
    info = machine_info()
    print(f"장비: {info['cpu']} ({info['physical_cores']} 코어), {info['os']}, onnxruntime {info['onnxruntime']}")
    frames = load_frames(a.src, a.max_frames, a.every_s)
    print(f"프레임 {len(frames)}장, warmup {a.warmup}, repeat {a.repeat}")
    rows = bench(a.model, frames, a.threads, res, a.warmup, a.repeat, a.intervals, a.video_fps)
    machine_cols = {"cpu": info["cpu"], "os": info["os"], "onnxruntime": info["onnxruntime"]}
    write_csv(rows, a.out, machine_cols)
    summary = {
        "note": NOTE, "machine": info, "model": a.model.name, "src": a.src.name, "n_frames": len(frames),
        "warmup": a.warmup, "repeat": a.repeat, "video_fps": a.video_fps, "intervals": a.intervals,
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows,
    }
    js = a.out.with_suffix(".json")
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {a.out}, {js}\n{NOTE}")
    return 0


def _main_log(a) -> int:
    out_rows = []
    for spec in a.logs:
        label, _, path = spec.rpartition("=") if "=" in spec else ("", "", spec)
        p = Path(path)
        label = label or p.stem
        with p.open(encoding="utf-8") as f:
            summ = summarize_log(read_log(f), a.video_fps)
        for row in summ:
            out_rows.append({"label": label, **row})
            print(f"{label:>10} {row['stage']:>12}: ran {row['n_ran']}/{row['n_frames']}, "
                  f"median {row['median_ms']} ms, {row['fps']} fps, 영상 1분당 {row['s_per_video_min']} s")
    if not out_rows:
        raise SystemExit("로그가 비었다")
    write_csv(out_rows, a.out)
    print(f"저장: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
