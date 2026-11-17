"""가상 블랙박스의 탐지기 고르기 (`--detector onnx|fake|replay`, 계획 1.1·3.7).

세 탐지기 모두 `detect(image, frame=<프레임 이름>, t=<영상 시각 초>) -> list[Detection]` 하나만 내준다.
run.py 는 무엇이 고르게 되었는지 모르고 같은 방식으로 부르고 보낸다.

- onnx   (기본): ONNX 모델로 추론한다 (edge/blackbox/detect.py 의 Detector).
- fake   (1.1 걷는 골격): 수동 정답 라벨 CSV (`defect_id,t_start_s,t_end_s,defect_type,memo`,
         docs/ai-handoff.md 12장, eval_detector 가 읽는 것과 같은 파일) 를 읽어, 라벨마다 그 구간의 첫 분석
         프레임에서 그 유형을 1번 탐지한 것처럼 낸다. 박스·신뢰도는 고정값. 차선 마모도 라벨에서 나오므로
         run.py 는 fake 일 때 차선 측정기를 끈다. 서버로 가는지는 run.py 의 SENT_TYPES 가 정한다
         (시설물 guardrail_damage·sign_damage 라벨은 화면에만 그리고 보내지 않는다).
- replay (3.7): `--record` 로 남긴 탐지 기록(JSONL)을 읽어 추론 없이 같은 결과를 낸다. 모델 파일이 없거나
         시연 노트북 CPU 가 느려도 시연 흐름이 돈다. 기록의 영상 이름·탐지 주기가 지금 실행과 다르면 오류.

기록 파일 (JSONL, 첫 줄은 머리):

    {"kind": "daboyeo-detections", "version": 1, "model": "detector_v2.onnx", "video": "gn_a.mp4",
     "detect_every": 10, "created_at": "..."}
    {"frame": "gn_a@0.00s", "t": 0.0, "dets": [["pothole", 0.81, 812, 640, 990, 720], ...]}

replay 는 프레임 이름이 같으면 그 줄을, 없으면 (예: 시작 시점을 2프레임 늦춘 bb-03) 시각이 가장 가까운 줄을
기록 간격의 절반 안에서 쓴다. 밝기 등 다른 조건은 기록에 반영되지 않는다 (같은 탐지가 나온다).
"""

from __future__ import annotations

import bisect
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from edge.blackbox.detect import Detection, Detector

DETECTORS = ("onnx", "fake", "replay")
RECORD_KIND = "daboyeo-detections"
RECORD_VERSION = 2  # 2: 머리 줄에 탐지 기준(conf)을 적는다

# 라벨 유형(DefectType 값) → 블랙박스 탐지 유형(Detection.type, 탐지 클래스 이름 + lane_wear)
LABEL_TO_DETECTION = {
    "pothole": "pothole",
    "crack": "crack",
    "lane_wear": "lane_wear",
    "guardrail_damage": "guardrail",
    "sign_damage": "sign",
}
# 가짜 박스: 프레임 크기로 나눈 (x1, y1, x2, y2). 블랙박스 화면에서 그럴듯한 자리
FAKE_BOXES = {
    "pothole": (0.42, 0.72, 0.58, 0.84),
    "crack": (0.36, 0.66, 0.62, 0.80),
    "lane_wear": (0.30, 0.60, 0.70, 1.00),  # 차선 측정 영역 비슷하게 (서버로는 박스 없이 간다)
    "guardrail": (0.76, 0.48, 0.99, 0.70),
    "sign": (0.74, 0.12, 0.88, 0.34),
}
FAKE_CONFIDENCE = 0.9


class FrameDetector(Protocol):
    name: str

    def detect(self, image: np.ndarray, frame: str, t: float) -> list[Detection]: ...


class OnnxMode:
    def __init__(self, model: str, conf, threads: int = 1):
        self.name = Path(model).name
        self.conf = conf  # 기록 머리 줄에 남긴다 (replay 가 같은 기준인지 확인)
        self._det = Detector(model, conf=conf, threads=threads)

    def detect(self, image: np.ndarray, frame: str, t: float) -> list[Detection]:
        return self._det.detect(image)


def default_labels_path(video: str | Path) -> Path:
    """`experiments/gt_events/<영상 이름에서 확장자 뺀 것>.csv` (handoff 12장)."""
    return Path("experiments/gt_events") / f"{Path(video).stem}.csv"


class FakeDetector:
    """정답 라벨을 탐지처럼 낸다. 라벨 하나당 1번, 라벨 구간이 시작된 뒤 처음 부른 프레임에서.
    memo 가 `amb` 로 시작하는 애매한 라벨은 기본으로 내지 않는다 (확실한 정답만 서버로, handoff 18-4)."""

    def __init__(self, labels: str | Path, point_half_s: float = 1.0, include_amb: bool = False):
        from daboyeo_ai.eval.video_events import load_gt

        path = Path(labels)
        if not path.exists():
            raise FileNotFoundError(f"라벨 CSV 가 없다: {path} (--labels 로 준다)")
        self.name = f"fake:{path.name}"
        # t_end_s 를 비운 라벨은 load_gt 가 t_start_s ± point_half_s 창으로 바꿔 준다
        events = load_gt(path, point_half_s=point_half_s)
        self.skipped_amb = 0 if include_amb else sum(e.memo.lower().startswith("amb") for e in events)
        if not include_amb:
            events = [e for e in events if not e.memo.lower().startswith("amb")]
        self.events = sorted(events, key=lambda e: e.t_start)
        self._fired: set[int] = set()
        self._started = False

    @property
    def remaining(self) -> int:
        return len(self.events) - len(self._fired)

    def detect(self, image: np.ndarray, frame: str, t: float) -> list[Detection]:
        if not self._started:
            # 주행 시작(--start-offset) 전에 이미 끝난 라벨은 이 차량이 보지 못한 것으로 친다
            self._started = True
            self._fired.update(i for i, e in enumerate(self.events) if e.t_end < t)
        h, w = image.shape[:2]
        out = []
        for i, e in enumerate(self.events):
            if e.t_start > t:
                break
            if i in self._fired:
                continue
            self._fired.add(i)
            kind = LABEL_TO_DETECTION[e.defect_type]
            x1, y1, x2, y2 = FAKE_BOXES[kind]
            out.append(Detection(kind, FAKE_CONFIDENCE, (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))))
        return out


def _source_name(args) -> str:
    return Path(args.video).name if args.video else args.segment


class Recorder:
    """다른 탐지기를 감싸 결과를 JSONL 로 남긴다. 줄마다 flush 하므로 중간에 끊겨도 앞부분은 남는다."""

    def __init__(self, inner: FrameDetector, path: str | Path, source: str, detect_every: int):
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._write({
            "kind": RECORD_KIND, "version": RECORD_VERSION, "model": inner.name, "video": source,
            "detect_every": detect_every, "conf": getattr(inner, "conf", None),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })

    def _write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._fh.flush()

    def detect(self, image: np.ndarray, frame: str, t: float) -> list[Detection]:
        dets = self.inner.detect(image, frame, t)
        self._write({"frame": frame, "t": round(t, 4),
                     "dets": [[d.type, round(d.confidence, 4), *map(int, d.bbox)] for d in dets]})
        return dets

    def close(self) -> None:
        self._fh.close()


class ReplayDetector:
    def __init__(self, path: str | Path, source: str, detect_every: int, conf=None):
        """conf: 지금 쓰려는 탐지 기준. 주면 기록을 남길 때의 기준과 같아야 한다 (다르면 기록을 다시 뜬다)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"탐지 기록이 없다: {path} (onnx 로 한 번 --record {path} 를 붙여 돌린다)")
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        try:
            head = json.loads(lines[0]) if lines else None
        except json.JSONDecodeError:
            head = None
        if not isinstance(head, dict) or head.get("kind") != RECORD_KIND:
            raise ValueError(f"{path}: 탐지 기록 파일이 아니다 (첫 줄에 kind={RECORD_KIND!r} 가 없다)")
        if head.get("version") != RECORD_VERSION:
            raise ValueError(f"{path}: 기록 형식 버전 {head.get('version')} 은 읽지 못한다 (지원: {RECORD_VERSION}). onnx 로 --record 를 붙여 다시 뜬다")
        if head.get("video") != source:
            raise ValueError(f"{path}: {head.get('video')!r} 의 기록이다. 지금 영상은 {source!r}")
        if head.get("detect_every") != detect_every:
            raise ValueError(f"{path}: 탐지 주기 {head.get('detect_every')} 로 남긴 기록이다. "
                             f"--detect-every {head.get('detect_every')} 로 돌린다 (지금 {detect_every})")
        rec_conf = head.get("conf")
        if conf is not None and rec_conf is not None and _norm_conf(rec_conf) != _norm_conf(conf):
            raise ValueError(f"{path}: 탐지 기준 {rec_conf} 로 남긴 기록이다. 지금 기준은 {conf} — "
                             f"onnx 로 --record 를 붙여 지금 기준으로 다시 뜬다")
        self.conf = rec_conf
        self.model = str(head.get("model", "?"))
        self.name = f"replay:{self.model}"
        self.by_frame: dict[str, list[Detection]] = {}
        times: list[tuple[float, str]] = []
        for n, ln in enumerate(lines[1:], start=2):
            try:
                rec = json.loads(ln)
                dets = [Detection(str(k), float(c), (int(x1), int(y1), int(x2), int(y2)))
                        for k, c, x1, y1, x2, y2 in rec["dets"]]
                self.by_frame[rec["frame"]] = dets
                times.append((float(rec["t"]), rec["frame"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise ValueError(f"{path}:{n}: 기록 줄을 읽지 못했다 ({e})") from None
        times.sort()
        self._t = [t for t, _ in times]
        self._names = [f for _, f in times]
        gaps = [b - a for a, b in zip(self._t, self._t[1:]) if b > a]
        self._tol = (statistics.median(gaps) / 2) if gaps else 0.5

    @property
    def frames(self) -> int:
        return len(self.by_frame)

    def detect(self, image: np.ndarray, frame: str, t: float) -> list[Detection]:
        dets = self.by_frame.get(frame)
        if dets is None and self._t:
            j = bisect.bisect_left(self._t, t)
            near = min((k for k in (j - 1, j) if 0 <= k < len(self._t)), key=lambda k: abs(self._t[k] - t))
            if abs(self._t[near] - t) <= self._tol + 1e-6:
                dets = self.by_frame[self._names[near]]
        return [Detection(d.type, d.confidence, d.bbox) for d in dets or []]


def _norm_conf(conf):
    """숫자 하나 또는 {클래스: 기준} 을 비교할 수 있게 (소수 넷째 자리)."""
    if isinstance(conf, dict):
        return {str(k): round(float(v), 4) for k, v in conf.items()}
    return round(float(conf), 4)


def _conf_from_args(args):
    if getattr(args, "conf", None) is not None:
        return args.conf
    from daboyeo_ai.inference import load_class_conf

    return load_class_conf(args.conf_json)


def make_detector(args) -> FrameDetector:
    """run.py 의 인자로 탐지기를 만든다. 파일이 없거나 기록이 맞지 않으면 FileNotFoundError·ValueError."""
    kind = getattr(args, "detector", "onnx")
    record = getattr(args, "record", None)
    if kind == "replay":
        if not record:
            raise ValueError("--detector replay 는 --record <기록.jsonl> 이 필요하다")
        return ReplayDetector(record, _source_name(args), args.detect_every, conf=_conf_from_args(args))
    if kind == "fake":
        if not args.video:
            raise ValueError("--detector fake 는 --video 와 함께 쓴다 (라벨 시각이 영상 시각이다)")
        det: FrameDetector = FakeDetector(args.labels or default_labels_path(args.video),
                                          include_amb=getattr(args, "include_amb", False))
    elif kind == "onnx":
        if not args.model:
            raise ValueError("--detector onnx 는 --model <파일.onnx> 이 필요하다")
        if not Path(args.model).exists():
            raise FileNotFoundError(f"모델 파일이 없다: {args.model} (Drive daboyeo_data/models/ 에서 받는다)")
        det = OnnxMode(args.model, conf=_conf_from_args(args), threads=args.threads)
    else:
        raise ValueError(f"모르는 탐지기 {kind!r} (쓸 수 있는 값: {', '.join(DETECTORS)})")
    if record:
        det = Recorder(det, record, _source_name(args), args.detect_every)
    return det
