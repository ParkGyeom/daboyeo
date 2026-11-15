"""실제 주행 영상 평가의 공통 부분 — 정답(GT) CSV, 프레임별 탐지 캐시, 이벤트 묶기·맞추기 (계획 4.1·4.3).

`eval_detector`(유형별 P/R/F1) 와 `interval_sweep`(간격별 누락률) 이 같이 쓴다.

정답 CSV (`experiments/gt_events/<video_id>.csv`, 제안 — `docs/ai-handoff.md` 12장):

    defect_id,t_start_s,t_end_s,defect_type,memo
    D001,12.0,14.5,pothole,오른쪽 차로
    D002,30.2,,crack,t_end_s 비우면 t_start_s ± 1초
    G001,41.0,43.0,guardrail_damage,시설물은 '있음' 만 (파손 판정 안 함)

- 헤더 대신 `t_s`(한 시각) 만 써도 된다 → `t_s ± point_half_s`(기본 1초) 창.
- 헤더는 영어/한국어 둘 다 (`유형`·`시각`·`메모`·`ID` …), 유형도 한국어(`포트홀`·`균열`·`차선마모`·`가드레일`·`표지판`)
  나 탐지 클래스 이름(`guardrail`·`sign`) 을 받는다. `#` 로 시작하는 줄은 주석.

평가 유형 (`EVAL_TYPES`): 결함 3종(pothole·crack·lane_wear) 과 시설물 2종(guardrail_damage·sign_damage).
**시설물은 파손 판정을 하지 않으므로 (PRD v0.5) "시설물이 있는 위치를 찾았나" 로만 잰다** — 결함 지표와 섞지 않는다.

맞추기 규칙 (탐지 1건 = 분석한 프레임 1장에서 그 유형 탐지가 있음 = "hit"):
- 정답 이벤트는 창 `[t_start − tol_s, t_end + tol_s]` 안에 같은 유형 hit 가 **하나라도** 있으면 찾음 (재현율·누락률).
- 예측 이벤트 = 같은 유형 hit 를 시간순으로 이어 붙인 것 (간격이 `merge_gap_s` 이하면 같은 이벤트).
  예측 이벤트의 hit 중 하나라도 같은 유형 정답 창 안에 있으면 맞음 (정밀도). 한 결함이 이벤트 두 개로 쪼개져도
  정밀도는 깎지 않고, 쪼개짐은 `분당 이벤트 수` 로 드러난다.
- 차선 마모 hit 는 `LaneWearTracker` 가 `confirmed=True` 를 준 프레임이다 (한 프레임 is_worn 이 아니다).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from daboyeo_contracts import CLASS_NAMES, DETECTOR_TO_DEFECT, DefectType

# 평가 유형 순서 (표·CSV 출력 순서)
DEFECT_TYPES: tuple[str, ...] = (DefectType.POTHOLE.value, DefectType.CRACK.value, DefectType.LANE_WEAR.value)
FACILITY_TYPES: tuple[str, ...] = (DefectType.GUARDRAIL_DAMAGE.value, DefectType.SIGN_DAMAGE.value)
EVAL_TYPES: tuple[str, ...] = DEFECT_TYPES + FACILITY_TYPES
DETECTOR_TYPES: tuple[str, ...] = tuple(DETECTOR_TO_DEFECT[c].value for c in CLASS_NAMES)


def type_group(t: str) -> str:
    return "facility_presence" if t in FACILITY_TYPES else "defect"


_TYPE_ALIASES: dict[str, str] = {
    # 영어 (DefectType 값 · 탐지 클래스 이름)
    "pothole": "pothole", "crack": "crack", "lane_wear": "lane_wear", "lanewear": "lane_wear", "lane": "lane_wear",
    "guardrail_damage": "guardrail_damage", "guardrail": "guardrail_damage",
    "sign_damage": "sign_damage", "sign": "sign_damage",
    # 한국어
    "포트홀": "pothole", "포트 홀": "pothole", "구멍": "pothole",
    "균열": "crack", "크랙": "crack", "도로균열": "crack", "도로 균열": "crack",
    "차선마모": "lane_wear", "차선 마모": "lane_wear", "차선": "lane_wear", "마모": "lane_wear",
    "가드레일": "guardrail_damage", "가드레일파손": "guardrail_damage", "가드레일 파손": "guardrail_damage",
    "표지판": "sign_damage", "표지판파손": "sign_damage", "표지판 파손": "sign_damage",
}

_COL_ALIASES: dict[str, tuple[str, ...]] = {
    "defect_id": ("defect_id", "id", "결함id", "결함_id", "아이디"),
    "t_start_s": ("t_start_s", "t_start", "start_s", "start", "시작", "시작_s", "시작(초)"),
    "t_end_s": ("t_end_s", "t_end", "end_s", "end", "끝", "끝_s", "끝(초)"),
    "t_s": ("t_s", "t", "time_s", "time", "시각", "시각_s", "시각(초)", "초"),
    "defect_type": ("defect_type", "type", "class", "유형", "결함유형", "결함_유형", "종류"),
    "memo": ("memo", "note", "notes", "메모", "비고"),
}


def normalize_type(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    if key in _TYPE_ALIASES:
        return _TYPE_ALIASES[key]
    key2 = key.replace(" ", "").replace("_", "")
    for k, v in _TYPE_ALIASES.items():
        if k.replace(" ", "").replace("_", "") == key2:
            return v
    raise ValueError(f"모르는 결함 유형 {raw!r} — 쓸 수 있는 값: {', '.join(EVAL_TYPES)} (또는 한국어 이름)")


@dataclass(frozen=True)
class GtEvent:
    video_id: str
    defect_id: str
    defect_type: str  # EVAL_TYPES 중 하나
    t_start: float
    t_end: float
    memo: str = ""


def _parse_time(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    if ":" in s:  # mm:ss(.s) 또는 hh:mm:ss
        out = 0.0
        for part in s.split(":"):
            out = out * 60 + float(part)
        return out
    return float(s)


def load_gt(path: str | Path, video_id: str | None = None, point_half_s: float = 1.0) -> list[GtEvent]:
    """정답 CSV 를 읽는다. 시각은 초(12.5) 또는 `mm:ss` 둘 다 받는다."""
    path = Path(path)
    vid = video_id or path.stem
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    reader = csv.reader(lines)
    header = [h.strip().lower() for h in next(reader)]
    col: dict[str, int] = {}
    for name, aliases in _COL_ALIASES.items():
        for i, h in enumerate(header):
            if h in aliases:
                col[name] = i
                break
    if "defect_type" not in col or not ({"t_start_s", "t_s"} & col.keys()):
        raise ValueError(f"{path}: 헤더에 유형(defect_type) 과 시각(t_start_s 또는 t_s) 열이 있어야 한다 — 받은 헤더 {header}")

    def get(row: list[str], name: str) -> str:
        i = col.get(name)
        return row[i].strip() if i is not None and i < len(row) else ""

    out: list[GtEvent] = []
    for n, row in enumerate(reader, start=2):
        if not any(c.strip() for c in row):
            continue
        try:
            dtype = normalize_type(get(row, "defect_type"))
            t0 = _parse_time(get(row, "t_start_s"))
            t1 = _parse_time(get(row, "t_end_s"))
            tp = _parse_time(get(row, "t_s"))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
        if t0 is None and tp is None:
            raise ValueError(f"{path}:{n}: 시각이 비어 있다")
        if t0 is None:  # t_s 만
            t0, t1 = tp - point_half_s, tp + point_half_s
        elif t1 is None:
            t0, t1 = t0 - point_half_s, t0 + point_half_s
        if t1 < t0:
            raise ValueError(f"{path}:{n}: t_end_s({t1}) < t_start_s({t0})")
        did = get(row, "defect_id") or f"{vid}#{n}"
        out.append(GtEvent(vid, did, dtype, max(0.0, t0), t1, get(row, "memo")))
    return out


# ---- 탐지기 쪽 -----------------------------------------------------------------------


class DetectorLike(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list: ...  # list[Detection] (class_name, conf 만 쓴다)


@dataclass
class FrameRecord:
    """분석한 프레임 1장의 결과. 탐지는 가장 낮은 conf 기준으로 저장하고 나중에 거른다."""

    frame: int
    t: float
    dets: list[tuple[str, float]] = field(default_factory=list)  # (DefectType 값, conf)
    # LaneWearResult 중 tracker 가 쓰는 값 (lane_found, contrast_p25, paint_len). 차선을 안 쟀으면 None
    lane: tuple[bool, float | None, float] | None = None


def iter_video_frames(video: str | Path, every: int = 1, fps: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """(프레임 번호, 시각 초, BGR 프레임) — `every` 프레임마다. fps 를 모르면 30 으로 둔다."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없다: {video}")
    f = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not math.isfinite(f) or f <= 0:
        f = 30.0
    i = 0
    try:
        while True:
            if i % every:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, i / f, frame
            i += 1
    finally:
        cap.release()


def video_info(video: str | Path, fps: float | None = None) -> tuple[float, int]:
    """(fps, 프레임 수). 프레임 수를 못 읽으면 0."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    f = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return (f if math.isfinite(f) and f > 0 else 30.0), n


def scan_frames(
    frames: Iterable[tuple[int, float, np.ndarray]],
    detector: DetectorLike | None,
    lane_meter=None,
    progress: Callable[[int], None] | None = None,
) -> list[FrameRecord]:
    """프레임마다 탐지기·차선 측정을 돌려 기록한다 (탐지기 conf 는 호출 쪽이 가장 낮은 값으로 둔다)."""
    out = []
    for k, (i, t, frame) in enumerate(frames):
        dets = []
        if detector is not None:
            dets = [(DETECTOR_TO_DEFECT[d.class_name].value, float(d.conf)) for d in detector.detect(frame)]
        lane = None
        if lane_meter is not None:
            r = lane_meter.measure(frame)
            lane = (bool(r.lane_found), r.contrast_p25, float(r.paint_len))
        out.append(FrameRecord(i, t, dets, lane))
        if progress is not None:
            progress(k)
    return out


def save_cache(path: str | Path, records: Sequence[FrameRecord], meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for r in records:
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def load_cache(path: str | Path) -> tuple[dict, list[FrameRecord]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])["meta"]
    recs = []
    for ln in lines[1:]:
        d = json.loads(ln)
        recs.append(FrameRecord(d["frame"], d["t"], [tuple(x) for x in d["dets"]],
                                tuple(d["lane"]) if d["lane"] is not None else None))
    return meta, recs


def subsample(records: Sequence[FrameRecord], every: int) -> list[FrameRecord]:
    """프레임 번호가 every 의 배수인 기록만 (캐시를 every=1 로 만들었으면 어떤 간격이든 된다)."""
    return [r for r in records if r.frame % every == 0]


# ---- conf 기준 ---------------------------------------------------------------------


def load_conf_json(path: str | Path) -> dict[str, float]:
    """클래스별 conf JSON → {DefectType 값: conf}. 여러 모양을 받는다:
    `{"pothole": 0.3, ...}`, `{"class_conf": {...}}`(conf_thresholds.json), `{"per_class_conf": {...}}`, `{"thresholds": {...}}`,
    값이 `{"conf": 0.3}` 같은 객체여도 된다. 키는 탐지 클래스 이름(pothole·crack·guardrail·sign) 또는 DefectType 값."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    def as_map(d) -> dict[str, float] | None:
        if not isinstance(d, dict):
            return None
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                v = v.get("conf", v.get("threshold", v.get("value")))
            try:
                t = normalize_type(str(k))
            except ValueError:
                continue
            if isinstance(v, int | float) and t in DETECTOR_TYPES:
                out[t] = float(v)
        return out or None

    for key in ("class_conf", "per_class_conf", "conf_per_class", "thresholds", "per_class", "conf", "classes"):
        m = as_map(raw.get(key)) if isinstance(raw, dict) else None
        if m:
            return m
    m = as_map(raw)
    if not m:
        raise ValueError(f"{path}: 클래스별 conf 를 찾지 못했다 (예: {{\"pothole\": 0.3, \"crack\": 0.25}})")
    return m


def conf_for(conf: float | dict[str, float], dtype: str, default: float = 0.25) -> float:
    return conf.get(dtype, default) if isinstance(conf, dict) else conf


# ---- 이벤트 -----------------------------------------------------------------------


@dataclass(frozen=True)
class PredEvent:
    video_id: str
    defect_type: str
    t_start: float
    t_end: float
    hits: tuple[float, ...]  # 이 이벤트에 든 hit 시각
    max_conf: float


def detector_hits(records: Sequence[FrameRecord], conf: float | dict[str, float],
                  default_conf: float = 0.25) -> dict[str, list[tuple[float, float]]]:
    """유형별 [(시각, 그 프레임 최고 conf)] — conf 기준을 넘은 탐지가 있는 프레임만."""
    out: dict[str, list[tuple[float, float]]] = {t: [] for t in DETECTOR_TYPES}
    for r in records:
        best: dict[str, float] = {}
        for dtype, c in r.dets:
            if c >= conf_for(conf, dtype, default_conf) and c > best.get(dtype, -1.0):
                best[dtype] = c
        for dtype, c in best.items():
            out[dtype].append((r.t, c))
    return out


def lane_wear_hits(records: Sequence[FrameRecord], tracker_factory: Callable | None = None) -> list[tuple[float, float]]:
    """새 `LaneWearTracker` 로 분석한 프레임을 차례로 넣어 `confirmed` 프레임의 (시각, confidence)."""
    from daboyeo_ai.lane_wear import LaneWearResult, LaneWearTracker

    tracker = (tracker_factory or LaneWearTracker)()
    out = []
    for r in records:
        if r.lane is None:
            continue
        found, p25, paint = r.lane
        res = LaneWearResult(lane_found=found, contrast=None, is_worn=False, confidence=0.0, lane_pixel_ratio=0.0,
                             n_segments=0, contrast_p25=p25, paint_len=paint)
        j = tracker.update(res, t=r.t)
        if j.confirmed:
            out.append((r.t, float(j.confidence)))
    return out


def group_events(video_id: str, dtype: str, hits: Sequence[tuple[float, float]], merge_gap_s: float) -> list[PredEvent]:
    """시간순 hit 를 간격 merge_gap_s 이하끼리 이어 이벤트로 묶는다."""
    events: list[PredEvent] = []
    cur: list[tuple[float, float]] = []
    for t, c in sorted(hits):
        if cur and t - cur[-1][0] > merge_gap_s:
            events.append(_mk(video_id, dtype, cur))
            cur = []
        cur.append((t, c))
    if cur:
        events.append(_mk(video_id, dtype, cur))
    return events


def _mk(video_id: str, dtype: str, cur: list[tuple[float, float]]) -> PredEvent:
    return PredEvent(video_id, dtype, cur[0][0], cur[-1][0], tuple(t for t, _ in cur), max(c for _, c in cur))


def in_window(t: float, g: GtEvent, tol_s: float) -> bool:
    return g.t_start - tol_s <= t <= g.t_end + tol_s


@dataclass
class TypeCounts:
    n_gt: int = 0
    n_gt_found: int = 0
    n_pred: int = 0
    n_pred_ok: int = 0
    minutes: float = 0.0
    missed: list[str] = field(default_factory=list)  # 놓친 정답 defect_id

    def add(self, o: TypeCounts) -> None:
        self.n_gt += o.n_gt
        self.n_gt_found += o.n_gt_found
        self.n_pred += o.n_pred
        self.n_pred_ok += o.n_pred_ok
        self.minutes += o.minutes
        self.missed.extend(o.missed)

    @property
    def recall(self) -> float | None:
        return self.n_gt_found / self.n_gt if self.n_gt else None

    @property
    def miss_rate(self) -> float | None:
        return 1 - self.n_gt_found / self.n_gt if self.n_gt else None

    @property
    def precision(self) -> float | None:
        return self.n_pred_ok / self.n_pred if self.n_pred else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def per_min(self) -> float | None:
        return self.n_pred / self.minutes if self.minutes else None


def match(gt: Sequence[GtEvent], preds: Sequence[PredEvent], dtype: str, tol_s: float, minutes: float) -> TypeCounts:
    g = [e for e in gt if e.defect_type == dtype]
    p = [e for e in preds if e.defect_type == dtype]
    hits = [t for e in p for t in e.hits]
    found = [e for e in g if any(in_window(t, e, tol_s) for t in hits)]
    ok = [e for e in p if any(in_window(t, x, tol_s) for t in e.hits for x in g)]
    return TypeCounts(len(g), len(found), len(p), len(ok), minutes,
                      [e.defect_id for e in g if e not in found])


def evaluate_video(
    video_id: str,
    records: Sequence[FrameRecord],
    gt: Sequence[GtEvent],
    conf: float | dict[str, float],
    duration_s: float,
    tol_s: float = 0.5,
    merge_gap_s: float = 1.0,
    lane_hits: Sequence[tuple[float, float]] | None = None,
    default_conf: float = 0.25,
) -> dict[str, TypeCounts]:
    """한 영상의 유형별 집계. lane_hits 를 안 주면 차선 마모는 빠진다."""
    hits = detector_hits(records, conf, default_conf)
    if lane_hits is not None:
        hits[DefectType.LANE_WEAR.value] = list(lane_hits)
    minutes = duration_s / 60.0
    out = {}
    for dtype in EVAL_TYPES:
        if dtype not in hits:
            continue
        preds = group_events(video_id, dtype, hits[dtype], merge_gap_s)
        out[dtype] = match(gt, preds, dtype, tol_s, minutes)
    return out


def fmt(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:.{nd}f}"


# ---- 영상 → 기록 (캐시 포함) ----------------------------------------------------------


@dataclass
class VideoRecords:
    video_id: str
    records: list[FrameRecord]
    fps: float
    duration_s: float
    every: int  # 기록이 몇 프레임마다인지
    meta: dict


def scan_or_load(
    video: str | Path,
    every: int,
    conf_floor: float,
    detector: DetectorLike | Callable[[float], DetectorLike] | None,
    lane: bool = True,
    cache_dir: str | Path | None = None,
    fps: float | None = None,
    model_name: str = "",
    log: Callable[[str], None] = print,
) -> VideoRecords:
    """캐시(`<cache_dir>/<video_id>.jsonl`) 가 맞으면 읽고, 아니면 영상을 돌려 만든다.

    캐시가 맞는 조건: 같은 모델 이름, 캐시 간격이 every 의 약수, 캐시 conf_floor ≤ 요청, 차선(lane) 을 쟀음(필요하면).
    detector 는 탐지기 객체 또는 `conf -> 탐지기` 함수 (함수면 캐시가 맞을 때 모델을 열지 않는다).
    """
    video = Path(video)
    vid = video.stem
    cache = Path(cache_dir) / f"{vid}.jsonl" if cache_dir else None
    if cache is not None and cache.exists():
        meta, recs = load_cache(cache)
        if (meta.get("model") == model_name and every % int(meta["every"]) == 0
                and float(meta["conf_floor"]) <= conf_floor + 1e-9 and (meta.get("lane") or not lane)):
            log(f"[{vid}] 캐시 사용 {cache} (every={meta['every']}, 프레임 {len(recs)})")
            return VideoRecords(vid, subsample(recs, every), float(meta["fps"]), float(meta["duration_s"]), every, meta)
        log(f"[{vid}] 캐시가 조건과 달라 다시 돈다 ({cache})")
    vfps, n_total = video_info(video, fps)
    det = detector(conf_floor) if callable(detector) and not hasattr(detector, "detect") else detector
    meter = None
    if lane:
        from daboyeo_ai.lane_wear import LaneWearMeter

        meter = LaneWearMeter()
    n_expect = (n_total + every - 1) // every if n_total else 0

    def progress(k: int) -> None:
        if k and k % 200 == 0:
            log(f"[{vid}] {k}/{n_expect or '?'} 프레임")

    recs = scan_frames(iter_video_frames(video, every, vfps), det, meter, progress)
    n_frames = n_total or ((recs[-1].frame + every) if recs else 0)
    duration = n_frames / vfps
    meta = {"video": str(video), "model": model_name, "every": every, "conf_floor": conf_floor, "lane": lane,
            "fps": vfps, "n_frames": n_frames, "duration_s": duration}
    if cache is not None:
        save_cache(cache, recs, meta)
        log(f"[{vid}] 캐시 저장 {cache}")
    return VideoRecords(vid, recs, vfps, duration, every, meta)


def find_gt(video: str | Path, gt_dir: str | Path) -> Path | None:
    p = Path(gt_dir) / f"{Path(video).stem}.csv"
    return p if p.exists() else None
