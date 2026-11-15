"""차선 마모 판정을 **실측 재귀반사도(LTL-M)** 와 비교한다 (오픈 이슈 #6).

AI Hub 「도로 로드마크 인식을 위한 주행 영상 데이터」(dataSetSn 654) 검증 묶음 VS4.비온전(복합) + VL4 를 쓴다.
프레임마다 차량에 단 이동식 재귀반사 측정기(LTL-M) 값 하나(단위 mcd, 문서상 노면표시 반사도)와 차선 polyline
라벨이 있다. 프레임은 약 1초 간격의 짧은 토막이라 `LaneWearTracker` 의 warmup·5 중 3 투표는 맞지 않는다 —
**한 프레임 판정**만 잰다.

    uv run python -m daboyeo_ai.eval.lane_wear_roadmark \\
        --ltlm data/roadmark/vs4/LTL-M --labels data/roadmark/vl4/2D_JSON --gps data/roadmark/vs4/GPS --all-images \\
        --images ".../VS4.비온전(복합).zip" --out experiments/lane_wear_roadmark.json

`--images` 는 `<name>.png` 가 든 폴더 또는 `IMAGE/<name>.png` 가 든 zip (압축을 풀지 않고 읽는다).

정답 만들기 (근거: docs/ai-handoff.md §8):
- LTL-M < `--no-paint-below`(30): 측정기 아래 페인트가 없음(점선 빈칸·차선 밖) 또는 측정 안 됨으로 보고 **뺀다**.
  값이 0~20 과 50~190 두 덩어리로 갈리고 20~30 은 비어 있어, 닳은 페인트가 연속으로 낮아진 값으로 보기 어렵다.
- 30 이상에서 `--worn-below`(100) 미만이면 "마모" (재도색 기준: 백색 70, 황색 100 mcd/(m²·lx)).
- 주행(drive) = 파일 이름의 `<도로>_<시간>_<날씨>_<복잡도>_<고유번호 앞 4자리>`. 상대 점수의 기준(중앙값)은 그 안의
  차선 찾은 프레임 전체(LTL-M 없음·30 미만 포함)로 만든다 — 라벨을 쓰지 않는다.

출력은 **집계 숫자만** 저장한다 (AI Hub 데이터는 재배포 금지 — 프레임별 값·이미지를 레포에 두지 않는다).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.eval.lane_wear_eval import auc_low_is_worn, best_f1, f1_at, make_folds, threshold_rows, two_fold_f1
from daboyeo_ai.lane_wear import PROVISIONAL_WEAR_SCORE, LaneWearMeter, LaneWearResult, LaneWearTracker

# VL4 라벨의 선 종류 (category_id). 381 중앙선, 382 유턴구역선, 383 차선, 384 버스전용차로, 385 길가장자리구역선,
# 386~388 진로변경제한선. 이 중 자기 차로 양옆 선이 될 수 있는 것.
LINE_CATS = frozenset(range(381, 389))
EGO_CATS = frozenset({381, 383, 384, 385})
NO_PAINT_BELOW = 30.0
WORN_BELOW = 100.0
SUBSETS = ("light", "light_clear", "night")


# ---- 입력 ------------------------------------------------------------------------------


def parse_name(name: str) -> dict[str, str]:
    """`0401_light_clear_smooth_01000085` → road/time/weather/flow/idx."""
    p = name.split("_")
    if len(p) != 5:
        raise ValueError(f"모르는 파일 이름 {name!r} (<도로>_<시간>_<날씨>_<복잡도>_<번호>)")
    return dict(zip(("road", "time", "weather", "flow", "idx"), p))


def drive_of(name: str) -> str:
    p = parse_name(name)
    return f"{p['road']}_{p['time']}_{p['weather']}_{p['flow']}_{p['idx'][:4]}"


def load_ltlm(folder: Path) -> dict[str, float]:
    """`<name>.csv` 마다 숫자 하나. 비었거나 숫자가 아니면 뺀다."""
    out = {}
    for f in sorted(folder.glob("*.csv")):
        txt = f.read_text(encoding="utf-8-sig").strip().split(",")[0].strip()
        try:
            out[f.stem] = float(txt)
        except ValueError:
            continue
    return out


def load_lines(path: Path) -> list[tuple[int, np.ndarray]]:
    """VL4 JSON 의 선 polyline [(category_id, N×2 원본 픽셀)]. `images` 가 리스트든 객체든 상관없다."""
    d = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for a in d.get("annotations", []):
        if a.get("category_id") in LINE_CATS and a.get("polyline"):
            pts = np.asarray(a["polyline"], float).reshape(-1, 2)
            if len(pts) >= 2:
                out.append((int(a["category_id"]), pts))
    return out


def load_gps(path: Path) -> tuple[float, float, float] | None:
    """GPS CSV 한 줄 → (UTC 초, 위도, 경도). 날짜는 없고 시각은 hhmmss.ss, 속도(쓰지 않음)는 knot.

    형식이 셋 섞여 있다: A `시각,위도,경도,속도,방위,…`(8칸), B `번호,시각,위도,경도,…`(9칸), C 는 A 뒤에 빈 칸.
    B 는 2·3번째가 아니라 3·4번째 칸이 한국 좌표 범위라서 구별한다. 좌표가 한국 범위가 아니면(다른 신호가 든
    잘못된 행) None.
    """
    if not path.exists():
        return None
    p = path.read_text(encoding="utf-8-sig").strip().split(",")

    def num(i: int) -> float | None:
        try:
            return float(p[i])
        except (IndexError, ValueError):
            return None

    for ti in (0, 1):  # A·C 는 시각이 0번째, B 는 1번째
        hms, lat, lon = num(ti), num(ti + 1), num(ti + 2)
        if hms is not None and lat is not None and lon is not None and _in_korea(lat, lon):
            t = int(hms // 10000) * 3600 + int(hms // 100 % 100) * 60 + hms % 100
            return t, lat, lon
    return None


def _in_korea(lat: float, lon: float) -> bool:
    return 33.0 <= lat <= 39.0 and 124.0 <= lon <= 132.0


def road_brightness(img: np.ndarray) -> float:
    """자기 차로 노면 밝기: 화면 아래 가운데(가로 0.3~0.7, 세로 0.75~0.9) V 중앙값. 차선은 거의 안 들어간다."""
    h, w = img.shape[:2]
    return float(np.median(img[int(0.75 * h):int(0.9 * h), int(0.3 * w):int(0.7 * w)].max(2)))


class ImageSource:
    """폴더(`<name>.png`) 또는 zip(`IMAGE/<name>.png`, 하위 폴더 이름은 상관없음)에서 프레임을 읽는다."""

    def __init__(self, path: Path):
        self.path = path
        self._zip = zipfile.ZipFile(path) if path.is_file() else None
        if self._zip is not None:
            self._names = {Path(n).stem: n for n in self._zip.namelist() if n.lower().endswith(".png")}
        else:
            self._names = {p.stem: str(p) for p in sorted(path.glob("*.png"))}

    def names(self) -> list[str]:
        return sorted(self._names)

    def read(self, name: str) -> np.ndarray | None:
        key = self._names.get(name)
        if key is None:
            return None
        buf = self._zip.read(key) if self._zip is not None else Path(key).read_bytes()
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


# ---- 차선 위치 확인 (VL4 polyline) ----------------------------------------------------------


def _point_to_polyline(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """점들(M×2) 에서 polyline(N×2) 까지의 최단 거리."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = ((pts[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None]
    proj = a[None] + np.clip(t, 0, 1)[..., None] * ab[None]
    return np.sqrt(((pts[:, None, :] - proj) ** 2).sum(-1)).min(1)


def segments_on_lines(segs: Iterable[tuple[int, int, int, int]], lines: list[tuple[int, np.ndarray]],
                      tol_px: float, step_px: float = 5.0) -> tuple[float, float]:
    """(선분 위 표본점 수, 그중 라벨 선에서 tol_px 안인 수). 길이 가중 정밀도 = 둘의 비."""
    tot = hit = 0.0
    for xa, ya, xb, yb in segs:
        n = max(2, int(np.hypot(xb - xa, yb - ya) / step_px) + 1)
        pts = np.stack([np.linspace(xa, xb, n), np.linspace(ya, yb, n)], 1)
        tot += n
        if lines:
            d = np.min([_point_to_polyline(pts, p) for _, p in lines], 0)
            hit += float(np.sum(d <= tol_px))
    return tot, hit


def ego_lines(lines: list[tuple[int, np.ndarray]], w: int, h: int, y_ref: float = 0.83,
              y_min: float = 0.65) -> dict[str, tuple[int, np.ndarray] | None]:
    """자기 차로 양옆 선: 아래쪽(높이 y_min 아래)까지 내려온 EGO_CATS 선을 y_ref 높이로 늘였을 때 화면 가운데에서
    가장 가까운 왼쪽·오른쪽 선."""
    best: dict[str, tuple[float, tuple[int, np.ndarray]] | None] = {"left": None, "right": None}
    for cat, p in lines:
        if cat not in EGO_CATS:
            continue
        q = p[np.argsort(-p[:, 1])]
        if q[0, 1] < y_min * h or q[0, 1] == q[1, 1]:
            continue
        (x1, y1), (x2, y2) = q[0], q[1]
        x = x1 + (y_ref * h - y1) * (x2 - x1) / (y2 - y1)
        side = "left" if x < w / 2 else "right"
        dist = abs(x - w / 2)
        if best[side] is None or dist < best[side][0]:
            best[side] = (dist, (cat, p))
    return {k: (v[1] if v else None) for k, v in best.items()}


def ego_offsets(lines: list[tuple[int, np.ndarray]], w: int, h: int, y_ref: float = 0.83) -> dict[str, float | None]:
    """자기 차로 왼쪽·오른쪽 선이 높이 y_ref 에서 화면 가운데로부터 떨어진 거리 (화면 폭 비율). 선이 없으면 None.
    측정기가 한쪽 선 위에 달려 있다면, 차가 그 선에서 멀어질수록 LTL-M 이 '페인트 없음'으로 떨어져야 한다."""
    out: dict[str, float | None] = {}
    for side, e in ego_lines(lines, w, h, y_ref).items():
        if e is None:
            out[side] = None
            continue
        q = e[1][np.argsort(-e[1][:, 1])]
        (x1, y1), (x2, y2) = q[0], q[1]
        out[side] = float(abs(x1 + (y_ref * h - y1) * (x2 - x1) / (y2 - y1) - w / 2) / w)
    return out


def label_contrast(img: np.ndarray, poly: np.ndarray, y_lo: float = 0.6, y_hi: float = 0.92,
                   horizon: float = 0.2, max_stripe: float = 0.035) -> float | None:
    """라벨 선 위치에서 잰 (페인트 V − 옆 노면 V) / 노면 V 의 중앙값. 우리 차선 찾기와 무관한 **기준(oracle) 대비**.

    행마다 선 x 의 ±d/2 안 V 의 최댓값(라벨이 몇 화소 어긋나도 페인트를 잡도록)을 페인트, x±(1.5~2.5)d 의 V 중앙값을
    노면으로 본다. d 는 `LaneWearMeter` 와 같은 원근 차선 폭 상한.
    """
    h, w = img.shape[:2]
    v = img.max(2).astype(np.float32)
    vals = []
    for y in range(int(y_lo * h), int(y_hi * h), 4):
        xs = []
        for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
            if y1 != y2 and min(y1, y2) <= y <= max(y1, y2):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        if not xs:
            continue
        x = int(round(xs[0]))
        d = max(3, int(max_stripe * w * (y - horizon * h) / ((1 - horizon) * h)))
        lo, hi = x - 5 * d // 2, x + 5 * d // 2
        if lo < 0 or hi >= w:
            continue
        row = v[y]
        paint = float(row[x - d // 2: x + d // 2 + 1].max())
        road = float(np.median(np.concatenate([row[lo: x - 3 * d // 2], row[x + 3 * d // 2 + 1: hi + 1]])))
        vals.append((paint - road) / max(road, 1.0))
    return float(np.median(vals)) if len(vals) >= 5 else None


# ---- 측정 ------------------------------------------------------------------------------


@dataclass
class Frame:
    name: str
    drive: str
    time: str
    weather: str
    ltl: float | None
    result: LaneWearResult
    has_label: bool = False
    ego_left: bool = False  # 자기 차로 왼쪽 라벨 선이 있나
    ego_right: bool = False
    seg_pts: float = 0.0  # 채택 선분 표본점 수
    seg_hit: float = 0.0  # 그중 라벨 선 위
    oracle: dict[str, float | None] = field(default_factory=dict)  # left/right 기준 대비
    road_v: float | None = None  # `road_brightness`
    gps: tuple[float, float, float] | None = None  # (UTC 초, 위도, 경도)
    ego_off: dict[str, float | None] = field(default_factory=dict)  # `ego_offsets`


def measure_frames(names: Iterable[str], src: ImageSource, meter: LaneWearMeter, ltl: dict[str, float],
                   labels: Path | None, tol_frac: float = 0.015, gps: Path | None = None) -> Iterator[Frame]:
    for name in names:
        img = src.read(name)
        if img is None:
            print(f"건너뜀 (읽기 실패): {name}", file=sys.stderr)
            continue
        p = parse_name(name)
        r, segs = meter.measure_debug(img)
        f = Frame(name, drive_of(name), p["time"], p["weather"], ltl.get(name), r, road_v=road_brightness(img),
                  gps=load_gps(gps / f"{name}.csv") if gps is not None else None)
        lp = labels / f"{name}.json" if labels is not None else None
        if lp is not None and lp.exists():
            h, w = img.shape[:2]
            lines = load_lines(lp)
            ego = ego_lines(lines, w, h)
            f.has_label, f.ego_left, f.ego_right = True, ego["left"] is not None, ego["right"] is not None
            f.ego_off = ego_offsets(lines, w, h)
            if r.lane_found:
                f.seg_pts, f.seg_hit = segments_on_lines(segs, lines, tol_frac * w)
            f.oracle = {s: (label_contrast(img, e[1]) if e is not None else None) for s, e in ego.items()}
        yield f


def refresh_meta(frames: list[Frame], labels: Path | None, gps: Path | None, size: tuple[int, int] = (1920, 1080)) -> None:
    """캐시에서 읽은 프레임의 GPS·라벨 선 거리를 다시 읽는다 (이미지 없이). 파서를 고친 뒤 다시 잴 필요가 없게."""
    for f in frames:
        if gps is not None:
            f.gps = load_gps(gps / f"{f.name}.csv")
        lp = labels / f"{f.name}.json" if labels is not None else None
        if lp is not None and lp.exists():
            f.ego_off = ego_offsets(load_lines(lp), *size)


# ---- 지표 ------------------------------------------------------------------------------


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        r[order] = np.arange(len(x))
        for v in np.unique(x[np.r_[np.diff(np.sort(x)) == 0, False]]):  # 같은 값은 평균 순위
            m = x == v
            r[m] = r[m].mean()
        return r
    if len(a) < 3:
        return float("nan")
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def scores(frames: list[Frame]) -> dict[str, np.ndarray]:
    """점수 이름 → 프레임별 점수 (낮을수록 마모, 차선 못 찾은 프레임은 NaN)."""
    found = [f for f in frames if f.result.lane_found]
    med: dict[str, dict[str, float]] = {}
    for fld in ("contrast", "contrast_p25"):
        med[fld] = {}
        for d in {f.drive for f in found}:
            med[fld][d] = float(np.median([getattr(f.result, fld) for f in found if f.drive == d]))
    tmed = {t: float(np.median([f.result.contrast_p25 for f in found if f.time == t])) for t in {f.time for f in found}}
    tr = LaneWearTracker()

    def col(fn) -> np.ndarray:
        return np.array([fn(f) if f.result.lane_found else np.nan for f in frames], float)

    return {
        "contrast": col(lambda f: f.result.contrast),
        "contrast_p25": col(lambda f: f.result.contrast_p25),
        "paint_len": col(lambda f: f.result.paint_len),
        "contrast ÷ 주행 중앙값": col(lambda f: f.result.contrast / med["contrast"][f.drive]),
        "★ score (p25 ÷ 주행 중앙값 × paint_len)": col(lambda f: tr.score(f.result, med["contrast_p25"][f.drive])),
        "score (p25 ÷ 낮/밤 전체 중앙값 × paint_len)": col(lambda f: tr.score(f.result, tmed[f.time])),
    }


def score_metrics(s: np.ndarray, ltl: np.ndarray, worn: np.ndarray, drives: list[str],
                  provisional: float | None) -> dict:
    ok = ~np.isnan(s)
    s, ltl, worn = s[ok], ltl[ok], worn[ok]
    drives = [d for d, o in zip(drives, ok) if o]
    f, t = best_f1(s, worn)
    out = {"n": int(len(s)), "n_worn": int(worn.sum()), "auc": auc_low_is_worn(s, worn), "best_f1": f,
           "best_threshold": t, "f1_2fold": two_fold_f1(s, worn, make_folds(drives, worn)) if len(s) else float("nan"),
           "spearman_ltl": spearman(s, ltl)}
    if provisional is not None:
        out["f1_at_provisional"] = f1_at(s, worn, provisional)
    return out


def subset_mask(frames: list[Frame], name: str) -> np.ndarray:
    if name == "light_clear":
        return np.array([f.time == "light" and f.weather == "clear" for f in frames])
    return np.array([f.time == name for f in frames])


def evaluate(frames: list[Frame], no_paint_below: float = NO_PAINT_BELOW, worn_below: float = WORN_BELOW,
             thresholds=(0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0)) -> dict:
    sc = scores(frames)  # 기준 중앙값은 낮/밤을 나누기 전에 전체 프레임으로 만든다
    ltl_all = np.array([np.nan if f.ltl is None else f.ltl for f in frames], float)
    found_all = np.array([f.result.lane_found for f in frames])
    out: dict = {"settings": {"no_paint_below": no_paint_below, "worn_below": worn_below,
                              "provisional_wear_score": PROVISIONAL_WEAR_SCORE},
                 "n_frames": len(frames), "n_with_ltl": int(np.sum(~np.isnan(ltl_all)))}
    star = "★ score (p25 ÷ 주행 중앙값 × paint_len)"
    for sub in SUBSETS:
        m = subset_mask(frames, sub)
        has = m & ~np.isnan(ltl_all)
        paint = has & (ltl_all >= no_paint_below)
        nopaint = has & (ltl_all < no_paint_below)
        worn_m = paint & (ltl_all < worn_below)
        res: dict = {
            "n": int(m.sum()), "n_ltl": int(has.sum()), "n_no_paint": int(nopaint.sum()),
            "n_paint": int(paint.sum()), "n_worn": int(worn_m.sum()),
            "ltl_percentiles_paint": dict(zip(("p10", "p25", "p50", "p75", "p90"),
                                              np.percentile(ltl_all[paint], [10, 25, 50, 75, 90]).round(1).tolist()))
            if paint.any() else {},
            "lane_found_rate": {"all": _rate(found_all[m]), "no_paint": _rate(found_all[nopaint]),
                                "worn": _rate(found_all[worn_m]), "normal": _rate(found_all[paint & ~worn_m])},
        }
        idx = np.flatnonzero(paint)
        worn = worn_m[idx]
        drives = [frames[i].drive for i in idx]
        res["scores"] = {k: score_metrics(v[idx], ltl_all[idx], worn, drives,
                                          PROVISIONAL_WEAR_SCORE if k == star else None) for k, v in sc.items()}
        # 30 미만도 마모로 친 경우 (참고: 측정기 아래 페인트 없음을 마모로 볼 때)
        idx2 = np.flatnonzero(has)
        w2 = ltl_all[idx2] < worn_below
        res["star_if_no_paint_is_worn"] = score_metrics(sc[star][idx2], ltl_all[idx2], w2,
                                                        [frames[i].drive for i in idx2], PROVISIONAL_WEAR_SCORE)
        # 정현우 11/12 기준: LTL-M < 30 = 마모 (온전 표본 100~160, 손상 0~16 사이에서 고른 값 — 매뉴얼·법 기준 아님)
        low = ltl_all[idx2] < no_paint_below
        d2 = [frames[i].drive for i in idx2]
        res["worn_is_below_cutoff"] = {k: score_metrics(sc[k][idx2], ltl_all[idx2], low, d2,
                                                        PROVISIONAL_WEAR_SCORE if k == star else None)
                                       for k in (star, "contrast", "paint_len")}
        res["accuracy"] = {"below_cutoff": _accuracy(sc[star][idx2], low, PROVISIONAL_WEAR_SCORE),
                           "below_worn": _accuracy(sc[star][idx], worn, PROVISIONAL_WEAR_SCORE)}
        s = sc[star][idx]
        ok = ~np.isnan(s)
        res["star_threshold_rows"] = threshold_rows(s[ok], worn[ok], thresholds)
        s_np = sc[star][nopaint]
        res["star_flag_rate_no_paint"] = {str(t): _rate(s_np[~np.isnan(s_np)] < t) for t in thresholds}
        res["oracle"] = _oracle(frames, paint, ltl_all, worn_below)
        res["sides"] = _sides(frames, has, ltl_all, no_paint_below)
        rv = np.array([np.nan if f.road_v is None else f.road_v for f in frames])
        okv = paint & ~np.isnan(rv)
        cst = sc["contrast"]
        okc = okv & ~np.isnan(cst)
        res["road_brightness"] = {"n": int(okv.sum()), "spearman_ltl": spearman(rv[okv], ltl_all[okv]),
                                  "spearman_contrast": spearman(rv[okc], cst[okc])}
        out[sub] = res
    out["lane_geometry"] = _geometry(frames)
    out["ltl_consistency"] = ltl_consistency(frames, no_paint_below)
    return out


def _rate(x: np.ndarray) -> float | None:
    return float(np.mean(x)) if len(x) else None


def _accuracy(score: np.ndarray, worn: np.ndarray, thr: float) -> dict:
    """잠정 기준에서의 정확도와 "전부 정상" 이라고 부를 때의 정확도. 차선을 못 찾으면(NaN) 정상으로 본다 (실제로도
    보내지 않는다). 양성이 20% 안팎이면 정확도만으로는 판단할 수 없다 — AUC·F1 을 같이 본다."""
    pred = ~np.isnan(score) & (np.nan_to_num(score, nan=np.inf) < thr)
    tp = int(np.sum(pred & worn))
    return {"n": int(len(worn)), "positive_rate": _rate(worn), "accuracy": _rate(pred == worn),
            "accuracy_all_intact": _rate(~worn), "precision": tp / max(1, int(pred.sum())),
            "recall": tp / max(1, int(worn.sum()))}


def _sides(frames: list[Frame], has: np.ndarray, ltl: np.ndarray, no_paint_below: float) -> dict:
    """측정기가 어느 쪽 선을 재나: 쪽마다 (1) 라벨 선이 있을 때·없을 때 LTL-M < 30 비율, (2) 그 선까지 거리와
    LTL-M(≥30) 의 Spearman, (3) 거리로 < 30 을 가르는 AUC (멀수록 < 30 이면 0.5 보다 크다)."""
    out = {}
    for side in ("left", "right"):
        idx = [i for i in np.flatnonzero(has) if frames[i].ego_off]
        off = np.array([np.nan if frames[i].ego_off.get(side) is None else frames[i].ego_off[side] for i in idx])
        v = ltl[idx]
        low = v < no_paint_below
        line = ~np.isnan(off)
        pm = line & ~low
        out[side] = {"n": len(idx), "low_rate_with_line": _rate(low[line]), "low_rate_without_line": _rate(low[~line]),
                     "offset_spearman_ltl": spearman(off[pm], v[pm]) if pm.sum() >= 3 else None,
                     "offset_auc_low": auc_low_is_worn(-off[line], low[line]) if line.any() else None}
    return out


def _oracle(frames: list[Frame], paint: np.ndarray, ltl: np.ndarray, worn_below: float) -> dict:
    """라벨 선 위치의 기준 대비가 LTL-M 과 어울리나 — 어느 쪽 선을 재는지, 화면 밝기로 재귀반사를 볼 수 있는지."""
    out = {}
    for side in ("left", "right"):
        idx = [i for i in np.flatnonzero(paint) if frames[i].oracle.get(side) is not None]
        if len(idx) < 3:
            out[side] = {"n": len(idx)}
            continue
        c = np.array([frames[i].oracle[side] for i in idx])
        v = ltl[idx]
        w = v < worn_below
        out[side] = {"n": len(idx), "spearman_ltl": spearman(c, v), "auc": auc_low_is_worn(c, w),
                     "n_worn": int(w.sum())}
    return out


def ltl_consistency(frames: list[Frame], no_paint_below: float = NO_PAINT_BELOW, max_dt: float = 2.0,
                    near_m: float = 10.0) -> dict:
    """LTL-M 이 믿을 만한가: (1) 같은 주행에서 max_dt 초 안의 이웃 프레임끼리, (2) 다른 주행이 near_m 안의 같은 곳을
    지날 때 (낮-낮·낮-밤·밤-밤). GPS 가 있는 프레임만."""
    fs = sorted((f for f in frames if f.gps is not None and f.ltl is not None), key=lambda f: (f.drive, f.gps[0]))
    a, b, prev_np, next_np = [], [], [], []
    for x, y in zip(fs, fs[1:]):
        if x.drive == y.drive and 0 < y.gps[0] - x.gps[0] < max_dt:
            prev_np.append(x.ltl < no_paint_below)
            next_np.append(y.ltl < no_paint_below)
            if x.ltl >= no_paint_below and y.ltl >= no_paint_below:
                a.append(x.ltl)
                b.append(y.ltl)
    pn, nn = np.array(prev_np, bool), np.array(next_np, bool)
    out: dict = {"consecutive": {"n_pairs": len(a), "spearman": spearman(np.array(a), np.array(b)),
                                 "median_abs_diff": float(np.median(np.abs(np.array(a) - b))) if a else None,
                                 "p_no_paint_after_no_paint": _rate(nn[pn]), "p_no_paint_after_paint": _rate(nn[~pn])}}
    paint = [f for f in fs if f.ltl >= no_paint_below]
    if len(paint) >= 2:
        lat = np.array([f.gps[1] for f in paint])
        lon = np.array([f.gps[2] for f in paint])
        xy = np.stack([lat * 111_000.0, lon * 111_000.0 * np.cos(np.radians(lat.mean()))], 1)
        pairs: dict[str, list[tuple[float, float]]] = {}
        for i in range(len(paint)):
            d = np.hypot(*(xy[i + 1:] - xy[i]).T)
            for j in np.flatnonzero(d < near_m) + i + 1:
                p, q = paint[i], paint[j]
                if p.drive == q.drive:
                    continue
                if p.time > q.time:
                    p, q = q, p
                pairs.setdefault(f"{p.time}-{q.time}", []).append((p.ltl, q.ltl))
        out["repeat_pass"] = {k: {"n_pairs": len(v), "spearman": spearman(np.array(v)[:, 0], np.array(v)[:, 1]),
                                  "mean_first": float(np.mean(np.array(v)[:, 0])),
                                  "mean_second": float(np.mean(np.array(v)[:, 1]))}
                              for k, v in sorted(pairs.items())}
    out["repeat_pass_no_paint"] = _repeat_low_agreement(fs, no_paint_below, near_m)
    return out


def _repeat_low_agreement(fs: list[Frame], no_paint_below: float, near_m: float) -> dict:
    """다른 주행이 같은 곳(near_m 안)을 지날 때 '< no_paint_below' 가 되풀이되나. 닳은 페인트(장소의 성질)라면
    한쪽이 낮을 때 다른 쪽도 낮아야 하고, 측정기가 순간적으로 페인트를 벗어난 것이라면 기저 비율 근처다."""
    if len(fs) < 2:
        return {"n_pairs": 0}
    lat = np.array([f.gps[1] for f in fs])
    lon = np.array([f.gps[2] for f in fs])
    xy = np.stack([lat * 111_000.0, lon * 111_000.0 * np.cos(np.radians(lat.mean()))], 1)
    pr = []
    for i in range(len(fs)):
        d = np.hypot(*(xy[i + 1:] - xy[i]).T)
        for j in np.flatnonzero(d < near_m) + i + 1:
            if fs[i].drive != fs[j].drive:
                pr.append((fs[i].ltl < no_paint_below, fs[j].ltl < no_paint_below))
    if not pr:
        return {"n_pairs": 0}
    a = np.array(pr, bool)
    n_low = int(a.sum())
    return {"n_pairs": len(a), "base_low_rate": float(a.mean()),
            "p_other_low_given_low": 2 * int(np.sum(a[:, 0] & a[:, 1])) / n_low if n_low else None}


def _geometry(frames: list[Frame]) -> dict:
    lab = [f for f in frames if f.has_label]
    out: dict = {"n_labelled": len(lab)}
    for sub in ("light", "night"):
        fs = [f for f in lab if f.time == sub]
        ego = [f for f in fs if f.ego_left or f.ego_right]
        none = [f for f in fs if not (f.ego_left or f.ego_right)]
        pts = sum(f.seg_pts for f in fs)
        hit = sum(f.seg_hit for f in fs)
        per = [f.seg_hit / f.seg_pts for f in fs if f.seg_pts > 0]
        out[sub] = {
            "n": len(fs),
            "lane_found_rate_with_ego_line": _rate(np.array([f.result.lane_found for f in ego])),
            "lane_found_rate_without_ego_line": _rate(np.array([f.result.lane_found for f in none])),
            "n_with_ego_line": len(ego), "n_without_ego_line": len(none),
            "segment_precision": hit / pts if pts else None,  # 채택 선분 길이 중 라벨 선 위 비율
            "frames_mostly_on_lines": _rate(np.array(per) >= 0.5) if per else None,
        }
    return out


# ---- 출력 ------------------------------------------------------------------------------


def to_markdown(res: dict) -> str:
    st = res["settings"]
    lines = [f"프레임 {res['n_frames']}개 (LTL-M 있음 {res['n_with_ltl']}). 페인트 없음 < {st['no_paint_below']:g}, "
             f"마모 < {st['worn_below']:g}, 잠정 기준 {st['provisional_wear_score']:g}", ""]
    for sub in SUBSETS:
        r = res[sub]
        lf = r["lane_found_rate"]
        lines.append(f"## {sub}: 프레임 {r['n']}, LTL-M {r['n_ltl']} (페인트 없음 {r['n_no_paint']}, 페인트 {r['n_paint']}, "
                     f"마모 {r['n_worn']}), LTL-M 분포 {r['ltl_percentiles_paint']}")
        lines.append("차선 찾음 비율: " + ", ".join(f"{k} {_fmt(v)}" for k, v in lf.items()))
        lines.append("")
        lines.append("| 점수 (낮을수록 마모) | n (마모) | AUC | best F1 | 기준값 | 2-fold F1 | Spearman(LTL-M) |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, m in list(r["scores"].items()) + [("★, 30 미만도 마모로", r["star_if_no_paint_is_worn"])]:
            lines.append(f"| {k} | {m['n']} ({m['n_worn']}) | {_fmt(m['auc'])} | {_fmt(m['best_f1'])} | "
                         f"{_fmt(m['best_threshold'])} | {_fmt(m['f1_2fold'])} | {_fmt(m['spearman_ltl'])} |")
        lines.append("")
        lines.append("| ★ 기준값 | 마모로 본 비율 | 정상→마모 | 정밀도 | 재현율 | F1 | 페인트 없음 프레임 중 마모로 본 비율 |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in r["star_threshold_rows"]:
            t = row["threshold"]
            lines.append(f"| {t:g} | {_pct(row['flag'])} | {_pct(row['clear_to_worn'])} | {_fmt(row['precision'])} | "
                         f"{_fmt(row['recall'])} | {_fmt(row['f1'])} | {_pct(r['star_flag_rate_no_paint'][str(t)])} |")
        lines.append("")
        rb = r["road_brightness"]
        lines.append(f"노면 밝기(차선 아님) n={rb['n']}: Spearman(LTL-M) {_fmt(rb['spearman_ltl'])}, "
                     f"Spearman(contrast) {_fmt(rb['spearman_contrast'])}")
        for key, title in (("below_worn", f"마모 < {st['worn_below']:g}"), ("below_cutoff", f"< {st['no_paint_below']:g} 을 마모로")):
            acc = r["accuracy"][key]
            lines.append(f"정확도 ({title}, ★ < {st['provisional_wear_score']:g}, 차선 못 찾음 = 정상): {_pct(acc['accuracy'])} "
                         f"vs 전부 정상 {_pct(acc['accuracy_all_intact'])} (양성 {_pct(acc['positive_rate'])}, "
                         f"정밀도 {_fmt(acc['precision'])}, 재현율 {_fmt(acc['recall'])})")
        lines.append(f"< {st['no_paint_below']:g} 을 마모로: " + ", ".join(
            f"{k} AUC {_fmt(m['auc'])} 2-fold F1 {_fmt(m['f1_2fold'])}" for k, m in r["worn_is_below_cutoff"].items()))
        lines.append("쪽별 (라벨 선): " + ", ".join(f"{s} " + " ".join(f"{k}={_fmt(v)}" for k, v in o.items())
                                                 for s, o in r["sides"].items()))
        lines.append("라벨 선 기준 대비 (oracle): " + ", ".join(
            f"{s} n={o['n']} Spearman {_fmt(o.get('spearman_ltl'))} AUC {_fmt(o.get('auc'))}"
            for s, o in r["oracle"].items()))
        lines.append("")
    lines.append("## 차선 위치 (VL4 polyline)")
    for sub in ("light", "night"):
        lines.append(f"- {sub}: " + ", ".join(f"{k} {_fmt(v)}" for k, v in res["lane_geometry"][sub].items()))
    lines.append("")
    lines.append(f"## LTL-M 일관성 (GPS): {json.dumps(res['ltl_consistency'], ensure_ascii=False)}")
    return "\n".join(lines)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.1%}"


def _clean(o):
    """JSON 저장용: NaN → None, 소수 4자리."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, float):
        return None if np.isnan(o) else round(o, 4)
    return o


def frame_to_cache(f: Frame) -> dict:
    d = asdict(f)
    d["result"] = asdict(f.result)
    return d


def frame_from_cache(d: dict) -> Frame:
    d = dict(d)
    if d.get("gps") is not None:
        d["gps"] = tuple(d["gps"])
    d["result"] = LaneWearResult(**d["result"])
    return Frame(**d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="차선 마모 판정 vs 실측 재귀반사도 LTL-M (AI Hub 로드마크 654)")
    ap.add_argument("--ltlm", type=Path, required=True, help="LTL-M CSV 폴더 (<name>.csv, 숫자 하나)")
    ap.add_argument("--images", type=Path, required=True, help="PNG 폴더 또는 VS4 zip")
    ap.add_argument("--labels", type=Path, help="VL4 2D_JSON 폴더 (차선 위치 확인용, 선택)")
    ap.add_argument("--gps", type=Path, help="GPS CSV 폴더 (LTL-M 일관성 확인용, 선택)")
    ap.add_argument("--all-images", action="store_true", help="LTL-M 없는 프레임도 잰다 (차선 위치 확인·기준값용)")
    ap.add_argument("--no-paint-below", type=float, default=NO_PAINT_BELOW)
    ap.add_argument("--worn-below", type=float, default=WORN_BELOW)
    ap.add_argument("--limit", type=int, help="앞에서 N 프레임만 (시험용)")
    ap.add_argument("--cache", type=Path, help="프레임별 측정 캐시 JSON (레포 밖에 둘 것 — AI Hub 재배포 금지)")
    ap.add_argument("--out", type=Path, help="집계 결과 JSON (레포에는 이것만)")
    a = ap.parse_args(argv)

    ltl = load_ltlm(a.ltlm)
    if a.cache is not None and a.cache.exists():
        frames = [frame_from_cache(d) for d in json.loads(a.cache.read_text(encoding="utf-8"))]
        refresh_meta(frames, a.labels, a.gps)
    else:
        src = ImageSource(a.images)
        names = src.names() if a.all_images else [n for n in src.names() if n in ltl]
        names = names[: a.limit] if a.limit else names
        frames = []
        for i, f in enumerate(measure_frames(names, src, LaneWearMeter(), ltl, a.labels, gps=a.gps)):
            frames.append(f)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(names)}", file=sys.stderr)
        if a.cache is not None:
            a.cache.write_text(json.dumps([frame_to_cache(f) for f in frames]), encoding="utf-8")
    res = _clean(evaluate(frames, a.no_paint_below, a.worn_below))
    print(to_markdown(res))
    if a.out is not None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
