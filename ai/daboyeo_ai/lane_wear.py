"""차선 마모 측정 — 학습 없이 영상 처리만 쓴다 (PRD FR-1.3, 계획 2.6).

가상 블랙박스는 `Detector` 와 같은 모양으로 쓴다:

    meter = LaneWearMeter()                 # 한 번 만들어 재사용
    tracker = LaneWearTracker()             # 주행(영상)마다 하나
    j = tracker.update(meter.measure(frame_bgr), t=timestamp_s)   # OpenCV BGR 프레임 그대로, t 는 선택
    if j.confirmed:                         # 확정 1건마다 1번 (r.is_worn · j.is_worn 으로 보내지 않는다)
        ...  # ObservationIn(defect_type=DefectType.LANE_WEAR, bbox=None, confidence=j.confidence, ...)

방법 (CPU 약 4 ms, 1280×720 기준. 화소 단위 값은 모두 작업 해상도 폭 640 기준이라 해상도와 무관하게 같게 동작):
1. 프레임을 폭 `work_width`(640) 로 줄이고, 아래쪽 육각형 ROI 만 본다.
2. 밝기 V = max(R,G,B). **극성**: 차선은 양옆 노면보다 밝은 줄무늬다. 행마다 x 의 밝기를 같은 행 x±d 와
   비교해 **양쪽 모두**보다 `rel_delta` 비율 이상 밝은 화소만 후보로 둔다. d 는 그 행에서의 차선 폭 상한
   (원근 때문에 위로 갈수록 좁다). 균열·이음매(어두운 선)와 연석·그림자 경계(한쪽만 밝음), 차선보다 넓은
   밝은 덩어리(차 뒷면·하늘)는 여기서 빠진다. 노란 차선은 색상·채도로 후보에 더한다.
3. 후보 마스크에서 확률적 Hough 로 선분을 찾고, 기하 조건으로 거른다:
   - 수평에서 20° 이상 (정지선·횡단보도 가로줄 제외)
   - 원근: 가운데에서 떨어진 선분은 늘였을 때 선분 위쪽, 지평선 높이쯤에서 화면 가운데와 만나야 한다
     (반대로 기운 선 제외). 가운데의 거의 수직인 선분은 화면 아래쪽에 있을 때만 (기둥·사람 제외).
     화면 맨 아래까지 늘인 x 가 화면에서 멀리 벗어나면 노면보다 높은 가드레일이다.
   - 선분을 따라 재 봐도 양옆보다 밝아야 하고, 양옆 밝기가 비슷해야 한다 (한쪽이 풀·보도인 연석·가드레일 제외)
   - **소실점 투표**: 남은 선분들이 가장 많이 모이는 한 점을 찾고, 그 점을 향하지 않는 선분은 버린다
     (노면 질감·물웅덩이 테두리 조각은 방향이 제각각이라 빠진다).
   남은 선분 길이 합이 `min_total_len` 보다 짧으면 못 찾은 것으로 둔다.
4. **contrast = (차선 V 중앙값 − 바로 옆 노면 V 중앙값) / 노면 V 중앙값.** 노면은 선분과 같은 행에서
   차선 폭의 2배 떨어진 양옆이다 (ROI 전체 노면과 비교하면 그림자·다른 차로 때문에 크게 흔들린다).
   극성 조건 때문에 contrast 는 항상 양수다.
5. contrast < threshold 이면 마모(is_worn). **이 절대 판정은 호환용이다** — 손 라벨 143장에서 AUC 0.56.
6. 페인트 양: `contrast_p25` (행마다 잰 대비의 하위 25%, 벗겨진 구간이 많으면 낮다) 와 `paint_len`
   (채택 선분 길이 합 ÷ 폭). 판정은 `LaneWearTracker` 가 같은 주행의 보통 수준과 비교해서 한다
   (`score = contrast_p25 ÷ 주행 중앙값 × paint_len < 1.0`, 같은 라벨에서 AUC 0.79) 하고, 최근 판정 5 프레임 중
   3 프레임 이상이 마모일 때만 "마모 관측" 1건으로 확정한다 (`WearJudgement.confirmed`).

한계 (판정에 반드시 반영할 것):
- **차선이 완전히 지워진 도로와 차선이 원래 없는 도로는 구별할 수 없다.** 그래서 차선 선분을 못 찾으면
  `lane_found=False, is_worn=False` 를 준다. 마모로 보고하지 않는다. 노면보다 `rel_delta`(8%) 이상 밝지 않은
  차선(밝기로는 노면과 같고 색만 남은 노란 차선 등)도 여기에 들어간다.
- 후보는 "양옆 노면보다 밝은 화소"라서, 일부만 남은 차선은 남은 페인트 위주로 재진다
  (contrast 가 실제보다 덜 낮게 나올 수 있음). 남은 양은 `lane_pixel_ratio` 로 참고한다.
- 노면보다 높은 줄무늬(가드레일 레일, 연석 위 경계)가 소실점을 향하면 차선으로 섞일 수 있다.
  햇빛 받은 밝은 노면에서는 선명한 차선도 contrast 가 0.2 안팎으로 낮게 나온다.
- **야간·비·역광**: 전조등·젖은 노면 반사·그림자가 밝기 대비를 크게 바꾼다. 품질 판별(F1 품질 단계)을
  통과한 낮 프레임에서만 부르고, 한 프레임 결과로 확정하지 말고 **같은 지점 여러 번 관측**으로 확정한다
  (오픈 이슈 #6, 기준값·확정 규칙은 실측 분포로 11/22 결정).
- ROI 기본값은 카메라가 차량 정면에 달려 있고 지평선이 화면 높이 0.2~0.5 사이라고 가정한다 (KRID·AI Hub
  도로장애물 프레임 300장으로 맞춤). 보닛이 화면 아래 8% 보다 크게 보이거나 각도가 다르면 `roi` 를 바꾼다.
  앞차가 ROI 를 가리면 차선을 못 찾을 수 있다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# 잠정값 (TBD) — 오픈 이슈 #6. 실제 프레임 분포(`python -m daboyeo_ai.eval.lane_wear_calib`)를 보고
# 11/22 에 다시 정한다. 실제 프레임 300장(2026-11-15) 분포: p5 0.22 / p50 0.50 / p95 1.2.
PROVISIONAL_THRESHOLD = 0.3

# 정규화 좌표 (x, y), 시계 방향 육각형. 위는 지평선이 높은(카메라가 아래로 기운) 블랙박스까지 덮도록 0.35,
# 아래 8% 는 보닛·대시보드 반사로 뺀다. 위쪽이 좁아 길가·건물을 덜 본다.
DEFAULT_ROI: tuple[tuple[float, float], ...] = (
    (0.30, 0.35), (0.70, 0.35), (1.00, 0.60), (1.00, 0.92), (0.00, 0.92), (0.00, 0.60))
# 줄무늬 폭 상한 d(y) 가 0 이 되는 높이. 실제 지평선은 블랙박스마다 0.2~0.5 라서 낮은 쪽(위쪽)으로 잡는다
# — 폭 상한이 실제보다 넉넉한 것은 괜찮지만 모자라면 가까운 차선을 놓친다.
HORIZON_Y = 0.2


@dataclass(frozen=True)
class LaneWearResult:
    lane_found: bool
    contrast: float | None  # (차선 V − 바로 옆 노면 V) / 노면 V. 항상 > 0. lane_found=False 면 None
    is_worn: bool  # lane_found 이고 contrast < threshold 일 때만 True
    confidence: float  # 0~1, is_worn 판정의 신뢰도 (ObservationIn.confidence 로 쓴다)
    lane_pixel_ratio: float  # 차선 화소 / ROI 화소 (진단용)
    n_segments: int  # 모든 조건을 통과한 Hough 선분 수 (진단용)
    # 아래 둘은 상대 판정(`LaneWearTracker`)용. lane_found=False 면 None / 0.
    contrast_p25: float | None = None  # 선분을 따라 행마다 잰 대비의 하위 25% 값. 페인트가 군데군데 벗겨지면 낮다
    paint_len: float = 0.0  # 채택 선분 길이 합 ÷ 작업 해상도 폭. 남은 페인트 양 (해상도와 무관)


_NOT_FOUND = {"lane_found": False, "contrast": None, "is_worn": False, "confidence": 0.0}


class LaneWearMeter:
    """차선 대 노면 밝기 대비를 재서 마모를 판정한다. 상태가 없어 프레임마다 독립이다.

    화소 단위 값(줄무늬 폭, Hough 길이·간격, 최소 화소 수)은 모두 작업 해상도 폭 640 기준이고,
    작업 해상도가 그보다 작으면(입력이 640 보다 작을 때) 같은 비율로 줄인다.
    """

    def __init__(
        self,
        threshold: float = PROVISIONAL_THRESHOLD,
        roi: tuple[tuple[float, float], ...] = DEFAULT_ROI,
        work_width: int = 640,
        rel_delta: float = 0.08,  # 양옆 노면보다 8% 이상 밝으면 차선 후보
        min_delta: float = 4.0,  # 단, 최소 밝기 4 단계 (잡음 방지)
        min_angle_deg: float = 20.0,  # 수평에서 이 각도 미만인 선분은 차선이 아님
        min_lane_pixels: int = 150,  # 작업 해상도(폭 640) 기준 차선 화소가 이보다 적으면 못 찾은 것으로
        min_coverage: float = 0.7,  # Hough 선분 위 화소 중 후보 비율 최소값
        min_total_len: float = 160.0,  # 작업 해상도(폭 640) 기준 채택 선분 길이 합이 이보다 짧으면 못 찾은 것으로
        horizon_y: float = HORIZON_Y,  # 줄무늬 폭 상한이 0 이 되는 높이 (정규화)
        max_stripe: float = 0.035,  # 화면 맨 아래에서 차선 폭 상한 (프레임 폭 비율). 위로 갈수록 원근으로 줄어든다
        center_band: float = 0.08,  # 화면 가운데 ± 이 비율 안의 선분은 기울기 방향을 따지지 않는다
        bottom_reach: float = 0.3,  # 선분을 맨 아래까지 늘인 x 가 화면 밖으로 이 비율 넘게 나가면 차선 아님
        vp_tol_deg: float = 3.0,  # 선분 방향과 (선분 → 소실점) 방향의 허용 각도 차
        max_flank_diff: float = 0.25,  # 양옆 노면 밝기 차가 밝은 쪽의 이 비율을 넘으면 차선 아님
        # 아래 셋은 `measure(..., lane_mask=)` 로 차선 분할 마스크를 줄 때만 쓴다 (handoff 21장, 실험). 안 주면 지금과 같다.
        lane_mask_min: float = 0.5,  # 선분 화소 중 (넓힌) 마스크 안 비율이 이보다 낮으면 차선 아님
        lane_mask_grow: float = 4.0,  # 마스크를 이만큼(폭 640 기준 화소) 넓혀서 쓴다 — 분할 경계가 페인트보다 좁을 때 대비
        mask_rel_delta: float | None = None,  # 마스크 안에서만 쓸 rel_delta (젖은 노면용으로 낮춤). None 이면 rel_delta
    ):
        if threshold <= 0:
            raise ValueError("threshold 는 0보다 커야 한다")
        if len(roi) < 3:
            raise ValueError("roi 는 꼭짓점 3개 이상")
        self.threshold = float(threshold)
        self.roi = tuple((float(x), float(y)) for x, y in roi)
        self.work_width = int(work_width)
        self.rel_delta = rel_delta
        self.min_delta = min_delta
        self.min_angle_deg = min_angle_deg
        self.min_lane_pixels = min_lane_pixels
        self.min_coverage = min_coverage
        self.min_total_len = min_total_len
        self.horizon_y = horizon_y
        self.max_stripe = max_stripe
        self.center_band = center_band
        self.bottom_reach = bottom_reach
        self.vp_tol_deg = vp_tol_deg
        self.max_flank_diff = max_flank_diff
        self.lane_mask_min = lane_mask_min
        self.lane_mask_grow = lane_mask_grow
        self.mask_rel_delta = mask_rel_delta
        self._mask_cache: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int, int, int]]] = {}

    # ---- 공개 API ----------------------------------------------------------------

    def measure(self, frame_bgr: np.ndarray, lane_mask: np.ndarray | None = None) -> LaneWearResult:
        """lane_mask: (선택) 차선 분할 마스크 H×W (0 이 아니면 차선, 프레임과 같은 크기 또는 같은 비율).
        주면 그 안의 줄무늬만 차선 후보로 본다 — 횡단보도·노면 글자·화살표를 빼려는 것 (handoff 21장, 실험)."""
        return self.measure_debug(frame_bgr, lane_mask)[0]

    def measure_debug(self, frame_bgr: np.ndarray, lane_mask: np.ndarray | None = None
                      ) -> tuple[LaneWearResult, list[tuple[int, int, int, int]]]:
        """measure + 채택된 선분 (원본 프레임 픽셀 좌표 x1,y1,x2,y2). 보정 도구의 오버레이용."""
        return self._finish(self._segments(frame_bgr, lane_mask))

    def _segments(self, frame_bgr: np.ndarray, lane_mask: np.ndarray | None = None) -> dict:
        """1~3단계: 후보 마스크와 기하 조건·소실점 투표를 통과한 선분 (작업 해상도·ROI 좌표) 과 선분별 표본.
        `_finish` 가 4~6단계(길이 조건·대비·페인트 양)를 한다. 평가 도구가 선분을 더 거른 뒤 `_finish` 를 부를 수 있게 나눴다."""
        _check_frame(frame_bgr)
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else frame_bgr
        sh, sw = small.shape[:2]
        u = sw / 640.0  # 화소 단위 값의 배율
        roi_mask, (x0, y0, x1, y1) = self._roi_mask(sw, sh)
        crop = small[y0:y1, x0:x1]
        ch = crop.shape[0]
        n_roi = int(np.count_nonzero(roi_mask))

        # 밝기 = HSV 의 V (= max(R,G,B)). HLS 의 L 로 재면 노란 차선이 햇빛 받은 노면과 밝기가 비슷해져 놓친다.
        # 회색 노면·흰 차선은 V 와 L 이 거의 같다.
        hls = cv2.cvtColor(crop, cv2.COLOR_BGR2HLS)
        light = cv2.GaussianBlur(crop.max(2), (3, 3), 0).astype(np.float32)

        # 행마다 차선 폭 상한 d(y): 원근 때문에 지평선에서 0, 화면 맨 아래에서 max_stripe·폭.
        yy = np.arange(y0, y1, dtype=np.float32)
        d_row = np.clip(self.max_stripe * sw * (yy - self.horizon_y * sh) / ((1 - self.horizon_y) * sh),
                        2.0, None).round().astype(np.int32)

        # 극성: 차선은 **양옆** 노면보다 밝은 줄무늬다. x 의 밝기를 x±d 의 밝기와 비교해 둘 다보다 밝아야 후보.
        # 균열·이음매(어두운 선)와 연석·그림자 경계(한쪽만 밝은 가장자리)는 여기서 빠진다.
        left = np.empty_like(light)
        right = np.empty_like(light)
        for d in np.unique(d_row):
            rows = d_row == d
            band = light[rows]
            lpad = np.concatenate([np.repeat(band[:, :1], d, 1), band[:, :-d]], 1)
            rpad = np.concatenate([band[:, d:], np.repeat(band[:, -1:], d, 1)], 1)
            left[rows], right[rows] = lpad, rpad
        side = np.maximum(left, right)
        ridge = light - side
        need = np.maximum(self.min_delta, self.rel_delta * side)
        in_lane = None
        if lane_mask is not None:
            in_lane = self._lane_region(lane_mask, sw, sh, (x0, y0, x1, y1), u)
            if self.mask_rel_delta is not None:
                need = np.where(in_lane, np.maximum(self.min_delta, self.mask_rel_delta * side), need)
        bright = ridge >= need
        hue, sat = hls[:, :, 0], hls[:, :, 2]
        yellow = (hue >= 15) & (hue <= 35) & (sat >= 70) & (ridge >= self.min_delta)
        cand_b = (bright | yellow) & roi_mask
        if in_lane is not None:
            cand_b &= in_lane
        cand = cand_b.astype(np.uint8) * 255

        lines = cv2.HoughLinesP(cand, 1, np.pi / 180, threshold=max(8, round(20 * u)),
                                minLineLength=max(8, round(18 * u)), maxLineGap=max(3, round(8 * u)))
        segs: list[tuple[int, int, int, int]] = []
        lane_vals: list[np.ndarray] = []
        road_vals: list[np.ndarray] = []
        row_vals: list[np.ndarray] = []  # 행마다의 대비 (차선 − 밝은 쪽 옆 노면) / 밝은 쪽 옆 노면
        for xa, ya, xb, yb in (lines.reshape(-1, 4) if lines is not None else ()):
            xa, ya, xb, yb = int(xa), int(ya), int(xb), int(yb)
            if ya == yb or np.degrees(np.arctan2(abs(yb - ya), abs(xb - xa))) < self.min_angle_deg:
                continue
            # 원근: 차선은 위로 갈수록 화면 가운데로 모인다. 가운데에서 떨어진 선분은 늘였을 때 선분보다 위,
            # 지평선 높이쯤(-0.2~0.7·높이)에서 화면 가운데 세로줄과 만나야 한다. 반대로 기운 선(그림자·균열·차
            # 옆면)과 거의 수직인 기둥은 여기서 빠진다.
            slope = (xb - xa) / (yb - ya)  # 아래로 1 화소 갈 때 x 변화
            xm = (xa + xb) / 2 + x0 - sw / 2
            if abs(xm) <= self.center_band * sw:
                # 가운데의 거의 수직인 선분은 화면 아래쪽(지평선에서 먼 곳)에 있을 때만 차선. 위쪽이면 기둥·사람.
                if abs(slope) < 0.18 and max(ya, yb) + y0 < 0.7 * sh:
                    continue
            else:
                if slope == 0:
                    continue
                y_center = (ya + yb) / 2 + y0 - xm / slope
                if not (-0.2 * sh <= y_center <= min(0.7 * sh, min(ya, yb) + y0)):
                    continue
            x_at_bottom = xa + x0 + (sh - (ya + y0)) * slope
            if not (-self.bottom_reach * sw <= x_at_bottom <= (1 + self.bottom_reach) * sw):
                continue
            if _coverage(cand, xa, ya, xb, yb) < self.min_coverage:
                continue
            if in_lane is not None and _coverage(in_lane.view(np.uint8), xa, ya, xb, yb) < self.lane_mask_min:
                continue
            lv, rv = _stripe_samples(light, d_row, xa, ya, xb, yb)
            if lv.size < 3:
                continue
            lf, rf = rv.reshape(2, -1)
            if np.median(lv - np.maximum(lf, rf)) < self.min_delta:
                continue  # 선분을 따라 재 봐도 양옆보다 밝지 않으면 차선이 아니다
            # 차선은 양옆이 같은 노면이다. 한쪽이 풀·보도·그늘인 가드레일·연석은 양옆 밝기가 크게 다르다.
            lm, rm = float(np.median(lf)), float(np.median(rf))
            if abs(lm - rm) > self.max_flank_diff * max(lm, rm, 1.0):
                continue
            segs.append((xa, ya, xb, yb))
            lane_vals.append(lv)
            road_vals.append(rv)
            side_v = np.maximum(lf, rf)
            row_vals.append((lv - side_v) / np.maximum(side_v, 1.0))

        # 소실점 투표: 차선(과 그 연장)은 한 점으로 모이고, 노면 질감·기둥·차 테두리 조각은 제각각이다.
        # 선분 쌍의 교점 중 가장 많은(긴) 선분이 지나가는 점을 소실점으로 두고, 그 점을 향하지 않는 선분은 버린다.
        keep = _vanishing_inliers(np.array(segs, np.float32).reshape(-1, 4) + [x0, y0, x0, y0], sw, sh,
                                  self.vp_tol_deg)
        segs = [segs[i] for i in keep]
        lane_vals = [lane_vals[i] for i in keep]
        road_vals = [road_vals[i] for i in keep]
        row_vals = [row_vals[i] for i in keep]
        return {"segs": segs, "lane_vals": lane_vals, "road_vals": road_vals, "row_vals": row_vals, "cand": cand,
                "d_row": d_row, "scale": scale, "origin": (x0, y0), "size": (sw, sh), "ch": ch, "n_roi": n_roi, "u": u}

    def _finish(self, p: dict) -> tuple[LaneWearResult, list[tuple[int, int, int, int]]]:
        segs, lane_vals, road_vals, row_vals = p["segs"], p["lane_vals"], p["road_vals"], p["row_vals"]
        cand, d_row, scale, u, ch, n_roi = p["cand"], p["d_row"], p["scale"], p["u"], p["ch"], p["n_roi"]
        (x0, y0), (sw, _) = p["origin"], p["size"]
        total_len = float(sum(np.hypot(xb - xa, yb - ya) for xa, ya, xb, yb in segs))

        if not segs:
            return LaneWearResult(**_NOT_FOUND, lane_pixel_ratio=0.0, n_segments=0), []

        near = np.zeros_like(cand)
        for xa, ya, xb, yb in segs:
            cv2.line(near, (xa, ya), (xb, yb), 255, max(3, int(d_row[(ya + yb) // 2])))
        n_lane = int(np.count_nonzero((near > 0) & (cand > 0)))

        segs_px = [(round((xa + x0) / scale), round((ya + y0) / scale), round((xb + x0) / scale),
                    round((yb + y0) / scale)) for xa, ya, xb, yb in segs]
        ratio = n_lane / max(1, n_roi)
        # 선 구조가 짧으면(거친 노면 질감의 짧은 조각 몇 개) 차선으로 보지 않는다
        if n_lane < self.min_lane_pixels * u * u or total_len < self.min_total_len * u:
            return LaneWearResult(**_NOT_FOUND, lane_pixel_ratio=ratio, n_segments=len(segs)), segs_px

        # 대비는 선분 바로 옆(같은 행, 차선 폭만큼 떨어진 곳)의 노면과 잰다. ROI 전체 노면과 비교하면
        # 그림자·다른 차로 때문에 값이 크게 흔들린다.
        lane_l = float(np.median(np.concatenate(lane_vals)))
        road_l = float(np.median(np.concatenate(road_vals)))
        contrast = (lane_l - road_l) / max(road_l, 1.0)
        is_worn = contrast < self.threshold
        # 페인트 양: 행마다의 대비 하위 25% (벗겨진 구간이 많으면 낮다) 와 남은 선 길이. 점선은 칠한 토막만
        # 선분으로 잡히므로 토막 사이 빈칸은 길이에서 빠질 뿐 하위 25% 를 낮추지 않는다.
        contrast_p25 = float(np.percentile(np.concatenate(row_vals), 25))
        paint_len = total_len / sw

        # 신뢰도 = (선 구조가 충분한가) × (기준값에서 얼마나 떨어져 있나). 기준값 바로 위아래면 0.5×support.
        support = min(1.0, total_len / max(1.0, 1.5 * ch))
        certainty = min(1.0, abs(contrast - self.threshold) / self.threshold)
        confidence = support * (0.5 + 0.5 * certainty)
        return LaneWearResult(
            lane_found=True, contrast=contrast, is_worn=is_worn, confidence=float(np.clip(confidence, 0, 1)),
            lane_pixel_ratio=ratio, n_segments=len(segs), contrast_p25=contrast_p25, paint_len=paint_len,
        ), segs_px

    # ---- 내부 ----------------------------------------------------------------------

    def _lane_region(self, lane_mask: np.ndarray, sw: int, sh: int, box: tuple[int, int, int, int],
                     u: float) -> np.ndarray:
        """분할 마스크 → 작업 해상도·ROI 상자 크기의 bool (lane_mask_grow 만큼 넓힘)."""
        m = np.asarray(lane_mask)
        if m.ndim != 2:
            raise ValueError(f"lane_mask 는 2차원이어야 한다 (받은 모양 {m.shape})")
        m = (m > 0).astype(np.uint8)
        if m.shape != (sh, sw):
            m = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST)
        g = max(0, round(self.lane_mask_grow * u))
        if g:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        x0, y0, x1, y1 = box
        return m[y0:y1, x0:x1] > 0

    def roi_polygon_px(self, w: int, h: int) -> np.ndarray:
        return np.array([[round(x * (w - 1)), round(y * (h - 1))] for x, y in self.roi], np.int32)

    def _roi_mask(self, w: int, h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        key = (w, h)
        if key not in self._mask_cache:
            poly = self.roi_polygon_px(w, h)
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0) + 1
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly], 255)
            self._mask_cache[key] = (m[y0:y1, x0:x1] > 0, (int(x0), int(y0), int(x1), int(y1)))
        return self._mask_cache[key]


# 상대 판정 잠정값 (TBD) — 오픈 이슈 #6. 시연 주행 영상으로 다시 확인한다
# (`python -m daboyeo_ai.eval.lane_wear_eval` 의 기준값별 표).
# 11/15 1.7 → 1.0: 손 라벨 143장(KRID·AI Hub 도로장애물)에서 1.7 은 차선 찾은 프레임의 41% 를 마모로 보고
# 선명 차선의 18% 를 마모로 잘못 봤다 (P 0.51 / R 0.68). 1.0 이면 28% · 12% (P 0.41 / R 0.29), 0.6 이면 19% · 4.5%.
# 한 프레임 정밀도는 기준값과 상관없이 0.5 안팎이라, 엄격함은 기준값이 아니라 **반복 관측**(아래 연속 프레임 투표 +
# 서버의 여러 차량 확정)에서 얻는다. 1.0 은 오탐을 줄이되 투표가 모을 마모 프레임은 남기는 절충이다.
PROVISIONAL_WEAR_SCORE = 1.0
# 연속 프레임 투표 (잠정, #6): 최근 판정 프레임 confirm_n 개 중 confirm_k 개 이상이 마모면 "마모 관측" 1건.
CONFIRM_K, CONFIRM_N = 3, 5
# 타임스탬프를 주면, 차선을 찾은 프레임 사이가 이 초보다 벌어질 때 투표 창을 비운다 (다른 구간으로 본다).
MAX_GAP_S = 5.0


@dataclass(frozen=True)
class WearJudgement:
    """`LaneWearTracker.update` 결과. 한 프레임의 상대 판정 + 연속 프레임 투표."""

    lane_found: bool
    ready: bool  # 기준(baseline) 을 만들 만큼 차선 프레임이 모였나 (warmup 이 끝났나)
    baseline: float | None  # 이 주행의 보통 contrast_p25 (최근 window 개 차선 프레임의 중앙값)
    score: float | None  # (contrast_p25 ÷ baseline) × paint_len. 낮을수록 마모. ready=False 면 None
    is_worn: bool  # 이 프레임 하나의 판정: ready 이고 score < threshold. **이것만으로 보내지 않는다**
    confidence: float  # 0~1, is_worn 판정의 신뢰도
    # 투표 (기본값이 있어 예전 코드는 그대로 동작)
    confirmed: bool = False  # 이 프레임에서 "마모 관측" 이 확정됐나. True 인 프레임에서만 ObservationIn 1건을 보낸다
    votes: int = 0  # 투표 창(최근 판정 프레임 confirm_n 개) 안의 마모 프레임 수
    n_voted: int = 0  # 투표 창에 든 판정 프레임 수 (≤ confirm_n)


class LaneWearTracker:
    """한 주행(같은 카메라·비슷한 조명) 안에서 **상대적으로** 마모를 판정하고, 연속 프레임 투표로 확정한다.
    주행마다 하나씩 만든다.

        meter, tracker = LaneWearMeter(), LaneWearTracker()   # 주행 시작
        j = tracker.update(meter.measure(frame_bgr), t=timestamp_s)   # t 는 선택 (초)
        if j.confirmed:   # j.is_worn 이 아니라 confirmed 일 때만, 확정 1건마다 1번
            ...  # ObservationIn(defect_type=LANE_WEAR, bbox=None, confidence=j.confidence, ...)

    왜 상대 판정: 절대 대비(contrast)는 마모보다 카메라·노면색·조명에 더 크게 좌우된다 (KRID 선명 중앙값 0.47 vs
    AI Hub 도로장애물 마모 0.72). 그래서 대비는 같은 주행의 보통 값으로 나누고, 페인트가 남은 길이를 곱한다.

    score = (contrast_p25 ÷ baseline) × paint_len
      - contrast_p25: 선분을 따라 행마다 잰 대비의 하위 25%. 벗겨진 구간이 많으면 낮다.
      - baseline: 최근 `window` 개 차선 프레임의 contrast_p25 중앙값 (마모 프레임이 섞여도 중앙값이라 덜 흔들린다).
      - paint_len: 남은 차선 선분 길이 합 ÷ 폭. 지워져 토막 난 차선은 짧다.
    차선을 찾은 프레임이 `warmup` 개 모이기 전에는 판정하지 않는다 (is_worn=False, ready=False).
    현재 프레임은 판정한 **다음에** 기준에 넣는다.

    왜 투표: 한 프레임 판정은 기준값과 상관없이 절반쯤 오탐이다 (손 라벨 정밀도 ~0.5). 그래서
      - 투표 창 = 최근 **판정한** 프레임(차선 찾음 + warmup 뒤) `confirm_n` 개의 is_worn.
        `lane_found=False` 프레임은 창에 들어가지도, 창을 비우지도 않는다 (가림·교차로 한두 프레임은 건너뛴다).
      - 창 안 마모가 `confirm_k` 개 이상이 되면 그 프레임에서 `confirmed=True` (마모 관측 1건).
      - 같은 구간에서 매 프레임 다시 내지 않도록, 확정한 뒤에는 **창 안 마모가 0 개가 될 때까지**
        (선명 프레임이 confirm_n 개 이어질 때까지) 다시 확정하지 않는다. `cooldown` 을 주면 확정 뒤 판정 프레임
        `cooldown` 개가 지나도 다시 확정할 수 있다 (긴 마모 구간에서 주기적으로 1건씩).
      - 구간 끊김: `update(r, t=...)` 로 타임스탬프(초)를 주면, 차선 찾은 프레임 사이가 `max_gap_s` 보다 벌어질 때
        투표 창을 비우고 다시 확정할 수 있게 한다 (기준은 그대로 — 같은 주행). t 를 안 주면 끊김을 모르므로
        영상이 바뀌거나 오래 멈췄다면 호출 쪽이 `reset()` 한다 (기준과 투표 모두 비운다).
    투표 규칙의 효과는 독립 프레임(손 라벨)으로는 잴 수 없다 — 시연 주행 영상으로 확인한다 (#6).

    한계: 주행 전체가 마모 도로면 기준 자체가 낮아져 마모를 놓친다. 터널·야간으로 조명이 크게 바뀌면
    `reset()` 하거나 새 tracker 를 쓴다. paint_len 은 보이는 차로 수에 따라 달라진다 (편도 1차로는 짧다).
    """

    def __init__(
        self,
        threshold: float = PROVISIONAL_WEAR_SCORE,
        window: int = 100,
        warmup: int = 20,
        confirm_k: int = CONFIRM_K,
        confirm_n: int = CONFIRM_N,
        cooldown: int | None = None,
        max_gap_s: float = MAX_GAP_S,
    ):
        if threshold <= 0:
            raise ValueError("threshold 는 0보다 커야 한다")
        if window < 1 or not 1 <= warmup <= window:
            raise ValueError("1 <= warmup <= window 여야 한다")
        if not 1 <= confirm_k <= confirm_n:
            raise ValueError("1 <= confirm_k <= confirm_n 이어야 한다")
        if cooldown is not None and cooldown < 1:
            raise ValueError("cooldown 은 None 또는 1 이상")
        if max_gap_s <= 0:
            raise ValueError("max_gap_s 는 0보다 커야 한다")
        self.threshold = float(threshold)
        self.window = int(window)
        self.warmup = int(warmup)
        self.confirm_k = int(confirm_k)
        self.confirm_n = int(confirm_n)
        self.cooldown = cooldown
        self.max_gap_s = float(max_gap_s)
        self._recent: deque[float] = deque(maxlen=self.window)
        self._votes: deque[bool] = deque(maxlen=self.confirm_n)
        self._armed = True  # 다음 확정을 낼 수 있나
        self._since_confirm = 0  # 마지막 확정 뒤 판정한 프레임 수
        self._last_t: float | None = None  # 마지막으로 차선을 찾은 프레임의 타임스탬프

    @property
    def baseline(self) -> float | None:
        if len(self._recent) < self.warmup:
            return None
        return float(np.median(self._recent))

    def reset(self) -> None:
        """기준과 투표를 모두 비운다 (새 주행·영상, 조명이 크게 바뀜)."""
        self._recent.clear()
        self._reset_votes()
        self._last_t = None

    def _reset_votes(self) -> None:
        self._votes.clear()
        self._armed = True
        self._since_confirm = 0

    def score(self, r: LaneWearResult, baseline: float) -> float:
        """기준값이 주어졌을 때의 점수. 오프라인 평가(주행 전체 중앙값을 기준으로)에도 쓴다."""
        return float(r.contrast_p25) / max(baseline, 1e-3) * r.paint_len

    def _vote_fields(self) -> dict:
        return {"votes": sum(self._votes), "n_voted": len(self._votes)}

    def update(self, r: LaneWearResult, t: float | None = None) -> WearJudgement:
        """프레임 하나를 판정한다. t = 프레임 타임스탬프(초, 선택) — 주면 끊김(`max_gap_s`) 뒤 투표를 비운다."""
        if not r.lane_found or r.contrast_p25 is None:
            return WearJudgement(lane_found=False, ready=self.baseline is not None, baseline=self.baseline,
                                 score=None, is_worn=False, confidence=0.0, **self._vote_fields())
        if t is not None:
            if self._last_t is not None and t - self._last_t > self.max_gap_s:
                self._reset_votes()
            self._last_t = float(t)
        base = self.baseline
        self._recent.append(float(r.contrast_p25))
        if base is None:
            return WearJudgement(lane_found=True, ready=False, baseline=None, score=None, is_worn=False,
                                 confidence=0.0, **self._vote_fields())
        s = self.score(r, base)
        worn = s < self.threshold
        certainty = min(1.0, abs(s - self.threshold) / self.threshold)

        self._votes.append(worn)
        self._since_confirm += 1
        votes = sum(self._votes)
        if not self._armed and (votes == 0 or (self.cooldown is not None and self._since_confirm >= self.cooldown)):
            self._armed = True
        confirmed = self._armed and votes >= self.confirm_k
        if confirmed:
            self._armed = False
            self._since_confirm = 0
        return WearJudgement(lane_found=True, ready=True, baseline=base, score=s, is_worn=worn,
                             confidence=float(0.5 + 0.5 * certainty), confirmed=confirmed, votes=votes,
                             n_voted=len(self._votes))


def _vanishing_inliers(segs: np.ndarray, w: int, h: int, tol_deg: float, max_segs: int = 40) -> list[int]:
    """선분(N×4, 프레임 좌표)들이 가장 많이 모이는 소실점을 찾아, 그 점을 향하는 선분 번호를 돌려준다.

    소실점 후보 = 긴 선분 max_segs 개의 쌍별 교점 중 화면 가운데 위쪽 (x 0.2~0.8, y −0.1~0.65 · 높이).
    점수 = 그 점을 tol_deg 안으로 향하는 선분 길이의 합. 후보가 없으면(선분 1개, 평행) 빈 목록.
    """
    if len(segs) < 2:
        return []
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-length)[:max_segs]
    s, ln = segs[order].astype(np.float64), length[order].astype(np.float64)
    p = np.stack([s[:, 0], s[:, 1], np.ones(len(s))], 1)
    q = np.stack([s[:, 2], s[:, 3], np.ones(len(s))], 1)
    lines = np.cross(p, q)  # 동차 좌표 직선
    i, j = np.triu_indices(len(s), 1)
    pts = np.cross(lines[i], lines[j])
    ok = np.abs(pts[:, 2]) > 1e-9
    vp = pts[ok, :2] / pts[ok, 2:3]
    inside = (vp[:, 0] > 0.2 * w) & (vp[:, 0] < 0.8 * w) & (vp[:, 1] > -0.1 * h) & (vp[:, 1] < 0.65 * h)
    vp = vp[inside]
    if len(vp) == 0:
        return []
    mid = (s[:, :2] + s[:, 2:]) / 2
    direc = (s[:, 2:] - s[:, :2]) / ln[:, None]
    to_vp = vp[:, None, :] - mid[None, :, :]  # (후보, 선분, 2)
    dist = np.linalg.norm(to_vp, axis=2) + 1e-9
    sin = np.abs(direc[None, :, 0] * to_vp[..., 1] - direc[None, :, 1] * to_vp[..., 0]) / dist
    inl = sin <= np.sin(np.radians(tol_deg))
    best = int(np.argmax((inl * ln[None, :]).sum(1)))
    return sorted(int(order[k]) for k in np.flatnonzero(inl[best]))


def _stripe_samples(light: np.ndarray, d_row: np.ndarray, xa: int, ya: int, xb: int, yb: int):
    """선분을 따라 행마다 (차선 밝기, [왼쪽 노면..., 오른쪽 노면...]) 를 뽑는다.

    차선 밝기는 선분 위치 ±d/3 안의 최대값 (Hough 위치가 줄무늬 가운데에서 1~2 화소 어긋나도 되게),
    노면은 같은 행에서 ±2d 떨어진 곳. 프레임 밖으로 나가는 행은 뺀다.
    """
    hgt, wid = light.shape
    lo, hi = sorted((ya, yb))
    ys = np.arange(lo, hi + 1)
    xs = np.round(xa + (ys - ya) * (xb - xa) / (yb - ya)).astype(np.int32)
    d = d_row[ys]
    ok = (xs - 2 * d >= 0) & (xs + 2 * d < wid)
    ys, xs, d = ys[ok], xs[ok], d[ok]
    if ys.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    r = np.maximum(1, d // 3)
    offs = np.arange(-int(r.max()), int(r.max()) + 1)
    cols = np.clip(xs[:, None] + np.clip(offs[None, :], -r[:, None], r[:, None]), 0, wid - 1)
    lane = light[ys[:, None], cols].max(1)
    road = np.concatenate([light[ys, xs - 2 * d], light[ys, xs + 2 * d]])
    return lane, road


def _coverage(mask: np.ndarray, xa: int, ya: int, xb: int, yb: int) -> float:
    n = max(2, int(max(abs(xb - xa), abs(yb - ya))) + 1)
    xs = np.linspace(xa, xb, n).round().astype(int)
    ys = np.linspace(ya, yb, n).round().astype(int)
    return float(np.count_nonzero(mask[ys, xs])) / n


def _check_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame_bgr 는 numpy 배열이어야 한다 (받은 것: {type(frame).__name__})")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame_bgr 는 (H, W, 3) BGR 이어야 한다 (받은 shape: {frame.shape})")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame_bgr 는 uint8 이어야 한다 (받은 dtype: {frame.dtype})")
    if frame.shape[0] < 32 or frame.shape[1] < 32:
        raise ValueError(f"프레임이 너무 작다: {frame.shape}")
