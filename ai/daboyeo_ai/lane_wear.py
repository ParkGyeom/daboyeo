"""차선 마모 측정 — 학습 없이 영상 처리만 쓴다 (PRD FR-1.3, 계획 2.6).

가상 블랙박스는 `Detector` 와 같은 모양으로 쓴다:

    meter = LaneWearMeter()                 # 한 번 만들어 재사용
    r = meter.measure(frame_bgr)            # OpenCV BGR 프레임 그대로
    if r.lane_found and r.is_worn:
        ...  # ObservationIn(defect_type=DefectType.LANE_WEAR, bbox=None, confidence=r.confidence, ...)

방법 (CPU 몇 ms, 1280×720 기준):
1. 프레임을 폭 `work_width`(640) 로 줄이고, 아래쪽 사다리꼴 ROI 만 본다.
2. HLS 밝기(L)를 흐리게 한 뒤 **가로 방향 top-hat** 을 한다. top-hat = L − (주변 노면 밝기 추정).
   주변 노면보다 `rel_delta` 비율 이상 밝은 화소를 차선 후보로 둔다 → 절대 밝기 기준이 아니라서
   흐려진 차선도 후보가 된다. 노란 차선은 색상(H)·채도(S) 로 따로 후보에 더한다.
3. 후보 마스크에서 확률적 Hough 로 선분을 찾고, 기울기가 차선다운 것(수평에서 25° 이상)만 남긴다.
   점 잡음·정지선·횡단보도 가로줄은 여기서 빠진다.
4. 남은 선분 근처의 후보 화소 = 차선, 차선에서 떨어진 ROI 화소 = 노면.
   **contrast = (차선 L 중앙값 − 노면 L 중앙값) / 노면 L 중앙값** (노면 대비 몇 배 더 밝은지).
   비율로 쓰는 이유: 낮·흐림처럼 전체 밝기가 달라져도 값이 덜 흔들린다.
5. contrast < threshold 이면 마모(is_worn).

한계 (판정에 반드시 반영할 것):
- **차선이 완전히 지워진 도로와 차선이 원래 없는 도로는 구별할 수 없다.** 그래서 차선 선분을 못 찾으면
  `lane_found=False, is_worn=False` 를 준다. 마모로 보고하지 않는다.
- 후보는 "노면보다 조금이라도 밝은 화소"라서, 일부만 남은 차선은 남은 페인트 위주로 재진다
  (contrast 가 실제보다 덜 낮게 나올 수 있음). 남은 양은 `lane_pixel_ratio` 로 참고한다.
- **야간·비·역광**: 전조등·젖은 노면 반사·그림자가 밝기 대비를 크게 바꾼다. 품질 판별(F1 품질 단계)을
  통과한 낮 프레임에서만 부르고, 한 프레임 결과로 확정하지 말고 **같은 지점 여러 번 관측**으로 확정한다
  (오픈 이슈 #6, 기준값·확정 규칙은 실측 분포로 11/22 결정).
- ROI 기본값은 카메라가 차량 정면·수평에 달려 있고 지평선이 화면 가운데쯤이라고 가정한다. 보닛이
  크게 보이거나 각도가 다르면 `roi` 를 바꾼다. 앞차가 ROI 를 가리면 차선을 못 찾을 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 잠정값 (TBD) — 오픈 이슈 #6. 실제 프레임 분포(`python -m daboyeo_ai.eval.lane_wear_calib`)를 보고
# 11/22 에 다시 정한다. 합성 시험에서 선명한 흰 차선은 1 이상, 흐린 차선은 0.1~0.2 정도.
PROVISIONAL_THRESHOLD = 0.3

# 정규화 좌표 (x, y), 시계 방향: 위-왼, 위-오, 아래-오, 아래-왼. 아래 5% 는 보닛 대비로 뺀다.
DEFAULT_ROI: tuple[tuple[float, float], ...] = ((0.40, 0.60), (0.60, 0.60), (1.00, 0.95), (0.00, 0.95))


@dataclass(frozen=True)
class LaneWearResult:
    lane_found: bool
    contrast: float | None  # (차선 L − 노면 L) / 노면 L. lane_found=False 면 None
    is_worn: bool  # lane_found 이고 contrast < threshold 일 때만 True
    confidence: float  # 0~1, is_worn 판정의 신뢰도 (ObservationIn.confidence 로 쓴다)
    lane_pixel_ratio: float  # 차선 화소 / ROI 화소 (진단용)
    n_segments: int  # 기울기 조건을 통과한 Hough 선분 수 (진단용)


_NOT_FOUND = {"lane_found": False, "contrast": None, "is_worn": False, "confidence": 0.0}


class LaneWearMeter:
    """차선 대 노면 밝기 대비를 재서 마모를 판정한다. 상태가 없어 프레임마다 독립이다."""

    def __init__(
        self,
        threshold: float = PROVISIONAL_THRESHOLD,
        roi: tuple[tuple[float, float], ...] = DEFAULT_ROI,
        work_width: int = 640,
        rel_delta: float = 0.08,  # 노면보다 8% 이상 밝으면 차선 후보
        min_delta: float = 4.0,  # 단, 최소 L 4 단계 (잡음 방지)
        min_angle_deg: float = 25.0,  # 수평에서 이 각도 미만인 선분은 차선이 아님
        min_lane_pixels: int = 150,  # 작업 해상도 기준 차선 화소가 이보다 적으면 못 찾은 것으로
        min_coverage: float = 0.7,  # Hough 선분 위 화소 중 후보 비율 최소값
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
        self._mask_cache: dict[tuple[int, int], tuple[np.ndarray, tuple[int, int, int, int]]] = {}

    # ---- 공개 API ----------------------------------------------------------------

    def measure(self, frame_bgr: np.ndarray) -> LaneWearResult:
        return self.measure_debug(frame_bgr)[0]

    def measure_debug(self, frame_bgr: np.ndarray) -> tuple[LaneWearResult, list[tuple[int, int, int, int]]]:
        """measure + 채택된 선분 (원본 프레임 픽셀 좌표 x1,y1,x2,y2). 보정 도구의 오버레이용."""
        _check_frame(frame_bgr)
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1.0 else frame_bgr
        roi_mask, (x0, y0, x1, y1) = self._roi_mask(small.shape[1], small.shape[0])
        crop = small[y0:y1, x0:x1]
        n_roi = int(np.count_nonzero(roi_mask))

        hls = cv2.cvtColor(crop, cv2.COLOR_BGR2HLS)
        raw_l = hls[:, :, 1]
        light = cv2.GaussianBlur(raw_l, (5, 5), 0)  # 후보 찾기용 (잡음 억제). 대비 값은 raw_l 로 잰다

        # 가로 opening = 차선보다 넓은 창에서 본 주변 노면 밝기. top-hat = 그보다 밝은 정도.
        k = max(9, (round(0.065 * small.shape[1]) | 1))
        background = cv2.morphologyEx(light, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
        tophat = light.astype(np.int16) - background.astype(np.int16)
        need = np.maximum(self.min_delta, self.rel_delta * background.astype(np.float32))
        bright = tophat >= need
        hue, sat = hls[:, :, 0], hls[:, :, 2]
        yellow = (hue >= 15) & (hue <= 35) & (sat >= 70) & (tophat >= self.min_delta)
        cand = ((bright | yellow) & roi_mask).astype(np.uint8) * 255

        roi_h = y1 - y0
        lines = cv2.HoughLinesP(cand, 1, np.pi / 180, threshold=max(15, roi_h // 6),
                                minLineLength=max(10, roi_h // 8), maxLineGap=max(3, roi_h // 20))
        segs: list[tuple[int, int, int, int]] = []
        total_len = 0.0
        if lines is not None:
            for xa, ya, xb, yb in lines.reshape(-1, 4):
                ang = np.degrees(np.arctan2(abs(yb - ya), abs(xb - xa)))
                # 선분 위 화소의 대부분이 후보여야 한다 (거친 노면 질감의 흩어진 점이 이어진 가짜 선 제거)
                if ang >= self.min_angle_deg and _coverage(cand, xa, ya, xb, yb) >= self.min_coverage:
                    segs.append((int(xa), int(ya), int(xb), int(yb)))
                    total_len += float(np.hypot(xb - xa, yb - ya))

        if not segs:
            return LaneWearResult(**_NOT_FOUND, lane_pixel_ratio=0.0, n_segments=0), []

        near = np.zeros_like(cand)
        band = max(3, k // 4)  # 선분 주변 띠 폭: 차선 두께를 덮을 만큼
        for xa, ya, xb, yb in segs:
            cv2.line(near, (xa, ya), (xb, yb), 255, band)
        lane = (near > 0) & (cand > 0)
        n_lane = int(np.count_nonzero(lane))
        # 노면: 차선 후보와 그 주변을 넓게 뺀 나머지 ROI
        excl = cv2.dilate(cv2.bitwise_or(cand, near), cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
        road = roi_mask & (excl == 0)

        segs_px = [(round((xa + x0) / scale), round((ya + y0) / scale), round((xb + x0) / scale),
                    round((yb + y0) / scale)) for xa, ya, xb, yb in segs]
        ratio = n_lane / max(1, n_roi)
        if n_lane < self.min_lane_pixels or np.count_nonzero(road) < self.min_lane_pixels:
            return LaneWearResult(**_NOT_FOUND, lane_pixel_ratio=ratio, n_segments=len(segs)), segs_px

        lane_l = float(np.median(raw_l[lane]))
        road_l = float(np.median(raw_l[road]))
        contrast = (lane_l - road_l) / max(road_l, 1.0)
        is_worn = contrast < self.threshold

        # 신뢰도 = (선 구조가 충분한가) × (기준값에서 얼마나 떨어져 있나). 기준값 바로 위아래면 0.5×support.
        support = min(1.0, total_len / max(1.0, 1.5 * roi_h))
        certainty = min(1.0, abs(contrast - self.threshold) / self.threshold)
        confidence = support * (0.5 + 0.5 * certainty)
        return LaneWearResult(
            lane_found=True, contrast=contrast, is_worn=is_worn, confidence=float(np.clip(confidence, 0, 1)),
            lane_pixel_ratio=ratio, n_segments=len(segs),
        ), segs_px

    # ---- 내부 ----------------------------------------------------------------------

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
