"""ONNX 탐지기 — `daboyeo_contracts.Detector` 구현 (박겸 → 정현우).

가상 블랙박스는 이렇게만 쓴다:

    det = OnnxDetector("ai/models/detector_v1.onnx", threads=1)
    detections = det.detect(frame_bgr)   # list[Detection], bbox 는 0~1 정규화

    # 클래스별 신뢰도 기준 (오픈 이슈 #3 초안, 정현우 확인 필요): 없는 클래스는 0.25
    det = OnnxDetector("ai/models/detector_v2.onnx", conf={"pothole": 0.3, "crack": 0.25, "guardrail": 0.4, "sign": 0.45})
    det = OnnxDetector("ai/models/detector_v2.onnx", conf=load_class_conf("experiments/conf_thresholds.json"))

    # (선택) 도로 영역을 좌우 2조각으로 잘라 한 번 더 본다 — 멀리 있는 작은 포트홀·균열용, 기본은 꺼짐 (handoff 14장)
    det = OnnxDetector("ai/models/detector_v2.onnx", windows=road_tiles(cols=2))

    # 시설물 파손 분류도 켜려면 (계획 2.7): guardrail·sign 탐지에 damage / damage_conf 가 채워진다
    det = OnnxDetector("ai/models/detector_v2.onnx", threads=1, damage_model="ai/models/damage_cls_v1.onnx")

전처리(letterbox)와 후처리(NMS)는 여기서 끝낸다. Ultralytics YOLO(v8/11) detect 모델을
`format="onnx"` 로 내보낸 기본 출력 `(1, 4 + nc, N)` 형식을 가정한다.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES, Detection

if TYPE_CHECKING:
    from .damage_classifier import DamageClassifier

FACILITY_CLASSES = ("guardrail", "sign")  # 파손 분류 대상 (계약: Detection.damage 는 이 둘만)
DEFAULT_CONF = 0.25


def class_conf_list(conf: float | Mapping[str, float], default: float = DEFAULT_CONF) -> list[float]:
    """`0.3` 또는 `{"pothole": 0.3, ...}` → CLASS_NAMES 순서의 클래스별 기준. 빠진 클래스는 default."""
    if not isinstance(conf, Mapping):
        return [float(conf)] * len(CLASS_NAMES)
    unknown = set(conf) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"unknown classes in conf: {sorted(unknown)} (expected {CLASS_NAMES})")
    return [float(conf.get(n, default)) for n in CLASS_NAMES]


def load_class_conf(path: str | Path) -> dict[str, float]:
    """`conf_thresholds.json`(`class_conf` 키) 또는 모델 카드(`class_conf_draft` 키) → {클래스: 기준}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("class_conf", "class_conf_draft"):
        if key in data:
            return {k: float(x) for k, x in data[key].items()}
    raise KeyError(f"{path}: no 'class_conf' or 'class_conf_draft' key")


Window = tuple[float, float, float, float]  # (x1, y1, x2, y2) 프레임 크기로 나눈 0~1 — 프레임의 어느 부분을 모델에 넣을지
FULL_FRAME: Window = (0.0, 0.0, 1.0, 1.0)


def road_tiles(cols: int = 2, top: float = 0.45, bottom: float = 1.0, overlap: float = 0.1,
               full_frame: bool = True) -> tuple[Window, ...]:
    """도로 영역(높이 top~bottom, 가로 전체)을 가로로 cols 조각(조각끼리 overlap 만큼 겹침)으로 나눈 창 목록.

    full_frame=True 면 맨 앞에 프레임 전체(지금과 같은 640 한 번)를 넣는다 — 표지판처럼 지평선 위에 있는 것을 놓치지 않게.
    1920×1080 에서 cols=2 면 조각 하나가 약 1056×594 → 640 으로 줄이는 비율이 0.61 (전체 프레임은 0.33) 이라
    도로 위 물체가 약 1.8배 크게 들어간다. 모델을 cols(+1) 번 돌리므로 시간도 그만큼 든다.
    """
    if cols < 1 or not 0 <= top < bottom <= 1 or not 0 <= overlap < 1:
        raise ValueError(f"bad road_tiles(cols={cols}, top={top}, bottom={bottom}, overlap={overlap})")
    w = 1.0 / (cols - (cols - 1) * overlap)  # 조각 너비: cols 개가 overlap 비율씩 겹쳐서 정확히 0~1 을 덮는다
    step = w * (1 - overlap)
    tiles = tuple((round(i * step, 6), top, round(min(1.0, i * step + w), 6), bottom) for i in range(cols))
    return ((FULL_FRAME,) if full_frame else ()) + tiles


def window_pixels(win: Window, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = win
    return round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)


def merge_nms(dets: Sequence[tuple[int, float, float, float, float, float]], iou: float = 0.45
              ) -> list[tuple[int, float, float, float, float, float]]:
    """여러 창에서 나온 [(cls, score, x1, y1, x2, y2)] (프레임 픽셀) 를 클래스별 NMS 로 합친다 (점수 높은 순)."""
    if len(dets) <= 1:
        return list(dets)
    rects = [[d[2], d[3], d[4] - d[2], d[5] - d[3]] for d in dets]
    idx = cv2.dnn.NMSBoxesBatched(rects, [d[1] for d in dets], [d[0] for d in dets], 0.0, iou)
    return sorted((dets[i] for i in np.asarray(idx).reshape(-1)), key=lambda t: -t[1])


@dataclass(frozen=True)
class Letterbox:
    ratio: float
    pad_x: float
    pad_y: float


def letterbox(img_bgr: np.ndarray, size: int) -> tuple[np.ndarray, Letterbox]:
    """비율을 유지해 size×size 로 맞추고 남는 곳은 회색(114)으로 채운다 (Ultralytics 와 같음)."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    px, py = (size - nw) / 2, (size - nh) / 2
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img_bgr
    top, left = round(py - 0.1), round(px - 0.1)
    out = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, Letterbox(r, left, top)


def decode_yolo(
    output: np.ndarray, lb: Letterbox, orig_w: int, orig_h: int,
    conf: float | Sequence[float] = DEFAULT_CONF, iou: float = 0.45,
) -> list[tuple[int, float, float, float, float, float]]:
    """YOLO 출력 `(1, 4+nc, N)` → [(cls, score, x1, y1, x2, y2)] (원본 이미지 픽셀 좌표), 클래스별 NMS.

    conf 는 숫자 하나 또는 class_id 순서의 클래스별 기준. NMS 가 클래스별이라 NMS 전에 클래스별로 거르든
    낮은 기준으로 NMS 한 뒤 거르든 결과가 같다 (박스를 지우는 박스는 항상 점수가 더 높으므로).
    """
    pred = output[0].T  # (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(pred)), cls]
    thr = np.full(scores_all.shape[1], float(conf)) if np.isscalar(conf) else np.asarray(conf, float)
    if len(thr) != scores_all.shape[1]:
        raise ValueError(f"{len(thr)} class thresholds for {scores_all.shape[1]} classes")
    keep = score >= thr[cls]
    if not keep.any():
        return []
    pred, cls, score = pred[keep], cls[keep], score[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - bw / 2 - lb.pad_x) / lb.ratio
    y1 = (cy - bh / 2 - lb.pad_y) / lb.ratio
    x2 = (cx + bw / 2 - lb.pad_x) / lb.ratio
    y2 = (cy + bh / 2 - lb.pad_y) / lb.ratio
    x1, x2 = np.clip(x1, 0, orig_w), np.clip(x2, 0, orig_w)
    y1, y2 = np.clip(y1, 0, orig_h), np.clip(y2, 0, orig_h)
    rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    idx = cv2.dnn.NMSBoxesBatched(rects, score.astype(float).tolist(), cls.astype(int).tolist(),
                                   float(thr.min()), iou)
    out = []
    for i in np.asarray(idx).reshape(-1):
        if x2[i] > x1[i] and y2[i] > y1[i]:
            out.append((int(cls[i]), float(score[i]), float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])))
    out.sort(key=lambda t: -t[1])
    return out


class OnnxDetector:
    """`Detector` 프로토콜 구현. threads 로 CPU 코어 사용을 제한한다 (PRD FR-1.6, 1~2코어).

    conf: 숫자 하나(모든 클래스) 또는 `{"pothole": 0.3, ...}` 클래스별 기준 (빠진 클래스는 0.25).
    windows: 프레임의 어느 부분들을 모델에 넣을지 (0~1 비율 창 목록, 예: `road_tiles(cols=2)`). 기본 None = 프레임 전체 한 번
    (지금까지와 같음). 창마다 640 letterbox 로 돌리고 결과를 프레임 좌표로 옮겨 클래스별 NMS 로 합친다 → 창 수만큼 느려진다.
    `self.class_conf` 는 class_id 순서 리스트, `self.conf` 는 그 최솟값 (예전 코드 호환).
    """

    def __init__(
        self,
        model_path: str | Path,
        threads: int = 1,
        conf: float | Mapping[str, float] = DEFAULT_CONF,
        iou: float = 0.45,
        damage_model: str | Path | DamageClassifier | None = None,
        windows: Sequence[Window] | None = None,
    ):
        import onnxruntime as ort  # 학습 환경 등 추론이 필요 없는 곳에서는 import 하지 않도록

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        nc = int(self.session.get_outputs()[0].shape[1]) - 4
        if nc != len(CLASS_NAMES):
            raise ValueError(f"model has {nc} classes, contract expects {len(CLASS_NAMES)} {CLASS_NAMES}")
        self.class_conf = class_conf_list(conf)
        self.conf, self.iou = min(self.class_conf), iou
        self.windows = tuple(tuple(float(v) for v in win) for win in windows) if windows else None
        for x1, y1, x2, y2 in self.windows or ():
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"window must be 0~1 (x1<x2, y1<y2): {(x1, y1, x2, y2)}")
        # 파손 분류기: 경로면 같은 threads 로 연다. 탐지 → 분류가 차례로 돌아서 코어를 더 쓰지 않는다
        if damage_model is None or not isinstance(damage_model, (str, Path)):
            self.damage = damage_model
        else:
            from .damage_classifier import DamageClassifier

            self.damage = DamageClassifier(damage_model, threads=threads)

    def raw(self, frame_bgr: np.ndarray) -> list[tuple[int, float, float, float, float, float]]:
        if not self.windows:
            return self._run(frame_bgr)
        h, w = frame_bgr.shape[:2]
        dets = []
        for win in self.windows:
            x1, y1, x2, y2 = window_pixels(win, w, h)
            for c, s, a, b, cc, d in self._run(frame_bgr[y1:y2, x1:x2]):
                dets.append((c, s, a + x1, b + y1, cc + x1, d + y1))
        return merge_nms(dets, self.iou)

    def _run(self, frame_bgr: np.ndarray) -> list[tuple[int, float, float, float, float, float]]:
        h, w = frame_bgr.shape[:2]
        img, lb = letterbox(frame_bgr, self.size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        return decode_yolo(out, lb, w, h, self.class_conf, self.iou)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        out = []
        for c, s, x1, y1, x2, y2 in self.raw(frame_bgr):
            extra = {}
            if self.damage is not None and CLASS_NAMES[c] in FACILITY_CLASSES:
                p = self.damage.prob_damaged(frame_bgr, (x1, y1, x2, y2))
                if p is not None:  # 너무 작은 박스는 분류하지 않는다 (학습 크롭에도 없었음) → None 그대로
                    extra = {"damage": self.damage.label(p), "damage_conf": p}
            out.append(Detection.from_class_id(c, conf=min(s, 1.0), bbox=(x1 / w, y1 / h, x2 / w, y2 / h), **extra))
        return out
