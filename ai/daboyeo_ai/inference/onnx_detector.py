"""ONNX 탐지기 — `daboyeo_contracts.Detector` 구현 (박겸 → 정현우).

가상 블랙박스는 이렇게만 쓴다:

    det = OnnxDetector("ai/models/detector_v1.onnx", threads=1)
    detections = det.detect(frame_bgr)   # list[Detection], bbox 는 0~1 정규화

전처리(letterbox)와 후처리(NMS)는 여기서 끝낸다. Ultralytics YOLO(v8/11) detect 모델을
`format="onnx"` 로 내보낸 기본 출력 `(1, 4 + nc, N)` 형식을 가정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES, Detection


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
    output: np.ndarray, lb: Letterbox, orig_w: int, orig_h: int, conf: float = 0.25, iou: float = 0.45,
) -> list[tuple[int, float, float, float, float, float]]:
    """YOLO 출력 `(1, 4+nc, N)` → [(cls, score, x1, y1, x2, y2)] (원본 이미지 픽셀 좌표), 클래스별 NMS."""
    pred = output[0].T  # (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(pred)), cls]
    keep = score >= conf
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
    idx = cv2.dnn.NMSBoxesBatched(rects, score.astype(float).tolist(), cls.astype(int).tolist(), conf, iou)
    out = []
    for i in np.asarray(idx).reshape(-1):
        if x2[i] > x1[i] and y2[i] > y1[i]:
            out.append((int(cls[i]), float(score[i]), float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])))
    out.sort(key=lambda t: -t[1])
    return out


class OnnxDetector:
    """`Detector` 프로토콜 구현. threads 로 CPU 코어 사용을 제한한다 (PRD FR-1.6, 1~2코어)."""

    def __init__(self, model_path: str | Path, threads: int = 1, conf: float = 0.25, iou: float = 0.45):
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
        self.conf, self.iou = conf, iou

    def raw(self, frame_bgr: np.ndarray) -> list[tuple[int, float, float, float, float, float]]:
        h, w = frame_bgr.shape[:2]
        img, lb = letterbox(frame_bgr, self.size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        return decode_yolo(out, lb, w, h, self.conf, self.iou)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        return [
            Detection.from_class_id(c, conf=min(s, 1.0), bbox=(x1 / w, y1 / h, x2 / w, y2 / h))
            for c, s, x1, y1, x2, y2 in self.raw(frame_bgr)
        ]
