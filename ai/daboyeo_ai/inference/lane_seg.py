"""차선 영역 분할 ONNX (YOLO-seg, 2클래스 lane·marking) → 프레임 크기 마스크 (handoff 23장, 실험).

    seg = LaneSegmenter("ai/models/lane_seg_v1.onnx", threads=1)
    mask = seg.lane_mask(frame_bgr)          # uint8 H×W, 차선 255 / 나머지 0
    r = LaneWearMeter(lane_mask_min=0.5).measure(frame_bgr, lane_mask=mask)

Ultralytics YOLO(v8/11) `-seg` 모델을 `format="onnx"` 로 내보낸 기본 출력을 가정한다:
`output0 (1, 4+nc+nm, N)` (상자·클래스 점수·마스크 계수), `output1 (1, nm, mh, mw)` (프로토타입).
전처리는 탐지기와 같은 letterbox, 마스크는 계수 × 프로토타입 → sigmoid → 상자 밖 0 → 원본 크기 → 0.5 이상.
Ultralytics 는 인스턴스마다 원본 크기로 키운 뒤 자르지만, 여기서는 같은 클래스 인스턴스를 프로토타입 크기에서 먼저
합친다 (클래스별 합집합만 필요하고 더 빠르다). 경계 1~2 픽셀이 다를 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .onnx_detector import Letterbox, letterbox

SEG_CLASS_NAMES = ("lane", "marking")  # daboyeo_ai.data.lane_seg 와 같은 순서


def decode_seg(
    out0: np.ndarray, protos: np.ndarray, lb: Letterbox, size: int, orig_w: int, orig_h: int,
    nc: int, conf: float = 0.25, iou: float = 0.5, classes: tuple[int, ...] = (0,),
) -> dict[int, np.ndarray]:
    """YOLO-seg 출력 → {class_id: uint8 마스크 (orig_h×orig_w, 0/255)}. 클래스별 NMS 뒤 인스턴스 합집합."""
    pred = out0[0].T  # (N, 4+nc+nm)
    scores_all = pred[:, 4:4 + nc]
    cls = scores_all.argmax(1)
    score = scores_all[np.arange(len(pred)), cls]
    keep = (score >= conf) & np.isin(cls, classes)
    nm, mh, mw = protos.shape[1:]
    out = {c: np.zeros((orig_h, orig_w), np.uint8) for c in classes}
    if not keep.any():
        return out
    pred, cls, score = pred[keep], cls[keep], score[keep]
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2  # letterbox 입력 픽셀
    rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)]
    idx = np.asarray(cv2.dnn.NMSBoxesBatched(rects, score.astype(float).tolist(), cls.astype(int).tolist(),
                                              float(conf), iou)).reshape(-1)
    if len(idx) == 0:
        return out
    coef = pred[idx, 4 + nc:4 + nc + nm]
    m = coef @ protos[0].reshape(nm, -1)
    m = (1.0 / (1.0 + np.exp(-m))).reshape(-1, mh, mw)
    sx, sy = mw / size, mh / size
    ys, xs = np.arange(mh)[:, None] + 0.5, np.arange(mw)[None, :] + 0.5
    # letterbox 안 실제 이미지 영역 (프로토타입 좌표)
    left, top = lb.pad_x * sx, lb.pad_y * sy
    right = (lb.pad_x + orig_w * lb.ratio) * sx
    bottom = (lb.pad_y + orig_h * lb.ratio) * sy
    for c in classes:
        sel = [k for k, i in enumerate(idx) if cls[i] == c]
        if not sel:
            continue
        acc = np.zeros((mh, mw), np.float32)
        for k in sel:
            i = idx[k]
            inside = (xs >= x1[i] * sx) & (xs < x2[i] * sx) & (ys >= y1[i] * sy) & (ys < y2[i] * sy)
            np.maximum(acc, np.where(inside, m[k], 0.0), out=acc)
        crop = acc[int(round(top)):int(round(bottom)), int(round(left)):int(round(right))]
        if crop.size == 0:
            continue
        full = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        out[c] = np.where(full > 0.5, 255, 0).astype(np.uint8)
    return out


class LaneSegmenter:
    """차선 분할 ONNX. threads 로 CPU 코어 사용을 제한한다 (탐지기와 같은 방식)."""

    def __init__(self, model_path: str | Path, threads: int = 1, conf: float = 0.25, iou: float = 0.5):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        outs = self.session.get_outputs()
        if len(outs) < 2:
            raise ValueError(f"{model_path}: 분할 모델이 아니다 (출력 {len(outs)}개, YOLO-seg 는 2개)")
        self.nm = int(outs[1].shape[1])
        self.nc = int(outs[0].shape[1]) - 4 - self.nm
        if self.nc != len(SEG_CLASS_NAMES):
            raise ValueError(f"model has {self.nc} classes, expected {len(SEG_CLASS_NAMES)} {SEG_CLASS_NAMES}")
        self.conf, self.iou = conf, iou

    def masks(self, frame_bgr: np.ndarray, classes: tuple[int, ...] = (0, 1)) -> dict[str, np.ndarray]:
        h, w = frame_bgr.shape[:2]
        img, lb = letterbox(frame_bgr, self.size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255.0
        out0, protos = self.session.run(None, {self.input_name: x})[:2]
        res = decode_seg(out0, protos, lb, self.size, w, h, self.nc, self.conf, self.iou, classes)
        return {SEG_CLASS_NAMES[c]: m for c, m in res.items()}

    def lane_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """uint8 H×W, 차선 255."""
        return self.masks(frame_bgr, classes=(0,))["lane"]
