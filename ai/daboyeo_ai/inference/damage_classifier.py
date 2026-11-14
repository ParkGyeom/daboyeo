"""시설물(가드레일·표지판) 파손 분류기 — ONNX Runtime (PRD FR-1.2, 계획 2.7).

    clf = DamageClassifier("ai/models/damage_cls_v1.onnx", threads=1)   # 옆의 damage_cls_v1.json 을 같이 읽는다
    p = clf.prob_damaged(frame_bgr, (x1, y1, x2, y2))                   # 원본 픽셀 좌표, 너무 작으면 None

보통은 직접 부르지 않고 `OnnxDetector(..., damage_model=...)` 가 guardrail/sign 탐지에 채워 준다.
결과는 '파손 의심'(suspected)일 뿐 확정이 아니다. 신고 제안은 여러 차량이 같은 판단을 낼 때만 한다 (FR-7.2).

크롭은 학습 데이터와 똑같이 만든다: `facility_crop` (CropWriter 와 같은 함수, pad 0.1 · 긴 변 224 이하) →
`square_input` (비율 유지 정사각형) → RGB 0~1. 클래스 순서·임계값은 모델 카드(.json)에서 읽는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.damage_cls import square_input
from ..data.yolo import Box, facility_crop

DEFAULT_CROP = {"pad": 0.1, "resize_max": 224, "min_side_px": 24}


class DamageClassifier:
    def __init__(
        self,
        model_path: str | Path,
        card_path: str | Path | None = None,
        threads: int = 1,
        threshold: float | None = None,
    ):
        import onnxruntime as ort

        model_path = Path(model_path)
        card_path = Path(card_path) if card_path is not None else model_path.with_suffix(".json")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        self.classes: list[str] = list(card["classes"])
        if "damaged" not in self.classes:
            raise ValueError(f"card classes {self.classes} has no 'damaged'")
        self.damaged_index = self.classes.index("damaged")
        self.threshold = float(threshold if threshold is not None else card["threshold"])
        crop = {**DEFAULT_CROP, **card.get("crop", {})}
        self.pad = float(crop["pad"])
        self.resize_max = int(crop["resize_max"])
        self.min_side_px = int(crop["min_side_px"])
        self.card = card

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        nc = self.session.get_outputs()[0].shape[1]
        if isinstance(nc, int) and nc != len(self.classes):
            raise ValueError(f"model has {nc} outputs, card lists {len(self.classes)} classes {self.classes}")

    def crop(self, frame_bgr: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> np.ndarray | None:
        """원본 프레임 + 픽셀 박스 → 학습과 같은 크롭 (짧은 변이 min_side_px 미만이면 None)."""
        x1, y1, x2, y2 = box_xyxy
        return facility_crop(frame_bgr, Box(0, x1, y1, x2, y2), self.pad, self.resize_max, self.min_side_px)

    def prob_damaged_crop(self, crop_bgr: np.ndarray) -> float:
        """`crop()` 결과(또는 학습 폴더의 크롭 jpg) → P(damaged)."""
        img = square_input(crop_bgr, self.size)
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0].astype(np.float64)
        if not np.isclose(out.sum(), 1.0, atol=1e-3) or (out < 0).any():  # 확률이 아니면 softmax
            out = np.exp(out - out.max())
            out /= out.sum()
        return float(np.clip(out[self.damaged_index], 0.0, 1.0))

    def prob_damaged(self, frame_bgr: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> float | None:
        c = self.crop(frame_bgr, box_xyxy)
        return None if c is None else self.prob_damaged_crop(c)

    def label(self, p: float) -> str:
        return "suspected" if p >= self.threshold else "normal"
