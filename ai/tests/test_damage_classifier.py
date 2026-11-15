"""DamageClassifier / OnnxDetector(damage_model=...) — onnxruntime 세션을 가짜로 바꿔 시험한다.

실제 ONNX 로 같은 경로를 도는 것은 노트북 03 의 일치 확인과 로컬 스모크 실행에서 본다.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from daboyeo_ai.inference import DamageClassifier, OnnxDetector

from test_onnx_detector import fake_output


class FakeSession:
    """경로 이름으로 탐지기/분류기를 흉내 낸다. 분류기는 입력 평균 밝기로 P(damaged) 를 정한다."""

    calls: list = []

    def __init__(self, path, so=None, providers=None):
        self.path = str(path)
        self.so = so
        self.kind = "cls" if Path(self.path).name.startswith("damage") else "det"
        FakeSession.calls.append(self)
        self.inputs_seen = []

    def get_inputs(self):
        size = 224 if self.kind == "cls" else 640
        return [SimpleNamespace(name="images", shape=[1, 3, size, size])]

    def get_outputs(self):
        return [SimpleNamespace(shape=[1, 2] if self.kind == "cls" else [1, 8, 8400])]

    def run(self, _, feeds):
        x = feeds["images"]
        self.inputs_seen.append(x)
        if self.kind == "det":
            # 640 입력 = 1280×720 의 0.5배, pad_y 140. guardrail(2) · pothole(0) · 아주 작은 sign(3)
            return [fake_output([
                (2, 0.9, 200, 300, 200, 60),
                (0, 0.8, 500, 400, 80, 80),
                (3, 0.7, 600, 200, 6, 6),
            ])]
        p_dmg = float(x.mean())  # 0~1
        order = self._order
        probs = [p_dmg, 1 - p_dmg] if order[0] == "damaged" else [1 - p_dmg, p_dmg]
        return [np.array([probs], np.float32)]


@pytest.fixture
def fake_ort(monkeypatch):
    FakeSession.calls = []
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    return FakeSession


def write_card(tmp_path, classes, threshold=0.5, name="damage_cls_v1"):
    onnx = tmp_path / f"{name}.onnx"
    onnx.write_bytes(b"fake")
    (tmp_path / f"{name}.json").write_text(json.dumps({"classes": classes, "threshold": threshold}))
    FakeSession._order = classes
    return onnx


@pytest.mark.parametrize("classes", [["damaged", "normal"], ["normal", "damaged"]])
def test_reads_class_order_from_card(tmp_path, fake_ort, classes):
    clf = DamageClassifier(write_card(tmp_path, classes, threshold=0.6), threads=2)
    assert clf.classes == classes and clf.threshold == 0.6
    assert fake_ort.calls[-1].so.intra_op_num_threads == 2
    bright = np.full((60, 200, 3), 255, np.uint8)  # 정사각형으로 패딩(114)돼도 평균 > 0.6
    p = clf.prob_damaged_crop(bright)
    assert 0.6 < p < 1.0 and clf.label(p) == "suspected"
    assert clf.label(clf.prob_damaged_crop(np.zeros((60, 200, 3), np.uint8))) == "normal"
    x = fake_ort.calls[-1].inputs_seen[-1]
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32 and x.max() <= 1.0


def test_card_without_damaged_class_is_rejected(tmp_path, fake_ort):
    with pytest.raises(ValueError):
        DamageClassifier(write_card(tmp_path, ["ok", "bad"]))


def test_prob_damaged_uses_training_crop_and_skips_small(tmp_path, fake_ort):
    clf = DamageClassifier(write_card(tmp_path, ["damaged", "normal"]), threshold=0.3)
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[100:200, 100:500] = 255
    assert clf.prob_damaged(frame, (100, 100, 500, 200)) > 0.3
    assert clf.prob_damaged(frame, (0, 0, 10, 100)) is None  # 짧은 변 < 24 px


def test_detector_fills_damage_only_for_facilities(tmp_path, fake_ort):
    write_card(tmp_path, ["damaged", "normal"], threshold=0.0)  # 무엇이든 suspected
    det = OnnxDetector(tmp_path / "det.onnx", threads=1, damage_model=tmp_path / "damage_cls_v1.onnx")
    assert all(c.so.intra_op_num_threads == 1 for c in fake_ort.calls)
    out = {d.class_name: d for d in det.detect(np.full((720, 1280, 3), 90, np.uint8))}
    assert out["guardrail"].damage == "suspected" and 0 <= out["guardrail"].damage_conf <= 1
    assert out["pothole"].damage is None and out["pothole"].damage_conf is None
    assert out["sign"].damage is None  # 12 px 짜리 박스 → 분류 안 함


def test_detector_without_damage_model_is_unchanged(tmp_path, fake_ort):
    det = OnnxDetector(tmp_path / "det.onnx")
    assert det.damage is None
    assert all(d.damage is None and d.damage_conf is None for d in det.detect(np.zeros((720, 1280, 3), np.uint8)))
    assert len(fake_ort.calls) == 1


def test_detector_per_class_conf(tmp_path, fake_ort):
    """가짜 탐지 출력: guardrail 0.9, pothole 0.8, sign 0.7 → pothole 기준만 0.85 로 올리면 pothole 만 빠진다."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert OnnxDetector(tmp_path / "det.onnx").conf == 0.25  # 기본값은 그대로
    det = OnnxDetector(tmp_path / "det.onnx", conf={"pothole": 0.85, "guardrail": 0.3})
    assert det.class_conf == [0.85, 0.25, 0.3, 0.25] and det.conf == 0.25
    assert sorted(d.class_name for d in det.detect(frame)) == ["guardrail", "sign"]
    det = OnnxDetector(tmp_path / "det.onnx", conf=0.75)
    assert sorted(d.class_name for d in det.detect(frame)) == ["guardrail", "pothole"]
