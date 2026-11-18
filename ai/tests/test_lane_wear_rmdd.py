"""RMDD 차선 마모 평가 (handoff 24장) — 네트워크·실데이터 없이 합성 사진으로."""

import json

import cv2
import numpy as np
import pytest
from daboyeo_ai.eval import lane_wear_rmdd as rm
from daboyeo_ai.lane_wear import LaneWearMeter

W, H = 640, 360


def road(lane_v: int, seed: int = 0) -> np.ndarray:
    """하늘 + 노면(80) + 원근 차선 두 줄 (왼쪽 밝기 lane_v, 오른쪽 200)."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 80, np.uint8)
    img[: H // 3] = (200, 170, 140)
    img = np.clip(img + rng.normal(0, 3, (H, W, 1)), 0, 255).astype(np.uint8)
    cv2.line(img, (130, 345), (295, 220), (lane_v,) * 3, 6)
    cv2.line(img, (510, 345), (345, 220), (200,) * 3, 6)
    return img


LEFT = rm.Box("LD2", (0.15, 0.55, 0.5, 1.0))
RIGHT = rm.Box("LD0", (0.5, 0.55, 0.85, 1.0))
SKY = rm.Box("LD1", (0.0, 0.0, 1.0, 0.25))


def test_parse_label_box_and_polygon():
    boxes = rm.parse_label("8 0.5 0.5 0.2 0.4\n3 0.1 0.2 0.3 0.2 0.3 0.6\n")
    assert boxes[0].cls == "LD2" and boxes[0].group == "LD" and boxes[0].grade == 2
    assert boxes[0].xyxy == pytest.approx((0.4, 0.3, 0.6, 0.7))
    assert boxes[1].cls == "BD0" and boxes[1].xyxy == pytest.approx((0.1, 0.2, 0.3, 0.6))
    assert rm.parse_label("") == []


def test_unstretch_restores_aspect():
    sq = np.zeros((640, 640, 3), np.uint8)
    assert rm.unstretch(sq, 16 / 9).shape == (360, 640, 3)
    assert rm.unstretch(sq, 0).shape == (640, 640, 3)


def test_measure_image_assigns_segments_to_boxes():
    frame = road(lane_v=110)  # 왼쪽은 흐린 차선, 오른쪽은 선명
    fr, (left, right, sky) = rm.measure_image(LaneWearMeter(), frame, [LEFT, RIGHT, SKY])
    assert fr["found"]
    assert left["found"] and right["found"]
    assert left["contrast"] < right["contrast"]
    assert not sky["found"] and sky["contrast"] is None and not sky["in_roi"]
    assert left["grade"] == 2 and right["grade"] == 0


def test_load_items_counts_and_order(tmp_path):
    for split, name, text in [("train", "20_jpg.rf.a.jpg", "8 0.5 0.5 0.2 0.2\n6 0.2 0.5 0.1 0.1\n"),
                              ("valid", "3_jpg.rf.b.jpg", "6 0.5 0.5 0.2 0.2\n6 0.6 0.5 0.2 0.2\n"),
                              ("test", "100_jpg.rf.c.jpg", "")]:
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(tmp_path / split / "images" / name), road(200))
        (tmp_path / split / "labels" / (name[:-4] + ".txt")).write_text(text)
    items = rm.load_items(tmp_path)
    assert [it.num for it in items] == [3, 20, 100]
    c = rm.class_counts(items)
    assert c["all"]["LD0"] == {"boxes": 3, "images": 2}
    assert c["train"]["LD2"] == {"boxes": 1, "images": 1}
    assert c["test"]["LD0"]["boxes"] == 0


def test_bootstrap_auc_perfect_and_random():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    w = np.array([True, True, False, False])
    auc, lo, hi = rm.bootstrap_auc(s, w, np.arange(4), reps=50)
    assert auc == 1.0 and lo <= 1.0 and hi == 1.0
    assert np.isnan(rm.bootstrap_auc(s, np.zeros(4, bool), np.arange(4))[0])


def test_operating_points_threshold_from_train_only():
    recs = [{"split": "train", "grade": 0, "m": v} for v in np.linspace(1, 2, 11)]
    recs += [{"split": "test", "grade": 0, "m": 1.5}, {"split": "test", "grade": 2, "m": 0.5},
             {"split": "test", "grade": 2, "m": None}, {"split": "train", "grade": 2, "m": 0.0}]
    (row,) = rm.operating_points(recs, "m", false_rates=(0.1,))
    assert row["threshold"] == pytest.approx(1.1)
    assert row["2"] == {"flagged": 1, "n_found": 1, "n_all": 2}
    assert row["0"] == {"flagged": 0, "n_found": 1, "n_all": 1}


def test_main_end_to_end(tmp_path):
    root = tmp_path / "rmdd"
    for split in rm.SPLITS:
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
    # 왼쪽 차선이 흐릴수록 LD2, 오른쪽은 늘 선명한 LD0
    for k in range(30):
        split = "train" if k < 20 else "test"
        worn = k % 2 == 0
        name = f"{k}_jpg.rf.x.jpg"
        frame = cv2.resize(road(lane_v=105 if worn else 190, seed=k), (640, 640))  # Roboflow 처럼 늘린 사진
        cv2.imwrite(str(root / split / "images" / name), frame)
        cls_left = 8 if worn else 6
        (root / split / "labels" / f"{k}_jpg.rf.x.txt").write_text(
            f"{cls_left} 0.325 0.775 0.35 0.45\n6 0.675 0.775 0.35 0.45\n")
    out = tmp_path / "r.json"
    assert rm.main(["--root", str(root), "--json", str(out), "--csv-dir", str(tmp_path / "csv"), "--reps", "20"]) == 0
    d = json.loads(out.read_text())
    ld = d["variants"]["base"]["boxes"]["LD"]
    assert ld["per_grade"]["2"]["n_boxes"] == 15
    assert ld["auc"]["contrast"]["0_vs_2"]["auc"] > 0.9
    assert d["class_counts"]["all"]["LD0"]["boxes"] == 45
    assert (tmp_path / "csv" / "base_boxes.csv").exists()
