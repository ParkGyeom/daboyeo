"""차선 마모 발표 자료 (handoff 24-8) — 합성 영상·라벨로, 네트워크 없이."""

import csv
import json

import cv2
import numpy as np
from daboyeo_ai.eval import lane_wear_figs as lf
from daboyeo_ai.eval import lane_wear_rmdd as rm

from test_lane_wear_rmdd import road


def test_tier_color_bands():
    assert lf.tier_color(1.2) == lf.GREEN
    assert lf.tier_color(0.8) == lf.GREEN
    assert lf.tier_color(0.6) == lf.YELLOW
    assert lf.tier_color(0.2) == lf.RED


def test_put_text_draws_something():
    img = np.zeros((60, 300, 3), np.uint8)
    lf.put_text(img, (5, 5), "차선 대비 1.0", 20)
    assert img.any()


def test_pick_examples_near_median_one_per_image_and_skips_small():
    rows = [{"cls": "LD2", "split": "test", "img": str(k), "contrast": str(c), "box": json.dumps(b)}
            for k, c, b in [(1, 0.1, [0, 0, .5, .5]), (2, 0.5, [0, 0, .5, .5]), (2, 0.52, [0, 0, .5, .5]),
                            (3, 0.55, [0, 0, .5, .5]), (4, 0.5, [0, 0, .1, .1]), (5, 0.9, [0, 0, .5, .5])]]
    rows.append({"cls": "LD2", "split": "train", "img": "6", "contrast": "0.5", "box": json.dumps([0, 0, .5, .5])})
    picked = lf.pick_examples(rows, "LD2", n=2)
    assert [r["img"] for r in picked] == ["2", "3"]  # 중앙값(0.5)에 가까운 순, 같은 사진 한 번, 작은 박스·train 제외


def test_render_video_writes_file(tmp_path):
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 360))
    for k in range(20):
        w.write(road(lane_v=200 if k < 10 else 110, seed=k))
    w.release()
    out = tmp_path / "o.mp4"
    res = lf.render_video(video, 0.0, 1.9, out, width=640, graph_h=120)
    assert out.exists() and out.stat().st_size > 0
    assert res["frames"] == 20 and res["found"] >= 15
    assert res["very_faint"] + res["faint"] > 0  # 뒤쪽 흐린 차선이 기준(전체 중앙값)보다 낮게 나온다


def test_render_rmdd_sheet(tmp_path):
    root = tmp_path / "rmdd"
    rows = []
    for k, (cls, lane_v) in enumerate([("LD2", 110), ("LD2", 115), ("LD0", 200), ("LD0", 190)]):
        (root / "test" / "images").mkdir(parents=True, exist_ok=True)
        (root / "test" / "labels").mkdir(parents=True, exist_ok=True)
        name = f"{k}_jpg.rf.x"
        cv2.imwrite(str(root / "test" / "images" / f"{name}.jpg"), cv2.resize(road(lane_v, seed=k), (640, 640)))
        cid = rm.NAMES.index(cls)
        (root / "test" / "labels" / f"{name}.txt").write_text(f"{cid} 0.325 0.775 0.35 0.45\n")
        rows.append({"cls": cls, "split": "test", "img": str(k), "contrast": "0.5" if cls == "LD2" else "1.0",
                     "box": json.dumps([0.15, 0.55, 0.5, 1.0])})
    rows.append({"cls": "LD1", "split": "test", "img": "9", "contrast": "0.8", "box": json.dumps([0, 0, 1, 1])})
    boxes = tmp_path / "b.csv"
    with boxes.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    out = tmp_path / "s.jpg"
    res = lf.render_rmdd(root, boxes, out)
    assert out.exists()
    assert len(res["examples"]["LD2"]) == 2 and len(res["examples"]["LD0"]) == 2
    assert res["medians"] == {"LD0": 1.0, "LD1": 0.8, "LD2": 0.5}
