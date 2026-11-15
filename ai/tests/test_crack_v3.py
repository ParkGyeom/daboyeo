"""179 크랙 라벨 규칙 v3 (`CrackMerge`) — 기본값은 v2 와 같아야 한다."""

import json
import zipfile
from collections import Counter

import cv2
import numpy as np
import pytest

from daboyeo_ai.data import prepare_local, selection
from daboyeo_ai.data.pipeline import process_road_obstacle, relabel_road_obstacle
from daboyeo_ai.data.sources import CRACK, parse_coco_like
from daboyeo_ai.data.yolo import CrackMerge, YoloDetWriter, merge_polylines


def crack_js(file_name, polylines, w=1280, h=720):
    """실제 179 크랙 형식: images 는 객체 하나, bbox·segmentation 은 비고 polyline 에 [[x,y,...]]."""
    return {
        "images": {"id": 1, "file_name": file_name, "width": w, "height": h},
        "annotations": [
            {"id": k, "image_id": 1, "category_id": 1, "bbox": [], "segmentation": [], "polyline": [pl]}
            for k, pl in enumerate(polylines, 1)
        ],
        "categories": [{"id": 1, "name": "Crack"}],
    }


# 이어진 크랙 하나를 짧은 선분 3개로 그린 것 (1280x720 원본 px) + 멀리 떨어진 조각 하나
CHAIN = [[100, 600, 140, 580], [142, 579, 200, 540], [203, 538, 260, 500]]
FAR = [[1000, 200, 1060, 230]]


def test_merge_connected_segments_into_one_box():
    lines = [[(a[0], a[1]), (a[2], a[3])] for a in CHAIN + FAR]
    stats = Counter()
    boxes = merge_polylines(CRACK, lines, 1280, 720, CrackMerge(), stats)
    assert len(boxes) == 2 and stats["crack_polylines"] == 4 and stats["crack_merged_boxes"] == 2
    big = max(boxes, key=lambda b: b.width)
    assert (big.x1, big.y1, big.x2, big.y2) == pytest.approx((100, 500, 260, 600))
    assert all(b.cls == CRACK for b in boxes)


def test_merge_gap_controls_grouping():
    # 두 선 사이 간격 40 px(원본) = 20 px(640 기준): merge_px 16 이면 따로, 48 이면 하나
    lines = [[(100, 300), (300, 300)], [(340, 300), (540, 300)]]
    assert len(merge_polylines(CRACK, lines, 1280, 720, CrackMerge(merge_px=16))) == 2
    assert len(merge_polylines(CRACK, lines, 1280, 720, CrackMerge(merge_px=48))) == 1


def test_thin_vertical_crack_gets_min_thickness_and_short_bits_dropped():
    vertical = [(640, 100), (641, 500)]          # 폭 1 px 짜리 세로 크랙
    speck = [(50, 50), (60, 52)]                 # 640 기준 길이 5 px → 버림
    stats = Counter()
    boxes = merge_polylines(CRACK, [vertical, speck], 1280, 720, CrackMerge(min_thick_px=12), stats)
    assert len(boxes) == 1 and stats["crack_group_short"] == 1
    b = boxes[0]
    assert b.width == pytest.approx(24)          # 640 기준 12 px = 원본 24 px
    assert (b.y1, b.y2) == pytest.approx((100, 500))
    assert (b.x1 + b.x2) / 2 == pytest.approx(640.5)


def test_min_thickness_box_stays_inside_image():
    boxes = merge_polylines(CRACK, [[(0, 10), (0, 300)]], 1280, 720, CrackMerge())
    assert boxes[0].x1 == 0 and boxes[0].x2 == pytest.approx(12)


def test_cap_keeps_longest_groups():
    lines = [[(100 * k, 100), (100 * k, 100 + 20 * k)] for k in range(1, 9)]  # 서로 떨어진 8개, 길이 다름
    stats = Counter()
    boxes = merge_polylines(CRACK, lines, 1280, 720, CrackMerge(max_boxes=3, min_len_px=0), stats)
    assert len(boxes) == 3 and stats["crack_group_capped"] == 5
    assert sorted(round(b.height) for b in boxes) == [120, 140, 160]


def test_parse_coco_like_default_is_v2_and_v3_merges():
    js = crack_js("a_00010.png", CHAIN + FAR)
    v2 = parse_coco_like(js)[0]
    assert len(v2.boxes) == 4                    # 어노테이션마다 박스 (v2 그대로)
    stats = Counter()
    v3 = parse_coco_like(js, stats, CrackMerge())[0]
    assert len(v3.boxes) == 2 and stats["crack_merged_boxes"] == 2


def test_parse_coco_like_v3_ignores_bad_points_and_keeps_bbox_annotations():
    js = crack_js("a_00010.png", [[None, None, 100, 600, 140, 580, 142, 579, 200, 540], [None, None]])
    js["categories"].append({"id": 2, "name": "pothole"})
    js["annotations"].append({"id": 9, "image_id": 1, "category_id": 2, "bbox": [10, 10, 50, 40]})
    stats = Counter()
    li = parse_coco_like(js, stats, CrackMerge())[0]
    assert stats["ann_bad_points"] == 2 and stats["ann_bad_geometry"] == 1
    assert sorted(b.cls for b in li.boxes) == [0, CRACK]


def _crack_zips(tmp_path, stems):
    labels, images = tmp_path / "L.zip", tmp_path / "I.zip"
    img = cv2.imencode(".png", np.full((720, 1280, 3), 90, np.uint8))[1].tobytes()
    with zipfile.ZipFile(labels, "w") as lz, zipfile.ZipFile(images, "w") as iz:
        for s in stems:
            lz.writestr(f"x/{s}_PLINE.json", json.dumps(crack_js(f"{s}.png", CHAIN + FAR)))
            iz.writestr(f"y/{s}.png", img)
    return labels, images


def read_labels(root):
    return {p.stem: p.read_text().split("\n")[:-1] for p in root.glob("labels/*/*.txt")}


def test_process_road_obstacle_v3_vs_relabel_give_same_labels(tmp_path):
    stems = [f"V2F_seq_{i:05d}" for i in range(0, 10)]
    labels, images = _crack_zips(tmp_path, stems)
    v2 = YoloDetWriter(tmp_path / "v2" / "det")
    process_road_obstacle(labels, images, v2, stride=1)
    direct = YoloDetWriter(tmp_path / "v3" / "det")
    process_road_obstacle(labels, images, direct, stride=1, crack_merge=CrackMerge())
    relab = YoloDetWriter(tmp_path / "re" / "det")
    stats = relabel_road_obstacle(labels, tmp_path / "v2" / "det", relab, CrackMerge())

    assert stats["written"] == 10 and stats["relabel_no_json"] == 0
    assert read_labels(tmp_path / "re" / "det") == read_labels(tmp_path / "v3" / "det")
    assert all(len(v) == 4 for v in read_labels(tmp_path / "v2" / "det").values())
    assert all(len(v) == 2 for v in read_labels(tmp_path / "re" / "det").values())
    # 이미지는 그대로 (하드링크 또는 복사)
    src = sorted(p.name for p in (tmp_path / "v2" / "det").glob("images/*/*.jpg"))
    assert sorted(p.name for p in (tmp_path / "re" / "det").glob("images/*/*.jpg")) == src


def test_relabel_v2_rule_reproduces_original_labels(tmp_path):
    labels, images = _crack_zips(tmp_path, [f"V2F_seq_{i:05d}" for i in range(5)])
    process_road_obstacle(labels, images, YoloDetWriter(tmp_path / "v2" / "det"), stride=1)
    relabel_road_obstacle(labels, tmp_path / "v2" / "det", YoloDetWriter(tmp_path / "re" / "det"), None)
    assert read_labels(tmp_path / "re" / "det") == read_labels(tmp_path / "v2" / "det")


def test_named_crack_rows_and_cli_defaults():
    rows = [
        dict(label="a/2.CRACK/x/C_Mainroad_B02.zip", kind="CRACK", image="i/B02.zip"),
        dict(label="a/2.CRACK/x/C_Mainroad_B04.zip", kind="CRACK", image="i/B04.zip"),
        dict(label="a/2.CRACK/x/C.Mainroad_G01.zip", kind="CRACK", image=None),
    ]
    assert selection.named_crack_rows(rows, ["C_Mainroad_B04.zip", "C_Mainroad_B02.zip"], exclude=rows[:1]) == [rows[1]]
    with pytest.raises(KeyError):
        selection.named_crack_rows(rows, ["C.Mainroad_G01.zip"])
    args = prepare_local.parse_args([])
    assert args.crack_rule == "v2" and args.tag == "" and prepare_local.CRACK_RULES["v2"] is None
    assert prepare_local.Settings().crack_merge is None


def test_zero_size_label_uses_nominal_size(tmp_path):
    js = crack_js("a_00010.png", CHAIN, w=0, h=0)
    stats = Counter()
    li = parse_coco_like(js, stats, CrackMerge())[0]
    assert stats["crack_size_assumed"] == 1 and len(li.boxes) == 1

    labels, images = _crack_zips(tmp_path, ["V2F_seq_00000"])
    process_road_obstacle(labels, images, YoloDetWriter(tmp_path / "v2" / "det"), stride=1)
    with zipfile.ZipFile(labels, "w") as lz:   # 같은 이미지, 라벨 크기만 0
        lz.writestr("x/V2F_seq_00000_PLINE.json", json.dumps(crack_js("V2F_seq_00000.png", CHAIN + FAR, 0, 0)))
    stats = relabel_road_obstacle(labels, tmp_path / "v2" / "det", YoloDetWriter(tmp_path / "re" / "det"), CrackMerge())
    assert stats["relabel_size_assumed"] == 1 and stats["written"] == 1
    assert len(read_labels(tmp_path / "re" / "det")["ro_V2F_seq_00000"]) == 2


def test_source_subset_links_only_prefix(tmp_path):
    from daboyeo_ai.data.yolo import source_subset

    root = tmp_path / "det"
    for split, name in (("test", "rdd_a"), ("test", "ro_b"), ("val", "rdd_c")):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "images" / split / f"{name}.jpg").write_bytes(b"x")
        (root / "labels" / split / f"{name}.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    y = source_subset(root, tmp_path / "sub", "rdd_")
    assert [p.name for p in (tmp_path / "sub" / "images" / "test").iterdir()] == ["rdd_a.jpg"]
    assert (tmp_path / "sub" / "labels" / "test" / "rdd_a.txt").read_text().startswith("1 ")
    assert "test: images/test" in y.read_text()
    source_subset(root, tmp_path / "sub", "ro_")   # 다시 부르면 이전 링크는 지운다
    assert [p.name for p in (tmp_path / "sub" / "images" / "test").iterdir()] == ["ro_b.jpg"]
