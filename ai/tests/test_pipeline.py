import json
import zipfile

import cv2
import numpy as np
import pytest

from daboyeo_ai.data.pipeline import (
    count_label_zip,
    peek_label_zip,
    process_krid,
    process_rdd2022,
    process_road_obstacle,
)
from daboyeo_ai.data.sources import CRACK, GUARDRAIL, POTHOLE, SIGN, parse_coco_like, road_obstacle_class
from daboyeo_ai.data.yolo import CropWriter, YoloDetWriter, write_data_yaml


def jpg(w=1280, h=720) -> bytes:
    img = np.full((h, w, 3), 128, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def read_labels(root):
    out = {}
    for p in root.glob("labels/*/*.txt"):
        out[p.stem] = [line.split() for line in p.read_text().splitlines()]
    return out


# ---------------------------------------------------------------- KRID

def krid_json(file_name, anns):
    return {
        "images": [{"id": 1, "file_name": file_name, "width": 1280, "height": 720}],
        "annotations": anns,
    }


def test_process_krid_boxes_polygons_and_damage(tmp_path):
    labels, images = tmp_path / "TL.zip", tmp_path / "TS.zip"
    with zipfile.ZipFile(labels, "w") as z:
        z.writestr("a/20230804_000001.json", json.dumps(krid_json("20230804_000001.jpg", [
            {"name": "도로 표지", "state": "정상", "bbox": [100, 100, 80, 60], "segmentation": []},
            # 가드레일은 폴리곤만 있고 bbox 가 비어 있다 (AI Hub 예시와 같음)
            {"name": "방호울타리", "state": "파손", "bbox": [], "segmentation": [200, 400, 600, 380, 610, 450, 210, 470]},
            {"name": "기둥", "state": "정상", "bbox": [10, 10, 5, 5], "segmentation": []},
        ]), ensure_ascii=False))
        z.writestr("a/20230804_000002.json", json.dumps(krid_json("20230804_000002.jpg", [
            {"name": "CCTV", "state": "정상", "bbox": [10, 10, 50, 50], "segmentation": []},
        ]), ensure_ascii=False))
    with zipfile.ZipFile(images, "w") as z:
        z.writestr("img/20230804_000001.jpg", jpg())
        z.writestr("img/20230804_000002.jpg", jpg())

    det, crops = YoloDetWriter(tmp_path / "det"), CropWriter(tmp_path / "cls")
    stats = process_krid(labels, images, det, crops)

    assert stats["written"] == 1 and stats["skip_no_target"] == 1
    lab = read_labels(tmp_path / "det")["krid_20230804_000001"]
    assert sorted(int(r[0]) for r in lab) == [GUARDRAIL, SIGN]
    sign = next(r for r in lab if int(r[0]) == SIGN)
    assert [float(v) for v in sign[1:]] == pytest.approx([140 / 1280, 130 / 720, 80 / 1280, 60 / 720], abs=1e-5)
    assert crops.stats["crop_guardrail_damaged"] == 1 and crops.stats["crop_sign_normal"] == 1
    # 저장된 이미지는 긴 변 640으로 줄어 있어야 한다
    saved = next((tmp_path / "det").glob("images/*/*.jpg"))
    assert max(cv2.imread(str(saved)).shape[:2]) == 640


def test_krid_stride_keeps_damaged_frames(tmp_path):
    labels, images = tmp_path / "TL.zip", tmp_path / "TS.zip"
    with zipfile.ZipFile(labels, "w") as lz, zipfile.ZipFile(images, "w") as iz:
        for i in range(6):  # 홀수 프레임만 파손 → stride=2 면 예전에는 파손이 전부 빠졌다
            lz.writestr(f"{i}.json", json.dumps(krid_json(f"20230804_{i:06d}.jpg", [
                {"name": "방호울타리", "state": "파손" if i % 2 else "정상", "bbox": [100, 100, 300, 50], "segmentation": []},
            ]), ensure_ascii=False))
            iz.writestr(f"img/20230804_{i:06d}.jpg", jpg())
    crops = CropWriter(tmp_path / "cls")
    stats = process_krid(labels, images, YoloDetWriter(tmp_path / "det"), crops, stride=2)
    assert crops.stats["crop_guardrail_damaged"] == 3  # 파손 프레임은 전부 유지
    assert crops.stats["crop_guardrail_normal"] == 3  # 정상 프레임은 0, 2, 4 만
    assert stats["skip_stride"] == 0 and stats["written_with_damage"] == 3


def test_count_krid_label_zip(tmp_path):
    p = tmp_path / "TL.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("1.json", json.dumps(krid_json("1.jpg", [
            {"name": "방호울타리", "state": "파손", "bbox": [], "segmentation": [0, 0, 1, 1]},
            {"name": "방호울타리", "state": "정상", "bbox": [], "segmentation": [0, 0, 1, 1]},
        ]), ensure_ascii=False))
    c = count_label_zip(p, "krid")
    assert c["방호울타리|파손"] == 1 and c["방호울타리|정상"] == 1 and c["_json_files"] == 1


# ---------------------------------------------------------------- 도로장애물/표면

@pytest.mark.parametrize("name,expected", [
    ("포트홀", POTHOLE), ("Pothole on road", POTHOLE), ("보수완료 포트홀", "negative"),
    ("Filled pothole", "negative"), ("맨홀", "negative"), ("크랙", CRACK), ("라바콘", None),
])
def test_road_obstacle_category_mapping(name, expected):
    assert road_obstacle_class(name) == expected


def coco(images, anns, cats):
    return {"info": {}, "images": images, "annotations": anns, "categories": cats}


def test_parse_coco_like_rejects_unknown_format():
    with pytest.raises(ValueError, match="not COCO-like"):
        parse_coco_like({"dataset": {}, "image": []})


def test_process_road_obstacle_pothole_negative_and_stride(tmp_path):
    cats = [{"id": 1, "name": "포트홀"}, {"id": 2, "name": "보수완료 포트홀"}]
    ims = [{"id": i, "file_name": f"V1F_HY_0002_Seoul_Day_{i:04d}.png", "width": 1920, "height": 1080} for i in range(10)]
    anns = [{"id": i, "image_id": i, "category_id": 1, "bbox": [500, 700, 120, 60]} for i in (0, 5)]
    anns += [{"id": 100, "image_id": 1, "category_id": 2, "bbox": [10, 900, 50, 30]}]  # 보수완료만 → 배경 후보
    labels, images = tmp_path / "ann.zip", tmp_path / "img.zip"
    with zipfile.ZipFile(labels, "w") as z:
        z.writestr("ann/a.json", json.dumps(coco(ims, anns, cats), ensure_ascii=False))
    with zipfile.ZipFile(images, "w") as z:
        for im in ims:
            z.writestr(f"img/{im['file_name']}", jpg(1920, 1080))

    det = YoloDetWriter(tmp_path / "det")
    stats = process_road_obstacle(labels, images, det, stride=5, negative_ratio=1.0)
    assert stats["written"] == 2  # 프레임 0, 5
    assert stats["skip_stride"] == 8
    det2 = YoloDetWriter(tmp_path / "det2")
    stats2 = process_road_obstacle(labels, images, det2, stride=1, negative_ratio=1.0)
    assert stats2["written"] == 2 and stats2["written_negative"] == 1
    assert det2.stats["img_background"] == 1


def test_process_road_obstacle_crack_from_mask(tmp_path):
    cats = [{"id": 1, "name": "크랙"}]
    ims = [{"id": 1, "file_name": "C_Mainroad_0001.png", "width": 640, "height": 360}]
    mask = np.zeros((360, 640), np.uint8)
    mask[100:110, 50:300] = 255  # 가로 균열 1개
    mask[200:300, 400:405] = 255  # 세로 균열 1개
    labels, images = tmp_path / "ann.zip", tmp_path / "img.zip"
    with zipfile.ZipFile(labels, "w") as z:
        z.writestr("ann/c.json", json.dumps(coco(ims, [], cats), ensure_ascii=False))
        z.writestr("ann/C_Mainroad_0001.png", cv2.imencode(".png", mask)[1].tobytes())
    with zipfile.ZipFile(images, "w") as z:
        z.writestr("img/C_Mainroad_0001.png", jpg(640, 360))
    det = YoloDetWriter(tmp_path / "det")
    stats = process_road_obstacle(labels, images, det, stride=1)
    assert stats["crack_from_mask"] == 1
    rows = read_labels(tmp_path / "det")["ro_C_Mainroad_0001"]
    assert len(rows) == 2 and all(int(r[0]) == CRACK for r in rows)


def test_peek_label_zip_shows_structure(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.json", json.dumps(coco([], [], [{"id": 1, "name": "포트홀"}]), ensure_ascii=False))
    out = peek_label_zip(p)
    assert "categories" in out and "포트홀" in out


# ---------------------------------------------------------------- RDD2022

def rdd_xml(fname, objs):
    o = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, (a, b, c, d) in objs
    )
    return f"<annotation><filename>{fname}</filename><size><width>600</width><height>600</height></size>{o}</annotation>"


def test_process_rdd2022_maps_classes_and_skips_countries(tmp_path):
    p = tmp_path / "rdd.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("RDD2022/Japan/train/annotations/xmls/Japan_000001.xml",
                   rdd_xml("Japan_000001.jpg", [("D40", (10, 10, 60, 60)), ("D10", (100, 100, 300, 120)), ("D43", (0, 0, 9, 9))]))
        z.writestr("RDD2022/Japan/train/images/Japan_000001.jpg", jpg(600, 600))
        z.writestr("RDD2022/China_Drone/train/annotations/xmls/China_Drone_000001.xml",
                   rdd_xml("China_Drone_000001.jpg", [("D40", (10, 10, 60, 60))]))
        z.writestr("RDD2022/China_Drone/train/images/China_Drone_000001.jpg", jpg(600, 600))
    det = YoloDetWriter(tmp_path / "det")
    stats = process_rdd2022(p, det)
    assert stats["written_Japan"] == 1 and stats["skip_country"] == 1
    rows = read_labels(tmp_path / "det")["rdd_Japan_000001"]
    assert sorted(int(r[0]) for r in rows) == [POTHOLE, CRACK]  # D43 은 버림


def test_process_rdd2022_nested_country_zip(tmp_path):
    inner = tmp_path / "Czech.zip"
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("train/annotations/xmls/Czech_000001.xml", rdd_xml("Czech_000001.jpg", [("D00", (10, 10, 200, 40))]))
        z.writestr("train/images/Czech_000001.jpg", jpg(600, 600))
    outer = tmp_path / "rdd.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.write(inner, "RDD2022/Czech.zip")
    stats = process_rdd2022(outer, YoloDetWriter(tmp_path / "det"))
    assert stats["written_Czech"] == 1


def test_data_yaml_lists_contract_classes(tmp_path):
    text = write_data_yaml(tmp_path).read_text()
    assert "0: pothole" in text and "3: sign" in text and "val: images/val" in text
