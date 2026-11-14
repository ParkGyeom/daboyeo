import json
import zipfile

import cv2
import numpy as np
import pytest

from daboyeo_ai.data.damage_cls import (
    TYPES_CSV,
    build_type_table,
    crop_member_info,
    crop_type,
    krid_image_facilities,
    read_type_table,
    split_crop_name,
    square_input,
    subsample_cls,
    write_type_table,
)
from daboyeo_ai.data.pipeline import process_krid
from daboyeo_ai.data.yolo import Box, CropWriter, YoloDetWriter, facility_crop


def krid_json(file_name, anns):
    return {"images": [{"id": 1, "file_name": file_name, "width": 1280, "height": 720}], "annotations": anns}


def jpg(w=1280, h=720) -> bytes:
    rng = np.random.default_rng(0)
    return cv2.imencode(".jpg", rng.integers(0, 255, (h, w, 3), np.uint8))[1].tobytes()


def test_split_crop_name_and_member_info():
    assert split_crop_name("krid_국도_경기_20230902_003961_12") == ("국도_경기_20230902_003961", 12)
    assert crop_member_info("cls/val/damaged/krid_a_0.jpg") == ("val", "damaged", "krid_a_0")
    assert crop_member_info("det/images/train/x.jpg") is None
    assert crop_member_info("cls/train/normal/") is None
    with pytest.raises(ValueError):
        split_crop_name("rdd_x_0")


def test_types_recovered_from_real_pipeline_order(tmp_path):
    """process_krid 가 만든 크롭 이름의 k 가 krid_image_facilities 의 k 번째 종류와 같아야 한다.

    k=1 (작은 박스, 크롭 없음) 과 무시되는 시설물(신호등)이 끼어 있어도 순서가 맞는지 본다.
    """
    labels, images = tmp_path / "TL.zip", tmp_path / "TS.zip"
    with zipfile.ZipFile(labels, "w") as lz, zipfile.ZipFile(images, "w") as iz:
        lz.writestr("x/20230804_000001.json", json.dumps(krid_json("20230804_000001.jpg", [
            {"name": "도로 표지", "state": "정상", "bbox": [100, 100, 80, 60]},
            {"name": "신호등", "state": "정상", "bbox": [10, 10, 50, 50]},          # 우리 클래스 아님 → 번호 없음
            {"name": "방호울타리", "state": "파손", "bbox": [300, 400, 10, 10]},     # 작아서 크롭 없음, 번호는 1
            {"name": "방호울타리", "state": "파손", "segmentation": [[400, 400, 700, 420, 690, 480]]},
        ])))
        iz.writestr("y/20230804_000001.jpg", jpg())
    crops = CropWriter(tmp_path / "cls")
    process_krid(labels, images, YoloDetWriter(tmp_path / "det"), crops)
    members = [f"cls/{p.relative_to(tmp_path / 'cls').as_posix()}" for p in (tmp_path / "cls").rglob("*.jpg")]
    assert sorted(m.rsplit("/", 1)[1] for m in members) == ["krid_20230804_000001_0.jpg", "krid_20230804_000001_2.jpg"]

    facilities = list(krid_image_facilities(labels))
    assert facilities == [("20230804_000001", "sgg", "011")]
    table, stats = build_type_table(members, facilities)
    assert table == {"20230804_000001": "sgg"}
    assert stats["missing"] == stats["conflict"] == stats["label_mismatch"] == 0
    split = members[0].split("/")[1]
    assert stats[f"{split}_normal_sign"] == 1 and stats[f"{split}_damaged_guardrail"] == 1


def test_build_type_table_flags_problems():
    facilities = [("a", "gs", "01"), ("b", "s", "0"), ("b", "g", "0")]  # b: 라벨 zip 끼리 다름
    members = [
        "cls/train/normal/krid_a_0.jpg",
        "cls/train/normal/krid_a_1.jpg",    # 라벨은 파손 → mismatch
        "cls/test/normal/krid_zzz_0.jpg",   # 라벨 없음
        "cls/test/normal/krid_a_5.jpg",     # 번호 범위 밖
        "det/images/train/krid_a.jpg",      # 무시
    ]
    table, stats = build_type_table(members, facilities)
    assert table == {"a": "gs"}
    assert (stats["conflict"], stats["label_mismatch"], stats["missing"]) == (1, 1, 2)


def test_type_table_roundtrip_and_lookup(tmp_path):
    p = write_type_table({"b_1": "gs", "a": "s"}, tmp_path / "t.csv.gz")
    first = p.read_bytes()
    assert write_type_table({"a": "s", "b_1": "gs"}, p).read_bytes() == first  # 결정적
    t = read_type_table(p)
    assert t == {"a": "s", "b_1": "gs"}
    assert crop_type("krid_b_1_1.jpg", t) == "sign"
    assert crop_type("krid_b_1_0", t) == "guardrail"
    assert crop_type("krid_b_1_7", t) is None and crop_type("krid_nope_0", t) is None and crop_type("x", t) is None


def test_committed_type_table_is_sane():
    t = read_type_table(TYPES_CSV)
    assert len(t) > 10_000
    assert all(v and set(v) <= {"g", "s"} for v in t.values())


def test_facility_crop_matches_cropwriter_geometry(tmp_path):
    img = np.zeros((720, 1280, 3), np.uint8)
    img[200:300, 100:500] = 255
    box = Box(2, 100, 200, 500, 300)  # 400×100, pad 0.1 → 480×120 → 긴 변 224 로
    c = facility_crop(img, box)
    assert c.shape == (round(120 * 224 / 480), 224, 3)
    assert facility_crop(img, Box(2, 0, 0, 20, 100)) is None          # 짧은 변 < 24
    edge = facility_crop(img, Box(2, 0, 0, 100, 50), size=1000)  # pad 가 이미지 밖으로 나가면 잘린다
    assert edge.shape[:2] == (55, 110)

    w = CropWriter(tmp_path)
    assert w.write("x", img, box, "normal", "g")
    written = cv2.imread(str(next(tmp_path.rglob("x.jpg"))))
    assert written.shape == c.shape


def test_square_input_keeps_aspect():
    crop = np.full((50, 200, 3), 255, np.uint8)
    sq = square_input(crop, 224)
    assert sq.shape == (224, 224, 3)
    assert sq[0, 0].tolist() == [114, 114, 114] and sq[112, 112].tolist() == [255, 255, 255]
    assert square_input(np.zeros((224, 224, 3), np.uint8)).sum() == 0  # 이미 정사각형 224 면 그대로


def _make_cls(root, split, n_damaged, n_normal):
    img = np.full((40, 120, 3), 200, np.uint8)
    for label, n in (("damaged", n_damaged), ("normal", n_normal)):
        d = root / split / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cv2.imwrite(str(d / f"krid_{split}{label}_{i}.jpg"), img)


def test_subsample_cls_ratio_deterministic_and_square(tmp_path):
    src = tmp_path / "src"
    _make_cls(src, "train", 3, 40)
    _make_cls(src, "val", 2, 40)
    _make_cls(src, "test", 1, 5)  # normal 이 모자라면 있는 만큼
    out = subsample_cls(src, tmp_path / "a")
    assert out["train"] == {"damaged": 3, "normal": 15, "normal_available": 40}
    assert out["val"]["normal"] == 20 and out["test"]["normal"] == 5
    picked = sorted(p.name for p in (tmp_path / "a" / "train" / "normal").iterdir())
    subsample_cls(src, tmp_path / "b")
    assert picked == sorted(p.name for p in (tmp_path / "b" / "train" / "normal").iterdir())
    assert all(n.endswith(".png") for n in picked)
    assert cv2.imread(str(tmp_path / "a" / "train" / "normal" / picked[0])).shape == (224, 224, 3)

    out = subsample_cls(src, tmp_path / "a", normal_ratio={"train": 2})  # 다시 만들면 이전 것은 지운다
    assert len(list((tmp_path / "a" / "train" / "normal").iterdir())) == 6
