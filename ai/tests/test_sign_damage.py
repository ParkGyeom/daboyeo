import csv
import zipfile

import cv2
import numpy as np

from daboyeo_ai.data.sign_damage import (
    RF_LABEL,
    build_cls_v2,
    build_rf_crops,
    degrade,
    dhash,
    hamming,
    mark_duplicates,
    pack,
    read_data_yaml_names,
    yolo_boxes,
)


def img(seed, w=640, h=480):
    rng = np.random.default_rng(seed)
    a = cv2.resize(rng.integers(0, 255, (12, 16, 3), np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    return a


def make_rf(root, names, items):
    """items: [(split, image_name, image, ["cls cx cy w h", ...])]"""
    root.mkdir(parents=True)
    (root / "data.yaml").write_text("train: ../train/images\nnc: 2\nnames: [" + ", ".join(f"'{n}'" for n in names) + "]\n")
    for split, name, im, lines in items:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / split / "images" / name), im)
        (root / split / "labels" / f"{name.rsplit('.', 1)[0]}.txt").write_text("\n".join(lines))


def test_label_map_covers_downloaded_classes():
    for n in ("bent", "broken-sheet", "broken sheet", "crack", "rust", "damaged"):
        assert RF_LABEL[n] == "damaged"
    for n in ("not-damaged", "normal", "healthy"):
        assert RF_LABEL[n] == "normal"
    assert RF_LABEL["graffiti"] is None


def test_yaml_names_and_yolo_boxes(tmp_path):
    make_rf(tmp_path / "p", ["broken sheet", "normal"], [])
    assert read_data_yaml_names(tmp_path / "p") == ["broken sheet", "normal"]
    t = tmp_path / "a.txt"
    t.write_text("1 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.3 0.1 0.3 0.3\n")
    (c0, b0), (c1, b1) = yolo_boxes(t, 100, 200)
    assert c0 == 1 and (b0.x1, b0.y1, b0.x2, b0.y2) == (40, 60, 60, 140)
    assert c1 == 0 and np.allclose((b1.x1, b1.y1, b1.x2, b1.y2), (10, 20, 30, 60))


def test_dhash_duplicates():
    a = img(1)
    b = cv2.GaussianBlur(a, (3, 3), 0)  # 거의 같음
    c = img(2)
    ha, hb, hc = dhash(a), dhash(b), dhash(c)
    assert hamming(ha, hb) <= 4 < hamming(ha, hc)
    assert mark_duplicates([ha, hc, hb]) == [0, 1, 0]


def test_degrade_small_and_deterministic():
    a = img(3, 400, 300)
    d1, d2 = degrade(a, "k"), degrade(a, "k")
    assert np.array_equal(d1, d2)
    assert 40 * 0.99 <= max(d1.shape[:2]) <= 160 * 1.01
    tiny = img(4, 30, 20)
    assert degrade(tiny, "x").shape[:2] == (20, 30)  # 이미 작으면 키우지 않는다


def test_build_rf_crops_maps_dedupes_and_filters(tmp_path):
    base = img(5)
    make_rf(tmp_path / "a", ["bent", "graffiti", "not-damaged"], [
        ("train", "x1.jpg", base, ["0 0.3 0.4 0.3 0.4", "1 0.7 0.5 0.2 0.3", "2 0.75 0.3 0.3 0.3"]),
        ("train", "pole.jpg", img(6), ["0 0.5 0.5 0.05 0.8"]),              # 세로로 길쭉 → 뺀다
    ])
    # 다른 출처에 같은 사진 (재업로드) → 중복
    make_rf(tmp_path / "b", ["damaged", "healthy"], [("valid", "x1_copy.jpg", base, ["0 0.3 0.4 0.3 0.4"])])
    rows, stats = build_rf_crops({"w/a": tmp_path / "a", "w/b": tmp_path / "b"}, tmp_path / "out")
    assert stats["excluded_graffiti"] == 1 and stats["dropped_aspect"] == 1 and stats["duplicate"] == 1
    kept = [r for r in rows if not r.dup_of]
    assert sorted(r.label for r in kept) == ["damaged", "normal"]
    dup = next(r for r in rows if r.dup_of)
    assert dup.source == "w/b" and dup.dup_of == next(r.file for r in kept if r.label == "damaged")
    for r in kept:
        assert (tmp_path / "out" / r.file).exists()
    assert not (tmp_path / "out" / dup.file).exists()
    with open(tmp_path / "out" / "sign_rf" / "manifest.csv") as f:
        assert len(list(csv.DictReader(f))) == 3
    z = pack(tmp_path / "out", tmp_path / "sign_rf.zip")
    names = zipfile.ZipFile(z).namelist()
    assert "sign_rf/manifest.csv" in names and all(n.startswith("sign_rf/") for n in names)


def test_build_cls_v2_keeps_val_test_krid_only(tmp_path):
    krid = tmp_path / "cls"
    table = {"s1": "s", "g1": "g", "n1": "ssgg"}
    files = {
        "train/damaged": ["krid_s1_0", "krid_g1_0"],
        "train/normal": [f"krid_n1_{k}" for k in range(4)],
        "val/damaged": ["krid_s1_0"], "val/normal": ["krid_n1_0", "krid_n1_2"],
        "test/damaged": ["krid_g1_0"], "test/normal": ["krid_n1_1"],
    }
    for d, names in files.items():
        (krid / d).mkdir(parents=True)
        for n in names:
            cv2.imwrite(str(krid / d / f"{n}.jpg"), img(7, 60, 40))
    rf = tmp_path / "rf" / "sign_rf"
    for split, label, n in (("train", "damaged", 3), ("train", "normal", 2), ("holdout", "damaged", 1)):
        (rf / split / label).mkdir(parents=True)
        for k in range(n):
            cv2.imwrite(str(rf / split / label / f"rf_x_{split}{k}.jpg"), img(8 + k, 50, 50))
    out = build_cls_v2(krid, tmp_path / "rf", tmp_path / "dst", "sign", {"train": 1, "val": 10, "test": 10},
                       types_table=table)
    assert out["train"] == {"krid_damaged": 1, "krid_normal": 1, "krid_normal_available": 2,
                            "rf_damaged": 3, "rf_normal": 2}
    assert out["val"]["krid_damaged"] == 1 and out["test"]["krid_damaged"] == 0
    assert not any("rf_" in p.name for p in (tmp_path / "dst").glob("val/*/*")) and \
        not any("rf_" in p.name for p in (tmp_path / "dst").glob("test/*/*"))
    assert cv2.imread(str(next((tmp_path / "dst" / "train" / "damaged").glob("rf_*.png")))).shape == (224, 224, 3)
    # rf 없이 all → v1b 와 같은 구성
    out2 = build_cls_v2(krid, None, tmp_path / "dst2", "all", {"train": 1}, types_table=table)
    assert out2["train"] == {"krid_damaged": 2, "krid_normal": 2, "krid_normal_available": 4}
