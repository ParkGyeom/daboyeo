import json
import zipfile

import cv2
import numpy as np

from daboyeo_ai.data.lane_seg import (
    block_group,
    convert,
    half_width,
    instances,
    label_lines,
    zip_dir,
)


def test_block_group_keeps_neighbours_together():
    a = block_group("0401_light_clear_smooth_06000045")
    assert a == block_group("0401_light_clear_smooth_06000049")
    assert a != block_group("0401_light_clear_smooth_06000050")
    assert a != block_group("0401_night_clear_smooth_06000045")
    assert block_group("odd_name") == "odd_name"


def test_half_width_grows_toward_bottom():
    hw = half_width(np.array([0, 486, 800, 1080]), 1080, 1920)
    assert hw[0] == hw[1] == 1.5  # 지평선 위·지평선
    assert hw[2] < hw[3]
    assert abs(hw[3] - 0.008 * 1920) < 1e-6


def test_instances_classes_and_shapes():
    anns = [
        {"category_id": 383, "polyline": [[960, 1080, 960, 500]]},           # 차선 (세로선)
        {"category_id": 415, "segmentation": [[100, 900, 300, 900, 300, 1000, 100, 1000]]},  # 화살표
        {"category_id": 401, "bbox": [500, 900, 100, 50]},                    # bbox 만 있는 글자
        {"category_id": 418},                                                 # 아무것도 없음
    ]
    from collections import Counter
    st = Counter()
    out = instances(anns, 1920, 1080, 0.5, st)
    assert [c for c, _ in out] == [0, 1, 1]
    lane = out[0][1]
    assert abs(lane[:, 0].mean() - 480) < 3            # 가운데 x (0.5 배)
    assert lane[:, 0].max() - lane[:, 0].min() > 10     # 아래쪽은 폭이 있다
    assert lane[:, 1].max() <= 540
    assert st["ann_empty"] == 1 and st["ann_bbox_only"] == 1
    lines = label_lines(out, 960, 540)
    vals = [float(v) for v in lines[0].split()[1:]]
    assert all(0 <= v <= 1 for v in vals) and len(vals) % 2 == 0 and len(vals) >= 6


def test_polyline_outside_image_is_clipped():
    out = instances([{"category_id": 381, "polyline": [[-500, 1200, 2500, 1200]]}], 1920, 1080, 0.5)
    assert out == []


def test_convert_and_zip(tmp_path):
    labels, imgs = tmp_path / "lab", tmp_path / "img"
    labels.mkdir()
    imgs.mkdir()
    for i in range(3):
        name = f"0401_light_rainy_smooth_0100{i:04d}"
        cv2.imwrite(str(imgs / f"{name}.png"), np.full((1080, 1920, 3), 90, np.uint8))
        (labels / f"{name}.json").write_text(json.dumps({
            "images": [{"time": "light", "weather": "rainy"}],
            "annotations": [{"category_id": 383, "polyline": [[900, 1080, 950, 600]]}],
        }))
    (labels / "0401_light_rainy_smooth_09999999.json").write_text(json.dumps({"annotations": []}))
    res = convert(labels, imgs, tmp_path / "out", max_side=480, preview_n=2)
    st = res["stats"]
    assert st["no_image"] == 1
    assert st.get("img_train", 0) + st.get("img_val", 0) == 3
    assert st["cond_light_rainy"] == 3
    split = "train" if st.get("img_train") else "val"   # 같은 묶음이라 한쪽에 다 들어간다
    lab = next((tmp_path / "out" / "labels" / split).glob("*.txt")).read_text().split()
    assert lab[0] == "0"
    assert cv2.imread(str(next((tmp_path / "out" / "images" / split).glob("*.jpg")))).shape[1] == 480
    y = (tmp_path / "out" / "data.yaml").read_text()
    assert "0: lane" in y and "1: marking" in y and "\ntest:" not in y
    z = zip_dir(tmp_path / "out", tmp_path / "ls.zip")
    names = zipfile.ZipFile(z).namelist()
    assert any(n.startswith("lane_seg/images/") for n in names)
    assert "lane_seg/stats.json" in names
    assert not any("preview" in n or n.endswith("data.yaml") for n in names)
