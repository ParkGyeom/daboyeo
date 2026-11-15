"""lane_wear_roadmark: 작은 합성 CSV/JSON/PNG 로만 시험한다 (AI Hub 파일은 쓰지 않는다)."""

import json
import zipfile

import cv2
import numpy as np
import pytest
from daboyeo_ai.eval import lane_wear_roadmark as rm
from daboyeo_ai.lane_wear import LaneWearResult

W, H = 1280, 720
LEFT = np.array([[260.0, 690.0], [590.0, 440.0]])
RIGHT = np.array([[1020.0, 690.0], [690.0, 440.0]])


def frame_with_lanes(paint=200, road=80, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), road, np.uint8)
    img[: H // 2] = (200, 170, 140)
    img = np.clip(img + rng.normal(0, 3, (H, W, 1)), 0, 255).astype(np.uint8)
    for p in (LEFT, RIGHT):
        cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), (paint,) * 3, 12)
    return img


def label_json(polys, cats=(383, 385), images_as_list=False):
    img = {"filename": "x.png", "width": W, "height": H, "time": "light"}
    return {"images": [img] if images_as_list else img,
            "annotations": [{"polyline": [p.ravel().tolist()], "category_id": c, "image_id": "1", "id": str(i)}
                            for i, (p, c) in enumerate(zip(polys, cats))]
            + [{"segmentation": [[0, 0, 1, 1, 1, 0]], "bbox": [0, 0, 1, 1], "category_id": 440, "image_id": "1"}],
            "categories": [{"id": 383, "name": "차선"}]}


def test_parse_name_and_drive():
    assert rm.parse_name("0401_light_clear_smooth_01000085")["time"] == "light"
    assert rm.drive_of("0401_night_rainy_traffic_03000012") == "0401_night_rainy_traffic_0300"
    with pytest.raises(ValueError):
        rm.parse_name("bad_name")


def test_load_ltlm_skips_empty(tmp_path):
    (tmp_path / "a_light_clear_smooth_01000001.csv").write_text("117\n")
    (tmp_path / "b_light_clear_smooth_01000002.csv").write_text("")
    (tmp_path / "c_light_clear_smooth_01000003.csv").write_text("n/a")
    assert rm.load_ltlm(tmp_path) == {"a_light_clear_smooth_01000001": 117.0}


@pytest.mark.parametrize("as_list", [False, True])
def test_load_lines_keeps_only_line_polylines(tmp_path, as_list):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(label_json([LEFT, RIGHT], images_as_list=as_list)))
    lines = rm.load_lines(p)
    assert [c for c, _ in lines] == [383, 385]
    assert lines[0][1].shape == (2, 2)


def test_segments_on_lines_hit_and_miss():
    lines = [(383, LEFT)]
    on = [(265, 686, 585, 444)]
    off = [(900, 690, 1000, 450)]
    tot, hit = rm.segments_on_lines(on, lines, tol_px=10)
    assert tot > 10 and hit == tot
    tot, hit = rm.segments_on_lines(off, lines, tol_px=10)
    assert hit == 0
    assert rm.segments_on_lines(on, [], tol_px=10)[1] == 0


def test_ego_lines_pick_nearest_each_side():
    far_left = np.array([[0.0, 700.0], [500.0, 450.0]])
    high = np.array([[700.0, 300.0], [800.0, 200.0]])  # 화면 아래까지 안 내려온 선은 무시
    lines = [(385, far_left), (383, LEFT), (383, RIGHT), (383, high), (386, RIGHT)]
    ego = rm.ego_lines(lines, W, H)
    assert ego["left"][1] is LEFT and ego["right"][1] is RIGHT
    assert rm.ego_lines([(383, LEFT)], W, H)["right"] is None


def test_label_contrast_tracks_paint_brightness():
    bright = rm.label_contrast(frame_with_lanes(paint=200), LEFT)
    dim = rm.label_contrast(frame_with_lanes(paint=110), LEFT)
    assert bright is not None and dim is not None
    assert bright == pytest.approx(200 / 80 - 1, abs=0.15)
    assert dim < bright
    assert rm.label_contrast(frame_with_lanes(), np.array([[0.0, 100.0], [10.0, 50.0]])) is None


def test_spearman():
    a = np.arange(10.0)
    assert rm.spearman(a, a * 3 + 1) == pytest.approx(1.0)
    assert rm.spearman(a, -a) == pytest.approx(-1.0)
    assert rm.spearman(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0])) > 0.9
    assert np.isnan(rm.spearman(np.ones(5), a[:5]))


def _png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def test_image_source_dir_and_zip(tmp_path):
    img = frame_with_lanes()
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "x_light_clear_smooth_01000001.png").write_bytes(_png_bytes(img))
    z = tmp_path / "VS4.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("IMAGE/x_light_clear_smooth_01000001.png", _png_bytes(img))
        zf.writestr("LTL-M/x_light_clear_smooth_01000001.csv", "5")
    for src in (rm.ImageSource(d), rm.ImageSource(z)):
        assert src.names() == ["x_light_clear_smooth_01000001"]
        assert src.read("x_light_clear_smooth_01000001").shape == (H, W, 3)
        assert src.read("missing") is None


def _frame(name, ltl, contrast_p25, paint_len=1.0, found=True):
    r = LaneWearResult(lane_found=found, contrast=contrast_p25 if found else None, is_worn=False, confidence=0.5,
                       lane_pixel_ratio=0.1, n_segments=2, contrast_p25=contrast_p25 if found else None,
                       paint_len=paint_len if found else 0.0)
    p = rm.parse_name(name)
    return rm.Frame(name, rm.drive_of(name), p["time"], p["weather"], ltl, r)


def test_evaluate_counts_and_perfect_score():
    frames = []
    for i in range(40):
        ltl = [10, 60, 90, 130, 160][i % 5]
        frames.append(_frame(f"0401_light_clear_smooth_0100{i:04d}", ltl, ltl / 100))
    frames.append(_frame("0401_light_clear_smooth_01009998", None, 1.0))  # LTL-M 없음: 기준(중앙값)에만 쓴다
    frames.append(_frame("0401_light_clear_smooth_01009999", 140, None, found=False))
    res = rm.evaluate(frames)
    day = res["light"]
    assert day["n_no_paint"] == 8 and day["n_paint"] == 33 and day["n_worn"] == 16
    assert day["lane_found_rate"]["normal"] == pytest.approx(16 / 17)
    star = day["scores"]["★ score (p25 ÷ 주행 중앙값 × paint_len)"]
    assert star["n"] == 32 and star["auc"] == pytest.approx(1.0) and star["best_f1"] == pytest.approx(1.0)
    assert star["spearman_ltl"] == pytest.approx(1.0)
    assert res["night"]["n"] == 0
    assert set(day["star_flag_rate_no_paint"]) == {"0.6", "0.8", "1.0", "1.2", "1.4", "1.7", "2.0"}


def test_load_gps_and_road_brightness(tmp_path):
    (tmp_path / "g.csv").write_text("64907.28,37.0658417,127.9272414,21.3,104.6,0.8,1.7,1.8\n")
    (tmp_path / "bad.csv").write_text("x\n")
    t, lat, lon = rm.load_gps(tmp_path / "g.csv")
    assert t == pytest.approx(6 * 3600 + 49 * 60 + 7.28) and lat == pytest.approx(37.0658417)
    assert rm.load_gps(tmp_path / "bad.csv") is None and rm.load_gps(tmp_path / "none.csv") is None
    assert rm.road_brightness(frame_with_lanes(road=80)) == pytest.approx(80, abs=3)


def test_ltl_consistency_consecutive_and_repeat_pass():
    frames = []
    for i, v in enumerate([100, 102, 10, 12, 140, 141]):  # 한 주행, 1초 간격
        f = _frame(f"0401_light_clear_smooth_0100{i:04d}", v, 1.0)
        f.gps = (1000.0 + i, 37.0 + i * 1e-3, 127.0)
        frames.append(f)
    for i, v in enumerate([130, 133]):  # 밤에 같은 곳(첫 두 프레임 자리)을 다시 지남
        f = _frame(f"0401_night_clear_smooth_0300{i:04d}", v, 1.0)
        f.gps = (5000.0 + i, 37.0 + i * 1e-3, 127.0)
        frames.append(f)
    c = rm.ltl_consistency(frames)
    assert c["consecutive"]["n_pairs"] == 3  # (100,102) (140,141) (130,133)
    assert c["consecutive"]["p_no_paint_after_no_paint"] == pytest.approx(0.5)
    rp = c["repeat_pass"]["light-night"]
    assert rp["n_pairs"] == 2 and rp["mean_first"] == pytest.approx(101) and rp["mean_second"] == pytest.approx(131.5)


def test_cache_roundtrip():
    f = _frame("0401_night_clear_smooth_03000001", 120, 0.4)
    f.oracle = {"left": 0.5, "right": None}
    f.gps = (1.0, 37.0, 127.0)
    g = rm.frame_from_cache(json.loads(json.dumps(rm.frame_to_cache(f))))
    assert g == f


def test_main_end_to_end_writes_aggregates_only(tmp_path):
    ltl_dir, lab_dir, img_dir = tmp_path / "LTL-M", tmp_path / "2D_JSON", tmp_path / "IMAGE"
    for d in (ltl_dir, lab_dir, img_dir):
        d.mkdir()
    names = []
    for i, (ltl, paint) in enumerate([(150, 220), (140, 210), (80, 120), (70, 115), (10, 200), (130, 200)]):
        n = f"0401_light_clear_smooth_0100{i:04d}"
        names.append(n)
        (ltl_dir / f"{n}.csv").write_text(f"{ltl}\n")
        (lab_dir / f"{n}.json").write_text(json.dumps(label_json([LEFT, RIGHT])))
        (img_dir / f"{n}.png").write_bytes(_png_bytes(frame_with_lanes(paint=paint, seed=i)))
    out = tmp_path / "res.json"
    cache = tmp_path / "cache.json"
    gps_dir = tmp_path / "GPS"
    gps_dir.mkdir()
    for i, n in enumerate(names):
        (gps_dir / f"{n}.csv").write_text(f"{100000 + i},37.{i:04d},127.0,30,0,0,0,0\n")
    args = ["--ltlm", str(ltl_dir), "--images", str(img_dir), "--labels", str(lab_dir), "--out", str(out),
            "--gps", str(gps_dir),
            "--cache", str(cache)]
    assert rm.main(args) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_frames"] == 6 and res["light"]["n_no_paint"] == 1 and res["light"]["n_worn"] == 2
    geo = res["lane_geometry"]["light"]
    assert geo["n_with_ego_line"] == 6 and geo["lane_found_rate_with_ego_line"] == 1.0
    assert geo["segment_precision"] > 0.9
    assert res["light"]["oracle"]["left"]["n"] == 5
    assert res["light"]["road_brightness"]["n"] == 5
    assert res["ltl_consistency"]["consecutive"]["n_pairs"] == 3  # 10 이 낀 두 쌍은 뺀다
    text = out.read_text(encoding="utf-8")
    assert not any(n in text for n in names)  # 프레임 이름(프레임별 값)은 결과에 남기지 않는다
    assert cache.exists() and rm.main(args) == 0  # 두 번째는 캐시에서 읽는다
