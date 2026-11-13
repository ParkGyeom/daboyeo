from pathlib import Path

from daboyeo_ai.data.aihub import AihubFile, merge_parts, parse_tree

HERE = Path(__file__).parent


def test_parse_real_krid_tree():
    files = parse_tree((HERE / "fixture_tree_71786.txt").read_text(encoding="utf-8"))
    by_name = {f.name: f for f in files}
    assert len(files) == 37  # Other 16 + TS 5 + TL 5 + VS 5 + VL 5 + SbL 1
    tl = by_name["TL_2.국도_02.도시부.zip"]
    assert tl.key == 547010
    assert tl.path.endswith("Training/02.라벨링데이터/TL_2.국도_02.도시부.zip")
    assert by_name["VL_3.지방도_02.도시부.zip"].path.split("/")[-3] == "Validation"
    assert by_name["SbL.zip"].size_bytes == 17 * 1024**3


def test_parse_real_road_obstacle_tree_paths():
    files = parse_tree((HERE / "fixture_tree_179_head.txt").read_text(encoding="utf-8"))
    ann = [f for f in files if "/1.Annotations/1.TOA/" in f.path]
    img = [f for f in files if "/2.Images_new_211012/1.TOA/" in f.path]
    assert ann and img
    a = next(f for f in ann if f.name == "1.Frontback_F01.zip")
    assert a.path.endswith("2.Validation/1.Annotations/1.TOA/1.Frontback/1.Frontback_F01.zip")
    # 라벨 zip 경로에서 이미지 zip 경로를 만들 수 있어야 한다 (노트북의 매칭 규칙)
    img_paths = {f.path for f in img}
    assert a.path.replace("1.Annotations", "2.Images_new_211012") in img_paths


def test_fetch_tree_falls_back_to_snapshot(monkeypatch, capsys):
    import urllib.error

    from daboyeo_ai.data import aihub

    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 502, "Bad Gateway", {}, None)

    monkeypatch.setattr(aihub.urllib.request, "urlopen", boom)
    for key, must_have in ((71786, "TL_2.국도_02.도시부.zip"), (179, "5.Mainroad_F01.zip")):
        files = aihub.fetch_tree(key, retries=2, wait_s=0)
        assert must_have in {f.name for f in files}
    assert "using snapshot" in capsys.readouterr().out
    # 스냅샷의 라벨↔이미지 짝이 노트북 규칙대로 맞아야 한다
    road = aihub.fetch_tree(179, retries=1, wait_s=0)
    paths = {f.path for f in road}
    ann = [f for f in road if "/1.Annotations/" in f.path]
    matched = [f for f in ann if f.path.replace("1.Annotations", "2.Images_new_211012") in paths]
    assert len(ann) > 50 and len(matched) / len(ann) > 0.8


def test_size_bytes_units():
    assert AihubFile("a/x.zip", "176 KB", 1).size_bytes == 176 * 1024
    assert AihubFile("a/x.zip", "1.5 GB", 1).size_bytes == int(1.5 * 1024**3)


def test_merge_parts_natural_order(tmp_path):
    d = tmp_path / "a" / "b"
    d.mkdir(parents=True)
    # part10 이 part2 보다 뒤에 와야 한다 (문자열 정렬이면 깨짐)
    for i in range(12):
        (d / f"x.zip.part{i}").write_bytes(bytes([i]))
    merged = merge_parts(tmp_path)
    assert merged == [d / "x.zip"]
    assert (d / "x.zip").read_bytes() == bytes(range(12))
    assert not list(d.glob("*.part*"))
