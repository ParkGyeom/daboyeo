import io
import json
import subprocess
import urllib.error
import zipfile
from collections import Counter

import cv2
import numpy as np
import pytest
from daboyeo_ai.data import aihub, prepare_local, selection
from daboyeo_ai.data.aihub import AihubFile
from daboyeo_ai.data.sources import CRACK, POTHOLE

SECRET = "sk-test-SECRET-1234567890"


def road_file(path, size="1 GB", key=1):
    return AihubFile(path, size, key)


# ---------------------------------------------------------------- selection

def test_krid_jobs_pair_labels_from_snapshot():
    files = aihub.parse_tree((aihub.SNAPSHOT_DIR / "71786.txt").read_text(encoding="utf-8"))
    jobs = selection.krid_jobs(files)
    assert [j.image.name for j in jobs] == list(selection.KRID_IMAGE_ZIPS)
    assert jobs[0].label.name == "TL_2.국도_02.도시부.zip"
    assert jobs[-1].label.name == "VL_3.지방도_02.도시부.zip"
    assert all("/02.라벨링데이터/" in j.label.path for j in jobs)
    assert len(selection.krid_label_files(files)) == 10
    with pytest.raises(KeyError):
        selection.krid_jobs(files, ["TS_9.없음.zip"])


def test_road_label_image_pairing_and_unique_part_names():
    files = aihub.parse_tree((aihub.SNAPSHOT_DIR / "179.txt").read_text(encoding="utf-8"))
    by_path = {f.path: f for f in files}
    labels = selection.road_label_files(files)
    pairs = [(f, selection.road_image_for(f, by_path)) for f in labels]
    assert sum(img is not None for _, img in pairs) / len(pairs) > 0.8
    # 179 는 라벨 zip 과 이미지 zip 이름이 같다 (그래서 download 는 fileSn 폴더에 받는다)
    assert all(lab.name == img.name for lab, img in pairs if img)
    jobs = [selection.Job("road", 179, img, lab) for lab, img in pairs if img]
    assert len({j.part for j in jobs}) == len(jobs)


def test_road_count_row_classifies_categories():
    lab = road_file("x/2.Validation/1.Annotations/1.TOA/5.Mainroad/5.Mainroad_F01.zip")
    img = road_file("x/2.Validation/2.Images_new_211012/1.TOA/5.Mainroad/5.Mainroad_F01.zip", "2 GB")
    counts = Counter({"포트홀": 10, "보수완료 포트홀": 3, "맨홀": 1, "크랙": 4, "차량": 99, "_json_files": 7})
    row = selection.road_count_row(lab, counts, img)
    assert row["kind"] == "TOA" and row["road"] == "Mainroad"
    assert (row["pothole"], row["crack"], row["negative"], row["json_files"]) == (10, 4, 4, 7)
    assert row["image_gb"] == pytest.approx(2.0)
    crack = selection.road_count_row(
        road_file("x/1.Annotations/2.CRACK/3.Industrialroads/a.zip"), Counter(), None)
    assert crack["kind"] == "CRACK" and crack["road"] == "other" and crack["image"] is None


def test_pick_road_rows_ranks_per_gb_and_filters():
    def row(name, kind, road, pothole=0, crack=0, gb=1.0, image=True):
        return dict(label=name, kind=kind, road=road, pothole=pothole, crack=crack,
                    image=f"img/{name}" if image else None, image_gb=gb if image else None)

    rows = [
        row("a", "TOA", "Mainroad", pothole=100, gb=10),   # 10/GB
        row("b", "TOA", "Frontback", pothole=50, gb=1),    # 50/GB
        row("c", "TOA", "other", pothole=1000, gb=1),      # 대상 도로 아님
        row("d", "TOA", "Mainroad", pothole=999, image=False),  # 이미지 zip 없음
        row("e", "TOA", "Mainroad", pothole=30, gb=1),     # 30/GB
        row("f", "CRACK", "Mainroad", crack=5, gb=1),
        row("g", "CRACK", "Frontback", crack=20, gb=2),
        row("h", "CRACK", "Mainroad", crack=0, gb=0.1),    # 0개는 고르지 않는다
    ]
    picked = selection.pick_road_rows(rows, n_pothole=2, n_crack=3)
    assert [r["label"] for r in picked] == ["b", "e", "g", "f"]


def test_krid_count_table_pivots_states():
    c = Counter({"도로 표지|정상": 5, "도로 표지|파손": 1, "방호울타리|정상": 9, "_json_files": 3})
    states, rows = selection.krid_count_table(c)
    assert states == ["정상", "파손"]
    assert rows == [{"name": "방호울타리", "정상": 9, "파손": 0}, {"name": "도로 표지", "정상": 5, "파손": 1}]


# ---------------------------------------------------------------- aihub

class FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, body: bytes):
        super().__init__("u", 502, "Bad Gateway", {}, io.BytesIO(body))


def test_fetch_tree_accepts_502_with_valid_tree_body(monkeypatch, capsys):
    body = (aihub.SNAPSHOT_DIR / "71786.txt").read_bytes()

    def bad_gateway(*a, **k):
        raise FakeHTTPError(body)

    monkeypatch.setattr(aihub.urllib.request, "urlopen", bad_gateway)
    files = aihub.fetch_tree(71786, retries=1, wait_s=0, use_snapshot=False)
    assert "TL_2.국도_02.도시부.zip" in {f.name for f in files}
    assert "snapshot" not in capsys.readouterr().out


def test_fetch_tree_502_with_garbage_body_uses_snapshot(monkeypatch, capsys):
    monkeypatch.setattr(aihub.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(FakeHTTPError(b"<html>502</html>")))
    files = aihub.fetch_tree(179, retries=2, wait_s=0)
    assert files and "using snapshot" in capsys.readouterr().out
    with pytest.raises(RuntimeError):
        aihub.fetch_tree(179, retries=1, wait_s=0, use_snapshot=False)


def fake_curl(body: str, code: str):
    def run(cmd, input=None, capture_output=True, text=True):
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w") as f:
            f.write(body)
        assert SECRET not in " ".join(cmd)  # 키는 명령줄(프로세스 목록)에 넣지 않는다
        assert SECRET in input
        return subprocess.CompletedProcess(cmd, 0, stdout=code, stderr="")
    return run


def test_download_geo_block_message_never_leaks_key(monkeypatch, tmp_path):
    body = f"AI 허브는 해외에서의 데이터 다운로드를 제한하고 있습니다. key={SECRET}"
    monkeypatch.setattr(aihub.subprocess, "run", fake_curl(body, "502"))
    with pytest.raises(RuntimeError) as e:
        aihub.download(71786, AihubFile("a/x.zip", "1 MB", 42), SECRET, tmp_path)
    msg = str(e.value)
    assert SECRET not in msg
    assert "outside Korea" in msg and "prepare_local" in msg
    assert not list(tmp_path.rglob("*.tar"))  # 실패한 응답 파일은 지운다


def test_download_uses_per_key_dir_for_same_name_files(monkeypatch, tmp_path):
    import tarfile

    def run(cmd, input=None, **k):
        out = cmd[cmd.index("-o") + 1]
        key = cmd[-1].rsplit("=", 1)[1]
        src = tmp_path / f"src{key}" / "dup.zip"
        src.parent.mkdir()
        src.write_bytes(key.encode())
        with tarfile.open(out, "w") as tf:
            tf.add(src, arcname="dup.zip")
        return subprocess.CompletedProcess(cmd, 0, stdout="502", stderr="")  # 502 여도 tar 면 성공

    monkeypatch.setattr(aihub.subprocess, "run", run)
    a = aihub.download(179, AihubFile("Training/dup.zip", "1 KB", 1), SECRET, tmp_path / "dl")
    b = aihub.download(179, AihubFile("Validation/dup.zip", "1 KB", 2), SECRET, tmp_path / "dl")
    assert a != b and a.read_bytes() == b"1" and b.read_bytes() == b"2"


# ---------------------------------------------------------------- prepare_local

def test_load_apikey_from_dotenv_in_main_checkout_of_worktree(tmp_path, monkeypatch):
    monkeypatch.delenv("AIHUB_APIKEY", raising=False)
    main = tmp_path / "repo"
    (main / ".git" / "worktrees" / "wt").mkdir(parents=True)
    (main / ".env").write_text(f"OTHER=1\nexport AIHUB_APIKEY='{SECRET}'\n")
    wt = main / ".claude" / "worktrees" / "wt"
    (wt / "ai").mkdir(parents=True)
    (wt / ".git").write_text(f"gitdir: {main / '.git' / 'worktrees' / 'wt'}\n")
    assert prepare_local.load_apikey(wt / "ai") == SECRET
    monkeypatch.setenv("AIHUB_APIKEY", "from-env")
    assert prepare_local.load_apikey(wt) == "from-env"


def test_load_apikey_missing_message_has_no_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("AIHUB_APIKEY", raising=False)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".env").write_text("SOMETHING_ELSE=" + SECRET)
    with pytest.raises(SystemExit) as e:
        prepare_local.load_apikey(tmp_path)
    assert "AIHUB_APIKEY not set" in str(e.value) and SECRET not in str(e.value)


def test_coco_like_check(tmp_path):
    ok, bad = tmp_path / "ok.zip", tmp_path / "bad.zip"
    with zipfile.ZipFile(ok, "w") as z:
        z.writestr("a.json", json.dumps({"images": [], "annotations": [], "categories": []}))
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("a.json", json.dumps({"info": {}, "objects": []}))
    assert prepare_local.coco_like_problem(ok) is None
    assert "missing" in prepare_local.coco_like_problem(bad)
    problems = prepare_local.check_road_format({"TOA": ok, "CRACK": bad}, tmp_path / "s.txt")
    assert len(problems) == 1 and problems[0].startswith("CRACK")
    assert "objects" in (tmp_path / "s.txt").read_text()


def _jpg():
    return cv2.imencode(".jpg", np.full((720, 1280, 3), 128, np.uint8))[1].tobytes()


def test_convert_job_and_part_zip_layout(tmp_path):
    """02_train_detector 는 parts/*.zip 에서 det/ 로 시작하는 항목만 푼다."""
    lab, img = tmp_path / "TL.zip", tmp_path / "TS.zip"
    with zipfile.ZipFile(lab, "w") as lz, zipfile.ZipFile(img, "w") as iz:
        lz.writestr("a.json", json.dumps({
            "images": [{"id": 1, "file_name": "v_000001.jpg", "width": 1280, "height": 720}],
            "annotations": [{"name": "방호울타리", "state": "파손", "bbox": [100, 100, 300, 80]}],
        }, ensure_ascii=False))
        iz.writestr("img/v_000001.jpg", _jpg())
    job = selection.Job("krid", 71786, AihubFile("TS.zip", "1 MB", 7), AihubFile("TL.zip", "1 KB", 8))
    stats = prepare_local.convert_job(job, lab, img, tmp_path / "work" / job.part)
    assert stats["written"] == 1
    road_lab, road_img = tmp_path / "rl.zip", tmp_path / "ri.zip"
    with zipfile.ZipFile(road_lab, "w") as lz, zipfile.ZipFile(road_img, "w") as iz:
        lz.writestr("a.json", json.dumps({
            "images": [{"id": 1, "file_name": "r_000000.png", "width": 1280, "height": 720}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 200, 200]}],
            "categories": [{"id": 1, "name": "포트홀"}],
        }, ensure_ascii=False))
        iz.writestr("r_000000.png", cv2.imencode(".png", np.zeros((720, 1280, 3), np.uint8))[1].tobytes())
    rjob = selection.Job("road", 179, AihubFile("ri.zip", "1 MB", 9), AihubFile("rl.zip", "1 KB", 10))
    prepare_local.convert_job(rjob, road_lab, road_img, tmp_path / "work" / rjob.part)

    dest = tmp_path / "krid.zip"
    zstats = prepare_local.build_part_zip([tmp_path / "work" / job.part, tmp_path / "work" / rjob.part], dest)
    names = zipfile.ZipFile(dest).namelist()
    assert any(n.startswith("det/images/") and n.endswith("krid_v_000001.jpg") for n in names)
    assert any(n.startswith("det/labels/") and n.endswith("ro_r_000000.txt") for n in names)
    assert any(n.startswith("cls/") for n in names)
    assert sum(v for k, v in zstats.items() if k.startswith("det_images_")) == 2
    txt = next(n for n in names if n.endswith("ro_r_000000.txt"))
    assert int(zipfile.ZipFile(dest).read(txt).split()[0]) == POTHOLE != CRACK


def test_parse_road_obstacle_real_format_dict_images_and_polyline():
    """179 실제 형식: images 가 객체 하나, 크랙은 bbox 없이 polyline (2026-11-14 받은 라벨에서 확인)."""
    from daboyeo_ai.data.sources import parse_coco_like

    js = {
        "info": {"contributor": "Konkuk_university"},
        "images": {"file_name": "V4F_x_07322.png", "height": 720, "width": 1280, "id": 1},
        "annotations": [
            {"segmentation": [], "polyline": [[1070.3, 668.5, 1036.7, 646.0, 942.0, 593.3]],
             "image_id": 1, "bbox": [], "category_id": 1, "area": 0, "id": 1},
        ],
        "categories": [{"id": 1, "name": "Crack"}],
    }
    [li] = parse_coco_like(js)
    assert li.file_name == "V4F_x_07322.png" and (li.width, li.height) == (1280, 720)
    [b] = li.boxes
    assert b.cls == CRACK and (b.x1, b.y1, b.x2, b.y2) == pytest.approx((942.0, 593.3, 1070.3, 668.5))


def test_coco_like_check_catches_unparseable(tmp_path):
    z = tmp_path / "x.zip"
    with zipfile.ZipFile(z, "w") as zf:  # images 객체에 id 가 없다 → 파싱 실패를 보고해야 한다
        zf.writestr("a.json", json.dumps({"images": {"file_name": "a.png"}, "annotations": [], "categories": []}))
    assert "parse_coco_like failed" in prepare_local.coco_like_problem(z)
