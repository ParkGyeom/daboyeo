"""zip 단위 변환 파이프라인. 노트북은 이 함수들을 zip 하나씩 호출한다.

원본 zip 을 통째로 풀지 않고, 라벨이 있는 이미지만 zip 에서 바로 읽어 변환한다
(Colab 디스크에 원본 + 압축 해제본이 동시에 올라가지 않게).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2

from .sources import (
    CRACK,
    ROAD_NOMINAL_SIZE,
    RDD_DEFAULT_COUNTRIES,
    count_coco_like,
    count_krid,
    parse_coco_like,
    parse_krid,
    parse_rdd_xml,
)
from .yolo import CrackMerge, CropWriter, YoloDetWriter, decode_image, frame_group, mask_boxes


def _stem(name: str) -> str:
    return PurePosixPath(name).stem


def _index_by_stem(zf: zipfile.ZipFile, exts: tuple[str, ...]) -> dict[str, str]:
    return {
        _stem(n): n for n in zf.namelist() if n.lower().endswith(exts) and not n.endswith("/")
    }


def _read_json(zf: zipfile.ZipFile, name: str) -> dict:
    raw = zf.read(name)
    for enc in ("utf-8-sig", "cp949"):
        try:
            return json.loads(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {name}")


def _group(stem: str, chunk: int) -> tuple[str, int | None]:
    """프레임 번호를 chunk 단위로 묶어 split 그룹을 만든다 (같은 구간은 같은 split)."""
    g, idx = frame_group(stem)
    return (f"{g}#{idx // chunk}" if idx is not None else g), idx


# ---------------------------------------------------------------- 라벨 집계

def count_label_zip(label_zip: Path, kind: str) -> Counter[str]:
    """kind: 'krid' 또는 'coco'. 라벨 zip 하나의 클래스별 개수."""
    total: Counter[str] = Counter()
    with zipfile.ZipFile(label_zip) as zf:
        for n in zf.namelist():
            if not n.lower().endswith(".json"):
                continue
            js = _read_json(zf, n)
            total += count_krid(js) if kind == "krid" else count_coco_like(js)
            total["_json_files"] += 1
    return total


def peek_label_zip(label_zip: Path, max_chars: int = 3000) -> str:
    """첫 JSON 의 구조를 사람이 볼 수 있게 요약한다 (형식 확인용)."""
    with zipfile.ZipFile(label_zip) as zf:
        names = zf.namelist()
        jsons = [n for n in names if n.lower().endswith(".json")]
        others = Counter(PurePosixPath(n).suffix.lower() for n in names if not n.endswith("/"))
        head = f"members={len(names)} by_ext={dict(others)}\nfirst_json={jsons[0] if jsons else None}\n"
        if not jsons:
            return head
        js = _read_json(zf, jsons[0])
    summary = {k: (v[:2] if isinstance(v, list) and k != "categories" else v) for k, v in js.items()}  # 카테고리는 전부
    return head + json.dumps(summary, ensure_ascii=False, indent=1)[:max_chars]


# ---------------------------------------------------------------- KRID (전국 도로시설물 영상정보)

def process_krid(
    label_zip: Path,
    image_zip: Path,
    det: YoloDetWriter,
    crops: CropWriter | None = None,
    *,
    stride: int = 1,
    chunk: int = 200,
    max_images: int | None = None,
    include_median: bool = False,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    with zipfile.ZipFile(label_zip) as lz, zipfile.ZipFile(image_zip) as iz:
        images = _index_by_stem(iz, (".jpg", ".jpeg", ".png"))
        for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
            li = parse_krid(_read_json(lz, n), include_median=include_median)
            if not li.boxes:
                stats["skip_no_target"] += 1
                continue
            stem = _stem(li.file_name)
            group, idx = _group(stem, chunk)
            has_damage = any(d for _, d in li.facilities)
            # 파손은 드물어서 프레임 간격(stride)으로 버리지 않는다
            if stride > 1 and idx is not None and idx % stride and not has_damage:
                stats["skip_stride"] += 1
                continue
            member = images.get(stem)
            if member is None:
                stats["skip_image_missing"] += 1
                continue
            img = decode_image(iz.read(member))
            if img is None:
                stats["skip_decode"] += 1
                continue
            stats["written_with_damage"] += has_damage
            det.write(f"krid_{stem}", img, li.boxes, group)
            if crops is not None:
                for k, (box, damaged) in enumerate(li.facilities):
                    crops.write(f"krid_{stem}_{k}", img, box, "damaged" if damaged else "normal", group)
            stats["written"] += 1
            if max_images and stats["written"] >= max_images:
                break
    return stats


# ---------------------------------------------------------------- 도로장애물/표면 인지 영상

def process_road_obstacle(
    label_zip: Path,
    image_zip: Path,
    det: YoloDetWriter,
    *,
    stride: int = 5,
    chunk: int = 300,
    max_images: int | None = None,
    negative_ratio: float = 0.1,
    min_mask_area: int = 64,
    crack_merge: CrackMerge | None = None,
) -> Counter[str]:
    """포트홀(TOA) 또는 크랙(CRACK) zip 한 쌍을 변환한다.

    크랙은 JSON 에 좌표가 없고 마스크 PNG 로 오는 경우가 있어, 박스가 없으면 같은 이름의
    마스크를 라벨 zip 에서 찾아 연결 영역별 박스로 만든다.
    crack_merge: None 이면 v2 규칙 (어노테이션마다 박스), 주면 v3 규칙 (`sources.parse_coco_like`).
    """
    stats: Counter[str] = Counter()
    with zipfile.ZipFile(label_zip) as lz, zipfile.ZipFile(image_zip) as iz:
        images = _index_by_stem(iz, (".png", ".jpg", ".jpeg"))
        masks = _index_by_stem(lz, (".png",))
        labeled: list = []
        for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
            labeled.extend(parse_coco_like(_read_json(lz, n), stats, crack_merge))
        positives = 0
        for li in labeled:
            stem = _stem(li.file_name)
            group, idx = _group(stem, chunk)
            if stride > 1 and idx is not None and idx % stride:
                stats["skip_stride"] += 1
                continue
            boxes = li.boxes
            if not boxes and not li.hard_negative and stem in masks:
                mask = decode_image(lz.read(masks[stem]))
                if mask is not None:
                    boxes = mask_boxes(CRACK, mask[..., 0] if mask.ndim == 3 else mask, min_mask_area)
                    stats["crack_from_mask"] += bool(boxes)
            if not boxes:
                if not li.hard_negative or stats["written_negative"] >= negative_ratio * max(positives, 1):
                    stats["skip_no_target"] += 1
                    continue
            member = images.get(stem)
            if member is None:
                stats["skip_image_missing"] += 1
                continue
            img = decode_image(iz.read(member))
            if img is None:
                stats["skip_decode"] += 1
                continue
            det.write(f"ro_{stem}", img, boxes, group)
            if boxes:
                positives += 1
                stats["written"] += 1
            else:
                stats["written_negative"] += 1
            if max_images and stats["written"] >= max_images:
                break
    return stats


def label_stems(label_zip: Path) -> set[str]:
    """라벨 zip 의 JSON 이름(`..._PLINE` / `..._BBOX` 를 뗀 것) = 짝 이미지 이름."""
    with zipfile.ZipFile(label_zip) as zf:
        return {_stem(n).removesuffix("_PLINE").removesuffix("_BBOX") for n in zf.namelist()
                if n.lower().endswith(".json")}


def process_crack_on_toa(
    crack_label_zips: list[Path],
    toa_label_zip: Path,
    toa_image_zip: Path,
    det: YoloDetWriter,
    *,
    crack_merge: CrackMerge | None = CrackMerge(),
    stride: int = 2,
    chunk: int = 300,
    max_images: int | None = None,
) -> Counter[str]:
    """도로장애물(179) 크랙 라벨을 **포트홀(TOA) 이미지 zip 의 같은 이름 프레임**에 붙인다 (v3, 2026-11-15).

    179 의 크랙 이미지 zip(`2.Images_new_211012/2.CRACK/`)은 라벨과 다른 장면이 들어 있다 (같은 이름의 TOA·CRACK 이미지
    215쌍이 전부 다른 사진이고, 크랙 선은 TOA 쪽 사진에 맞는다 — `docs/ai-handoff.md` 15장). 그래서 크랙 라벨 zip 들과
    TOA 라벨·이미지 zip 한 쌍을 받아, TOA 이미지 중 크랙 라벨이 있는 프레임만 쓴다. 박스 = 크랙(crack_merge 규칙) +
    같은 프레임의 TOA 포트홀 박스. 이름은 TOA 변환과 같은 `ro_<stem>` 이라 parts zip 을 만들 때 이 폴더를 먼저 넣으면
    TOA 쪽(포트홀만 있는 라벨) 사본을 덮는다. 프레임 번호 % stride 로 줄인다 (번호는 영상 안 위치라 사실상 무작위 추출).
    """
    stats: Counter[str] = Counter()
    crack: dict[str, list] = {}
    for lz_path in crack_label_zips:
        with zipfile.ZipFile(lz_path) as lz:
            for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
                for li in parse_coco_like(_read_json(lz, n), stats, crack_merge):
                    if li.boxes:
                        crack[_stem(li.file_name)] = li
    with zipfile.ZipFile(toa_label_zip) as lz, zipfile.ZipFile(toa_image_zip) as iz:
        images = _index_by_stem(iz, (".png", ".jpg", ".jpeg"))
        toa: dict[str, list] = {}
        for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
            for li in parse_coco_like(_read_json(lz, n)):
                toa[_stem(li.file_name)] = li.boxes
        for stem in sorted(crack):
            if stem not in images:
                stats["skip_not_in_toa_images"] += 1
                continue
            group, idx = _group(stem, chunk)
            if stride > 1 and idx is not None and idx % stride:
                stats["skip_stride"] += 1
                continue
            img = decode_image(iz.read(images[stem]))
            if img is None:
                stats["skip_decode"] += 1
                continue
            li = crack[stem]
            h, w = img.shape[:2]
            if (li.width, li.height) not in ((w, h), (0, 0)):
                stats["size_mismatch"] += 1  # 라벨 크기와 실제 이미지 크기가 다르면 맞는 프레임이 아니다
                continue
            pothole = [b for b in toa.get(stem, []) if b.cls != CRACK]
            stats["with_toa_pothole"] += bool(pothole)
            det.write(f"ro_{stem}", img, li.boxes + pothole, group)
            stats["written"] += 1
            if max_images and stats["written"] >= max_images:
                break
    return stats


def relabel_road_obstacle(label_zip: Path, src_root: Path, det: YoloDetWriter,
                          crack_merge: CrackMerge | None) -> Counter[str]:
    """이미 변환한 도로장애물 작업 폴더(src_root = .../det)의 이미지는 그대로 두고 라벨만 새 규칙으로 다시 쓴다.

    원본 이미지 zip 을 다시 받지 않으려고 쓴다 (AI Hub 는 이 Mac 에서만 받을 수 있고 zip 이 10 GB 안팎).
    고르는 이미지(stride·max_images)는 원래 변환과 같고, 이미지는 det 쪽에 하드링크(안 되면 복사)한다.
    라벨 JSON 이 없는 이미지는 건너뛴다 (`relabel_no_json`). 마스크 PNG 로 박스를 만든 이미지는 없다고 본다
    (179 실제 크랙은 polyline).
    """
    stats: Counter[str] = Counter()
    by_stem = {}
    with zipfile.ZipFile(label_zip) as lz:
        for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
            for li in parse_coco_like(_read_json(lz, n), stats, crack_merge):
                by_stem[_stem(li.file_name)] = li
    for img_path in sorted(Path(src_root).glob("images/*/ro_*.jpg")):
        split, name = img_path.parent.name, img_path.stem
        li = by_stem.get(name.removeprefix("ro_"))
        if li is None:
            stats["relabel_no_json"] += 1
            continue
        w, h = li.width, li.height
        if w <= 0 or h <= 0:  # 라벨 크기 0: 저장 이미지가 640x360 이면 원본 1280x720 으로 본다, 아니면 건너뜀
            small = cv2.imread(str(img_path))
            if small is None or small.shape[:2] != (360, 640):
                stats["relabel_bad_size"] += 1
                continue
            w, h = ROAD_NOMINAL_SIZE
            stats["relabel_size_assumed"] += 1
        dst = det.root / "images" / split / img_path.name
        dst.unlink(missing_ok=True)
        try:
            dst.hardlink_to(img_path)
        except OSError:
            shutil.copy2(img_path, dst)
        det.write_label(name, split, li.boxes, w, h, det.scale_for(w, h))
        stats["written"] += 1
    return stats


# ---------------------------------------------------------------- RDD2022

def _process_rdd_zip(
    zf: zipfile.ZipFile, det: YoloDetWriter, countries: tuple[str, ...], stats: Counter[str],
    forced_country: str | None = None,
) -> None:
    names = zf.namelist()
    images = {n for n in names if "train/images/" in n}
    for n in names:
        if "train/annotations/xmls/" not in n or not n.endswith(".xml"):
            continue
        country = next((c for c in countries if f"{c}/train/" in n), forced_country)
        if country is None:
            stats["skip_country"] += 1
            continue
        li = parse_rdd_xml(zf.read(n).decode("utf-8", errors="replace"))
        img_dir = n.replace("annotations/xmls/", "images/").rsplit("/", 1)[0]
        candidates = [f"{img_dir}/{li.file_name}", f"{img_dir}/{_stem(n)}.jpg"]
        img_member = next((c for c in candidates if c in images), None)
        if img_member is None:
            stats["skip_image_missing"] += 1
            continue
        img = decode_image(zf.read(img_member))
        if img is None:
            stats["skip_decode"] += 1
            continue
        stem = _stem(img_member)
        det.write(f"rdd_{stem}", img, li.boxes, group=stem)  # RDD 는 개별 사진이라 이름 자체가 그룹
        stats[f"written_{country}"] += 1


def process_rdd2022(zip_path: Path, det: YoloDetWriter, countries: tuple[str, ...] = RDD_DEFAULT_COUNTRIES) -> Counter[str]:
    """RDD2022 zip 변환. 바깥 zip 안에 나라별 zip 이 들어 있는 경우도 처리한다."""
    stats: Counter[str] = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        _process_rdd_zip(zf, det, countries, stats)
        for n in zf.namelist():
            country = next((c for c in countries if c in n), None)
            if not n.lower().endswith(".zip") or country is None:
                continue
            with tempfile.TemporaryDirectory() as td:
                p = Path(zf.extract(n, td))
                with zipfile.ZipFile(p) as iz:
                    _process_rdd_zip(iz, det, countries, stats, forced_country=country)
    return stats
