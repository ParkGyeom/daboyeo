"""zip 단위 변환 파이프라인. 노트북은 이 함수들을 zip 하나씩 호출한다.

원본 zip 을 통째로 풀지 않고, 라벨이 있는 이미지만 zip 에서 바로 읽어 변환한다
(Colab 디스크에 원본 + 압축 해제본이 동시에 올라가지 않게).
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .sources import (
    CRACK,
    RDD_DEFAULT_COUNTRIES,
    count_coco_like,
    count_krid,
    parse_coco_like,
    parse_krid,
    parse_rdd_xml,
)
from .yolo import CropWriter, YoloDetWriter, decode_image, frame_group, mask_boxes


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
) -> Counter[str]:
    """포트홀(TOA) 또는 크랙(CRACK) zip 한 쌍을 변환한다.

    크랙은 JSON 에 좌표가 없고 마스크 PNG 로 오는 경우가 있어, 박스가 없으면 같은 이름의
    마스크를 라벨 zip 에서 찾아 연결 영역별 박스로 만든다.
    """
    stats: Counter[str] = Counter()
    with zipfile.ZipFile(label_zip) as lz, zipfile.ZipFile(image_zip) as iz:
        images = _index_by_stem(iz, (".png", ".jpg", ".jpeg"))
        masks = _index_by_stem(lz, (".png",))
        labeled: list = []
        for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
            labeled.extend(parse_coco_like(_read_json(lz, n)))
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
