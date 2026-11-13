"""데이터셋별 라벨 파서 → 우리 클래스(`CLASS_NAMES`)의 Box 로 변환.

| 우리 클래스 | 전국 도로시설물(KRID) | 도로장애물/표면 | RDD2022 |
|---|---|---|---|
| pothole   | —                               | 포트홀            | D40 |
| crack     | —                               | 크랙 (마스크)      | D00·D10·D20 |
| guardrail | 방호울타리 (+선택: 중앙분리대)       | —                 | — |
| sign      | 도로 표지·안전 표지·도로이정표·도로명판 | —                 | — |
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field

from daboyeo_contracts import CLASS_NAMES

from .yolo import Box, bbox_xywh, polygon_box

POTHOLE, CRACK, GUARDRAIL, SIGN = (CLASS_NAMES.index(n) for n in ("pothole", "crack", "guardrail", "sign"))


def _norm(s: str) -> str:
    return "".join(s.split()).lower()


@dataclass
class LabeledImage:
    file_name: str
    width: int
    height: int
    boxes: list[Box] = field(default_factory=list)
    # 시설물 파손 분류용: (box, damaged)
    facilities: list[tuple[Box, bool]] = field(default_factory=list)
    hard_negative: bool = False  # 보수완료 포트홀·맨홀만 있는 이미지 등


# ---------------------------------------------------------------- KRID (전국 도로시설물 영상정보)

KRID_SIGN = {_norm(n) for n in ("도로 표지", "안전 표지", "도로이정표", "도로명판")}
KRID_GUARDRAIL = {_norm("방호울타리")}
KRID_MEDIAN = {_norm("중앙분리대")}


def parse_krid(js: dict, include_median: bool = False) -> LabeledImage:
    """KRID 라벨 JSON(이미지 1장) 파싱. 형식은 AI Hub 데이터셋 페이지(dataSetSn=71786)의 예시를 따른다."""
    for key in ("images", "annotations"):
        if key not in js:
            raise ValueError(f"KRID label missing '{key}'; keys={sorted(js)}")
    img = js["images"][0]
    out = LabeledImage(img["file_name"], int(img["width"]), int(img["height"]))
    guard = KRID_GUARDRAIL | (KRID_MEDIAN if include_median else set())
    for a in js["annotations"]:
        name = _norm(a.get("name", ""))
        cls = SIGN if name in KRID_SIGN else GUARDRAIL if name in guard else None
        if cls is None:
            continue
        box = bbox_xywh(cls, a["bbox"]) if a.get("bbox") else polygon_box(cls, a.get("segmentation") or [])
        if box is None:
            continue
        out.boxes.append(box)
        out.facilities.append((box, a.get("state") == "파손"))
    return out


def count_krid(js: dict) -> Counter[str]:
    c: Counter[str] = Counter()
    for a in js.get("annotations", []):
        c[f"{a.get('name', '?')}|{a.get('state', '?')}"] += 1
    return c


# ---------------------------------------------------------------- 도로장애물/표면 인지 영상

def road_obstacle_class(category_name: str) -> int | str | None:
    """카테고리 이름 → POTHOLE / CRACK / 'negative'(보수완료 포트홀·맨홀) / None."""
    n = _norm(category_name)
    if any(k in n for k in ("보수", "fill", "repair")) and any(k in n for k in ("포트홀", "pothole")):
        return "negative"
    if "맨홀" in n or "manhole" in n:
        return "negative"
    if "포트홀" in n or "pothole" in n:
        return POTHOLE
    if any(k in n for k in ("크랙", "crack", "균열")):
        return CRACK
    return None


def parse_coco_like(js: dict) -> list[LabeledImage]:
    """COCO 비슷한 형식(images / annotations / categories) 라벨을 파싱한다.

    도로장애물(179) 실제 형식 (2026-11-14 확인): JSON 하나에 이미지 1장이라 `images` 가 리스트가 아니라
    객체 하나이고, 크랙은 `bbox`·`segmentation` 이 비어 있고 `polyline` 에 점 좌표가 온다.
    둘 다 받는다. 키가 없으면 바로 실패하게 했다.
    """
    missing = [k for k in ("images", "annotations", "categories") if k not in js]
    if missing:
        raise ValueError(f"not COCO-like: missing {missing}; top-level keys={sorted(js)}")
    images = js["images"] if isinstance(js["images"], list) else [js["images"]]
    cats = {c["id"]: road_obstacle_class(c["name"]) for c in js["categories"]}
    by_id = {im["id"]: LabeledImage(im["file_name"], int(im["width"]), int(im["height"])) for im in images}
    negatives: set = set()
    for a in js["annotations"]:
        cls = cats.get(a["category_id"])
        img = by_id.get(a["image_id"])
        if cls is None or img is None:
            continue
        if cls == "negative":
            negatives.add(a["image_id"])
            continue
        if a.get("bbox"):
            box = bbox_xywh(cls, a["bbox"])
        else:
            box = polygon_box(cls, a.get("segmentation") or a.get("polyline") or [])
        if box is not None:
            img.boxes.append(box)
    for i in negatives:
        if not by_id[i].boxes:
            by_id[i].hard_negative = True
    return list(by_id.values())


def count_coco_like(js: dict) -> Counter[str]:
    cats = {c["id"]: c["name"] for c in js.get("categories", [])}
    return Counter(cats.get(a.get("category_id"), "?") for a in js.get("annotations", []))


# ---------------------------------------------------------------- RDD2022 (Pascal VOC XML)

RDD_MAP = {"D00": CRACK, "D10": CRACK, "D20": CRACK, "D40": POTHOLE}
# China_Drone 은 항공 시점이라 블랙박스 화각과 달라 뺀다. 테스트 split 은 라벨이 없다.
RDD_DEFAULT_COUNTRIES = ("Japan", "Czech", "India", "United_States", "China_MotorBike", "Norway")


def parse_rdd_xml(xml_text: str) -> LabeledImage:
    root = ET.fromstring(xml_text)
    size = root.find("size")
    out = LabeledImage(root.findtext("filename", ""), int(size.findtext("width")), int(size.findtext("height")))
    for obj in root.iter("object"):
        cls = RDD_MAP.get((obj.findtext("name") or "").strip())
        bb = obj.find("bndbox")
        if cls is None or bb is None:
            continue
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.boxes.append(Box(cls, x1, y1, x2, y2))
    return out
