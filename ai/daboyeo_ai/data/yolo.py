"""YOLO 형식 데이터셋 쓰기 (탐지 + 파손 분류 크롭).

탐지 데이터셋 구조 (Ultralytics 기본):
    root/images/{train,val,test}/<name>.jpg
    root/labels/{train,val,test}/<name>.txt   # "cls cx cy w h" (0~1)
    root/data.yaml

분류 데이터셋 구조 (Ultralytics classify):
    root/{train,val,test}/{normal,damaged}/<name>.jpg

train/val/test 는 `group`(같은 영상·같은 구간) 단위로 나눈다. 이웃 프레임이 서로 다른 split 에
들어가면 검증 점수가 부풀려지기 때문이다.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def clipped(self, w: int, h: int) -> Box:
        return Box(self.cls, max(0.0, self.x1), max(0.0, self.y1), min(float(w), self.x2), min(float(h), self.y2))

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def bbox_xywh(cls: int, bbox: list[float]) -> Box:
    x, y, w, h = bbox
    return Box(cls, x, y, x + w, y + h)


def polygon_box(cls: int, seg: list) -> Box | None:
    """세그멘테이션 좌표 → 외접 박스. `[x1,y1,x2,y2,...]` 와 `[[x1,y1,...], ...]` 둘 다 받는다."""
    flat: list[float] = []
    for item in seg:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    if len(flat) < 4:
        return None
    xs, ys = flat[0::2], flat[1::2]
    return Box(cls, min(xs), min(ys), max(xs), max(ys))


def mask_boxes(cls: int, mask: np.ndarray, min_area: int = 64) -> list[Box]:
    """이진 마스크(균열 등) → 연결 영역마다 박스 하나."""
    binary = (mask > 0).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append(Box(cls, float(x), float(y), float(x + w), float(y + h)))
    return boxes


def split_for(group: str, salt: str = "daboyeo", ratios: tuple[int, int] = (80, 90)) -> str:
    """group 문자열 해시로 train(80%)/val(10%)/test(10%) 를 고정 배정한다."""
    v = int(hashlib.sha1(f"{salt}:{group}".encode()).hexdigest(), 16) % 100
    return "train" if v < ratios[0] else "val" if v < ratios[1] else "test"


_FRAME_RE = re.compile(r"^(?P<group>.*?)[_-](?P<idx>\d+)$")


def frame_group(stem: str) -> tuple[str, int | None]:
    """`..._0005` 처럼 끝이 프레임 번호면 (영상 이름, 번호), 아니면 (stem, None)."""
    m = _FRAME_RE.match(stem)
    return (m["group"], int(m["idx"])) if m else (stem, None)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def decode_image(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class YoloDetWriter:
    def __init__(self, root: Path, max_side: int = 640, min_box_px: float = 4.0, jpeg_quality: int = 90):
        self.root = Path(root)
        self.max_side = max_side
        self.min_box_px = min_box_px
        self.jpeg_quality = jpeg_quality
        self.stats: Counter[str] = Counter()
        for s in SPLITS:
            (self.root / "images" / s).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / s).mkdir(parents=True, exist_ok=True)

    def write(self, name: str, img: np.ndarray, boxes: list[Box], group: str) -> str:
        """이미지 1장을 쓴다. boxes 가 비어 있으면 배경 이미지(빈 라벨)로 쓴다. 돌려주는 값은 split."""
        h, w = img.shape[:2]
        small = resize_max(img, self.max_side)
        scale = small.shape[1] / w
        lines = []
        for b in boxes:
            b = b.clipped(w, h)
            if b.width * scale < self.min_box_px or b.height * scale < self.min_box_px:
                self.stats["box_dropped_small"] += 1
                continue
            cx, cy = (b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {b.width / w:.6f} {b.height / h:.6f}")
            self.stats[f"box_{CLASS_NAMES[b.cls]}"] += 1
        split = split_for(group)
        cv2.imwrite(str(self.root / "images" / split / f"{name}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        (self.root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        self.stats[f"img_{split}"] += 1
        if not lines:
            self.stats["img_background"] += 1
        return split


class CropWriter:
    """시설물 파손 분류용 크롭. label 은 'normal' 또는 'damaged'."""

    def __init__(self, root: Path, size: int = 224, pad: float = 0.1, min_side_px: int = 24):
        self.root = Path(root)
        self.size, self.pad, self.min_side_px = size, pad, min_side_px
        self.stats: Counter[str] = Counter()

    def write(self, name: str, img: np.ndarray, box: Box, label: str, group: str) -> bool:
        h, w = img.shape[:2]
        if min(box.width, box.height) < self.min_side_px:
            self.stats["crop_dropped_small"] += 1
            return False
        px, py = box.width * self.pad, box.height * self.pad
        b = Box(box.cls, box.x1 - px, box.y1 - py, box.x2 + px, box.y2 + py).clipped(w, h)
        crop = img[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
        if crop.size == 0:
            return False
        split = split_for(group)
        out = self.root / split / label
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{name}.jpg"), resize_max(crop, self.size))
        self.stats[f"crop_{CLASS_NAMES[box.cls]}_{label}"] += 1
        return True


def cap_background(det_root: Path, ratio: float = 0.1, split: str = "train", salt: str = "daboyeo") -> dict[str, int]:
    """split 의 배경 이미지(빈 라벨)를 전체의 ratio 이하로 줄인다 (넘치는 것은 이미지·라벨 파일 삭제).

    어떤 배경을 남길지는 파일 이름 해시로 정해서, 다시 실행해도 같은 결과가 나온다.
    val/test 는 건드리지 않는다 (평가는 실제 분포 그대로).
    """
    det_root = Path(det_root)
    labels = sorted((det_root / "labels" / split).glob("*.txt"))
    bg = [p for p in labels if not p.read_text().strip()]
    n_pos = len(labels) - len(bg)
    keep = int(ratio / (1 - ratio) * n_pos) if ratio < 1 else len(bg)
    bg.sort(key=lambda p: hashlib.sha1(f"{salt}:{p.stem}".encode()).hexdigest())
    removed = 0
    for p in bg[keep:]:
        p.unlink()
        (det_root / "images" / split / f"{p.stem}.jpg").unlink(missing_ok=True)
        removed += 1
    for cache in (det_root / "labels").glob("*.cache"):  # Ultralytics 라벨 캐시는 다시 만들게 한다
        cache.unlink()
    return {"positive": n_pos, "background_before": len(bg), "background_kept": len(bg) - removed, "removed": removed}


def write_data_yaml(root: Path) -> Path:
    root = Path(root)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    p = root / "data.yaml"
    p.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    return p
