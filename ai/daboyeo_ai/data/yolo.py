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
import math
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


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def valid_points(seg: list) -> tuple[list[tuple[float, float]], int]:
    """세그멘테이션/폴리라인 좌표 → ((x, y) 점들, 버린 점 수).

    `[x1,y1,x2,y2,...]` 와 `[[x1,y1,...], ...]` 둘 다 받는다. 실제 179 크랙 라벨에는 `[None, None, ...]`
    로 시작하는 polyline 이 있어서 (2026-11-14, C_Frontback_B01 에서 179,518 개 중 3개), x·y 가 둘 다
    유한한 숫자인 점만 남긴다. 짝이 안 맞는 남는 좌표 하나도 버린다.
    """
    rings = seg if any(isinstance(r, (list, tuple)) for r in seg) else [seg]
    points: list[tuple[float, float]] = []
    dropped = 0
    for ring in rings:
        if not isinstance(ring, (list, tuple)):
            dropped += 1
            continue
        for i in range(0, len(ring) - 1, 2):
            x, y = ring[i], ring[i + 1]
            if _num(x) and _num(y):
                points.append((float(x), float(y)))
            else:
                dropped += 1
        dropped += len(ring) % 2
    return points, dropped


def polygon_box(cls: int, seg: list) -> Box | None:
    """세그멘테이션 좌표 → 외접 박스. 잘못된 점은 버리고, 유효한 점이 2개 미만이면 None."""
    points, _ = valid_points(seg or [])
    if len(points) < 2:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return Box(cls, min(xs), min(ys), max(xs), max(ys))


def valid_rings(seg: list) -> tuple[list[list[tuple[float, float]]], int]:
    """`valid_points` 와 같지만 선(ring)별로 따로 돌려준다. 점이 2개 미만인 선은 버린다."""
    rings = seg if any(isinstance(r, (list, tuple)) for r in seg) else [seg]
    out, dropped = [], 0
    for ring in rings:
        pts, d = valid_points(ring if isinstance(ring, (list, tuple)) else [ring])
        dropped += d
        if len(pts) >= 2:
            out.append(pts)
    return out, dropped


@dataclass(frozen=True)
class CrackMerge:
    """179 크랙 폴리라인 → 박스 규칙 (v3, 2026-11-15). 길이 단위는 **긴 변 ref_side 로 줄인 학습 이미지의 px**.

    179 크랙은 크랙 하나를 짧은 선분 여러 개(중앙값 24 px, 이미지당 19개)로 그려서, 선분마다 박스를 만들면
    가늘고 작은 박스 수백 개가 되고 36% 는 4 px 미만이라 버려진다. 그래서:
    1. 선을 merge_px 두께로 그려 서로 닿는(= merge_px 안쪽으로 가까운) 선끼리 한 묶음으로 본다.
    2. 묶음마다 원래 점들의 외접 박스 하나 (그린 두께만큼 부풀리지 않는다).
    3. 묶음의 선 길이 합이 min_len_px 미만이면 버린다 (사람도 못 알아볼 조각).
    4. 짧은 변을 min_thick_px 까지 가운데 기준으로 늘린다 (가는 세로·가로 크랙도 박스가 되게).
    5. 길이 합이 긴 순서로 max_boxes 개까지.
    """

    merge_px: float = 16
    min_len_px: float = 24
    min_thick_px: float = 12
    max_boxes: int = 20
    ref_side: int = 640


def merge_polylines(cls: int, polylines: list[list[tuple[float, float]]], width: int, height: int,
                    rule: CrackMerge = CrackMerge(), stats: Counter[str] | None = None) -> list[Box]:
    """한 이미지의 폴리라인들(원본 px) → 묶음별 박스(원본 px). 규칙은 `CrackMerge`."""
    stats = stats if stats is not None else Counter()
    lines = [np.asarray(p, dtype=np.float64) for p in polylines if len(p) >= 2]
    if not lines:
        return []
    s = min(1.0, rule.ref_side / max(width, height)) if rule.ref_side else 1.0
    w, h = max(1, round(width * s)), max(1, round(height * s))
    mask = np.zeros((h, w), np.uint8)
    thick = max(1, int(round(rule.merge_px)))
    scaled = [p * s for p in lines]
    for p in scaled:
        cv2.polylines(mask, [p.round().astype(np.int32)], False, 255, thickness=thick)
    _, lab = cv2.connectedComponents(mask, connectivity=8)
    groups: dict[int, list[np.ndarray]] = {}
    for p in scaled:
        x, y = np.clip(p[0].round().astype(int), 0, [w - 1, h - 1])
        groups.setdefault(int(lab[y, x]), []).append(p)  # 선은 이어 그렸으니 첫 점의 묶음이 선 전체의 묶음
    cands = []
    for ps in groups.values():
        length = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in ps)
        if length < rule.min_len_px:
            stats["crack_group_short"] += 1
            continue
        pts = np.concatenate(ps)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        cands.append((length, [x1, y1, x2, y2]))
    cands.sort(key=lambda c: -c[0])
    if len(cands) > rule.max_boxes:
        stats["crack_group_capped"] += len(cands) - rule.max_boxes
        cands = cands[:rule.max_boxes]
    out = []
    for _, (x1, y1, x2, y2) in cands:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw, hh = max(x2 - x1, rule.min_thick_px) / 2, max(y2 - y1, rule.min_thick_px) / 2
        out.append(Box(cls, (cx - hw) / s, (cy - hh) / s, (cx + hw) / s, (cy + hh) / s).clipped(width, height))
    stats["crack_polylines"] += len(lines)
    stats["crack_merged_boxes"] += len(out)
    return out


def valid_bbox(bbox) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(_num(v) for v in bbox)


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
        split = split_for(group)
        cv2.imwrite(str(self.root / "images" / split / f"{name}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        self.write_label(name, split, boxes, w, h, small.shape[1] / w)
        return split

    def write_label(self, name: str, split: str, boxes: list[Box], w: int, h: int, scale: float) -> int:
        """라벨 파일만 쓴다 (원본 크기 w·h, 저장 이미지 배율 scale). `write` 와 `relabel_road_obstacle` 이 같이 쓴다."""
        lines = []
        for b in boxes:
            b = b.clipped(w, h)
            if b.width * scale < self.min_box_px or b.height * scale < self.min_box_px:
                self.stats["box_dropped_small"] += 1
                continue
            cx, cy = (b.x1 + b.x2) / 2 / w, (b.y1 + b.y2) / 2 / h
            lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {b.width / w:.6f} {b.height / h:.6f}")
            self.stats[f"box_{CLASS_NAMES[b.cls]}"] += 1
        (self.root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        self.stats[f"img_{split}"] += 1
        if not lines:
            self.stats["img_background"] += 1
        return len(lines)

    def scale_for(self, w: int, h: int) -> float:
        """`resize_max(img, max_side)` 로 저장할 때의 배율 (이미지를 읽지 않고)."""
        s = self.max_side / max(w, h)
        return 1.0 if s >= 1 else round(w * s) / w


def facility_crop(
    img: np.ndarray, box: Box, pad: float = 0.1, size: int = 224, min_side_px: int = 24,
) -> np.ndarray | None:
    """시설물 박스(원본 픽셀 좌표) → 파손 분류 크롭. 학습 크롭(`CropWriter`)과 추론(`DamageClassifier`)이 같이 쓴다.

    박스 가로·세로를 각각 pad 비율만큼 넓히고 이미지 안으로 자른 뒤, 긴 변이 size 보다 크면 줄인다(INTER_AREA).
    짧은 변이 min_side_px 보다 작거나 잘린 영역이 비면 None.
    """
    h, w = img.shape[:2]
    if min(box.width, box.height) < min_side_px:
        return None
    px, py = box.width * pad, box.height * pad
    b = Box(box.cls, box.x1 - px, box.y1 - py, box.x2 + px, box.y2 + py).clipped(w, h)
    crop = img[int(b.y1):int(b.y2), int(b.x1):int(b.x2)]
    if crop.size == 0:
        return None
    return resize_max(crop, size)


class CropWriter:
    """시설물 파손 분류용 크롭. label 은 'normal' 또는 'damaged'."""

    def __init__(self, root: Path, size: int = 224, pad: float = 0.1, min_side_px: int = 24):
        self.root = Path(root)
        self.size, self.pad, self.min_side_px = size, pad, min_side_px
        self.stats: Counter[str] = Counter()

    def write(self, name: str, img: np.ndarray, box: Box, label: str, group: str) -> bool:
        if min(box.width, box.height) < self.min_side_px:
            self.stats["crop_dropped_small"] += 1
            return False
        crop = facility_crop(img, box, self.pad, self.size, self.min_side_px)
        if crop is None:
            return False
        split = split_for(group)
        out = self.root / split / label
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{name}.jpg"), crop)
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


SOURCE_PREFIXES = {"rdd2022": "rdd_", "road179": "ro_", "krid": "krid_"}  # 변환 때 붙인 파일 이름 접두사


def source_subset(det_root: Path, out_root: Path, prefix: str, split: str = "test") -> Path:
    """det_root 의 split 중 파일 이름이 prefix 로 시작하는 것만 모은 평가용 데이터셋 (심볼릭 링크) → data.yaml.

    출처별 평가용 (예: RDD2022 test 만, 도로장애물(179) test 만). Ultralytics 는 이미지 경로의 /images/ 를 /labels/ 로
    바꿔 라벨을 찾으므로 둘 다 링크한다. val·test 둘 다 이 split 을 가리킨다.
    """
    det_root, out_root = Path(det_root), Path(out_root)
    for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
        d = out_root / kind / split
        if d.exists():
            for p in d.iterdir():
                p.unlink()
        d.mkdir(parents=True, exist_ok=True)
        for p in sorted((det_root / kind / split).glob(f"{prefix}*{ext}")):
            (d / p.name).symlink_to(p.resolve())
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    y = out_root / "data.yaml"
    y.write_text(f"path: {out_root.resolve()}\ntrain: images/{split}\nval: images/{split}\ntest: images/{split}\n"
                 f"names:\n{names}\n")
    return y
