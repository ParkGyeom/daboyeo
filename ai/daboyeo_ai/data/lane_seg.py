"""차선 영역 분할 학습 데이터: AI Hub 087 「도로 로드마크 인식을 위한 주행 영상 데이터」 VL4/VS4 → YOLO-seg.

차선 마모 판정(`lane_wear.py`, 밝기 대비)이 횡단보도·노면 글자·화살표를 차선으로 잡는 헛것(handoff 17장)을 줄이려고,
**측정할 영역을 진짜 차선으로 한정하는 분할 모델**을 학습한다 (handoff 21장, 실험).

클래스 (2개):
    0 lane     차선 (VL4 supercategory roadlane·roadlane_ban = category 381~391: 중앙선·차선·버스전용차로·
               길가장자리구역선·진로변경제한선·주정차금지선). 라벨이 polyline(선 가운데)이라 원근에 맞춰 폭을 준다.
    1 marking  차선이 아닌 노면표시 (화살표·글자·속도제한·안전지대·횡단보도·정지선·유도선 …, 나머지 전부).
               차선과 헷갈리는 것을 모델이 따로 배우게 하려는 반대 클래스 — 마모 판정에는 0 만 쓴다.

입력: 라벨 폴더 `vl4/2D_JSON/<name>.json` (이미 받음) + 이미지 `VS4.비온전(복합).zip` (안의 `IMAGE/<name>.png`) 또는 폴더.
출력: `<out>/images/{train,val}/*.jpg`, `<out>/labels/{train,val}/*.txt` (YOLO-seg: `cls x1 y1 x2 y2 …` 정규화),
`<out>/data.yaml`, `<out>/stats.json`. 쪼개기는 주행(이름 앞 4자리 번호) 안 50 프레임 묶음 단위 해시 (이웃 프레임이
train·val 에 같이 들어가지 않게). test 는 없다 — 진짜 시험은 시연 클립 gn_a·gn_b 다.

  uv run python -m daboyeo_ai.data.lane_seg \\
      --labels data/roadmark/vl4/2D_JSON \\
      --images "data/roadmark/download/57809/087.도로_로드마크_인식을_위한_주행_영상_데이터/01.데이터/2.Validation/원천데이터/VS4.비온전(복합).zip" \\
      --out data/lane_seg --zip data/lane_seg.zip --preview 12
  # data/lane_seg.zip 을 Drive daboyeo_data/parts/ 에 올리고 ai/notebooks/04_train_lane_seg.ipynb
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from daboyeo_ai.data.yolo import resize_max, split_for

SEG_CLASS_NAMES = ("lane", "marking")
LANE_CATS = frozenset(range(381, 392))  # roadlane 381~388 + roadlane_ban 389~391
SPLIT_RATIOS = (85, 100)  # train 85 / val 15 / test 0
BLOCK = 50  # 같은 주행에서 연속 50 프레임은 같은 split
SALT = "lane_seg"


# ---- 이름 → 묶음 -------------------------------------------------------------------------


def block_group(name: str, block: int = BLOCK) -> str:
    """`0401_light_clear_smooth_06000045` → `0401_light_clear_smooth_0600#0` (주행 + 50 프레임 묶음)."""
    p = name.split("_")
    idx = p[-1]
    if len(p) != 5 or not idx.isdigit():
        return name
    drive = "_".join(p[:4]) + "_" + idx[:4]
    return f"{drive}#{int(idx[4:] or 0) // block}"


# ---- polyline → 폭 있는 다각형 ----------------------------------------------------------


def half_width(y: np.ndarray, h: int, w: int, horizon: float = 0.45, max_frac: float = 0.008,
               min_px: float = 1.5) -> np.ndarray:
    """화면 y 에서 차선 반폭(px). 지평선(horizon·h) 에서 0, 화면 맨 아래에서 max_frac·w (1920 폭이면 15 px).

    원근상 노면 폭은 지평선 아래 거리에 비례한다. 지평선 위나 가까운 곳은 min_px.
    """
    rel = np.clip((np.asarray(y, float) - horizon * h) / ((1 - horizon) * h), 0.0, 1.0)
    return np.maximum(rel * max_frac * w, min_px)


def draw_polyline(mask: np.ndarray, pts: np.ndarray, full_h: int, full_w: int, scale: float) -> None:
    """polyline(원본 픽셀 N×2) 을 원근 폭으로 mask(저장 크기)에 칠한다. 마디마다 사다리꼴 + 원."""
    if len(pts) < 2:
        return
    hw = half_width(pts[:, 1], full_h, full_w) * scale
    p = pts * scale
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        d = b - a
        n = np.hypot(*d)
        if n < 1e-6:
            continue
        nx, ny = -d[1] / n, d[0] / n
        quad = np.array([
            (a[0] + nx * hw[i], a[1] + ny * hw[i]), (b[0] + nx * hw[i + 1], b[1] + ny * hw[i + 1]),
            (b[0] - nx * hw[i + 1], b[1] - ny * hw[i + 1]), (a[0] - nx * hw[i], a[1] - ny * hw[i]),
        ])
        cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 255)
    for (x, y), r in zip(p, hw):
        cv2.circle(mask, (round(x), round(y)), max(1, round(r)), 255, -1)


def mask_polygons(mask: np.ndarray, min_area: float = 12.0, eps: float = 0.7) -> list[np.ndarray]:
    """칠한 mask → 바깥 윤곽 다각형들 (저장 크기 픽셀, 각 M×2, M≥3). 화면 밖은 mask 가 이미 잘랐다."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append(c.astype(float))
    return out


def _rings(seg) -> list[np.ndarray]:
    """COCO segmentation(리스트의 리스트 또는 평평한 리스트) → [N×2]. 숫자 아닌 것·3점 미만은 뺀다."""
    if not seg:
        return []
    if not isinstance(seg[0], (list, tuple)):
        seg = [seg]
    out = []
    for r in seg:
        try:
            a = np.asarray(r, float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        a = a[np.isfinite(a).all(1)]
        if len(a) >= 2:
            out.append(a)
    return out


def _box_ok(b) -> bool:
    try:
        return b is not None and len(b) == 4 and all(np.isfinite(float(v)) for v in b) and b[2] > 0 and b[3] > 0
    except (TypeError, ValueError):
        return False


def instances(annotations: Iterable[dict], full_w: int, full_h: int, scale: float,
              stats: Counter | None = None) -> list[tuple[int, np.ndarray]]:
    """VL4 annotations → [(cls, 다각형 M×2 저장 크기 px)]. 라벨 하나가 화면에서 여러 조각이면 조각마다 하나."""
    sw, sh = round(full_w * scale), round(full_h * scale)
    out = []
    for a in annotations:
        cat = a.get("category_id")
        cls = 0 if cat in LANE_CATS else 1
        m = np.zeros((sh, sw), np.uint8)
        if a.get("polyline"):
            for ring in _rings(a["polyline"]):
                draw_polyline(m, ring, full_h, full_w, scale)
        else:
            rings = [r for r in _rings(a.get("segmentation")) if len(r) >= 3]
            if not rings and _box_ok(a.get("bbox")):
                # 글자·속도제한 등 약 2,000 개는 다각형 없이 bbox 만 있다 → 사각형 (marking 은 마모 판정에 안 쓰므로 충분)
                x, y, bw, bh = (float(v) for v in a["bbox"])
                rings = [np.array([(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)])]
                if stats is not None:
                    stats["ann_bbox_only"] += 1
            if not rings:
                if stats is not None:
                    stats["ann_empty"] += 1
                continue
            cv2.fillPoly(m, [np.round(r * scale).astype(np.int32) for r in rings], 255)
        polys = mask_polygons(m)
        if stats is not None:
            stats[f"ann_{SEG_CLASS_NAMES[cls]}"] += 1
            stats[f"inst_{SEG_CLASS_NAMES[cls]}"] += len(polys)
            if not polys:
                stats["ann_dropped_small"] += 1
        out.extend((cls, p) for p in polys)
    return out


def label_lines(insts: list[tuple[int, np.ndarray]], w: int, h: int) -> list[str]:
    lines = []
    for cls, p in insts:
        xy = np.clip(p / (w, h), 0.0, 1.0).reshape(-1)
        lines.append(f"{cls} " + " ".join(f"{v:.5f}" for v in xy))
    return lines


# ---- 이미지 읽기 -------------------------------------------------------------------------


class Images:
    """zip(`…/<name>.png`, 하위 폴더 상관없음) 또는 폴더에서 이름으로 읽는다."""

    def __init__(self, path: Path):
        path = Path(path)
        self._zip = zipfile.ZipFile(path) if path.is_file() else None
        if self._zip is not None:
            self._names = {Path(n).stem: n for n in self._zip.namelist() if n.lower().endswith((".png", ".jpg"))}
        else:
            self._names = {p.stem: str(p) for p in sorted(path.iterdir()) if p.suffix.lower() in (".png", ".jpg")}

    def __contains__(self, name: str) -> bool:
        return name in self._names

    def read(self, name: str) -> np.ndarray | None:
        key = self._names.get(name)
        if key is None:
            return None
        buf = self._zip.read(key) if self._zip is not None else Path(key).read_bytes()
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


# ---- 변환 --------------------------------------------------------------------------------


def write_data_yaml(root: Path) -> Path:
    root = Path(root)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(SEG_CLASS_NAMES))
    p = root / "data.yaml"
    p.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return p


def preview(img: np.ndarray, insts: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """확인용 그림: lane 빨강 채움, marking 하늘색 윤곽."""
    over = img.copy()
    for cls, p in insts:
        q = np.round(p).astype(np.int32)
        if cls == 0:
            cv2.fillPoly(over, [q], (0, 0, 255))
        else:
            cv2.polylines(over, [q], True, (255, 255, 0), 2)
    return cv2.addWeighted(over, 0.6, img, 0.4, 0)


def convert(labels: Path, images: Path, out: Path, max_side: int = 960, jpeg_quality: int = 90,
            limit: int | None = None, preview_n: int = 0) -> dict:
    labels, out = Path(labels), Path(out)
    src = Images(images)
    for kind in ("images", "labels"):
        for s in ("train", "val"):
            (out / kind / s).mkdir(parents=True, exist_ok=True)
    if preview_n:
        (out / "preview").mkdir(exist_ok=True)
    stats: Counter = Counter()
    files = sorted(labels.glob("*.json"))[:limit]
    for i, f in enumerate(files):
        name = f.stem
        if name not in src:
            stats["no_image"] += 1
            continue
        d = json.loads(f.read_text(encoding="utf-8-sig"))
        img = src.read(name)
        if img is None:
            stats["bad_image"] += 1
            continue
        h, w = img.shape[:2]
        small = resize_max(img, max_side)
        sh, sw = small.shape[:2]
        insts = instances(d.get("annotations", []), w, h, sw / w, stats)
        split = split_for(block_group(name), SALT, SPLIT_RATIOS)
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        lines = label_lines(insts, sw, sh)
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[f"img_{split}"] += 1
        meta = (d.get("images") or [{}])
        meta = meta[0] if isinstance(meta, list) else meta
        stats[f"cond_{meta.get('time', '?')}_{meta.get('weather', '?')}"] += 1
        if not any(c == 0 for c, _ in insts):
            stats["img_no_lane"] += 1
        if preview_n and i % max(1, len(files) // preview_n) == 0:
            cv2.imwrite(str(out / "preview" / f"{name}.jpg"), preview(small, insts))
    write_data_yaml(out)
    res = {"classes": list(SEG_CLASS_NAMES), "lane_categories": sorted(LANE_CATS), "max_side": max_side,
           "split": {"train": SPLIT_RATIOS[0], "val": SPLIT_RATIOS[1] - SPLIT_RATIOS[0], "group": f"주행+{BLOCK}프레임"},
           "stats": dict(sorted(stats.items()))}
    (out / "stats.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def zip_dir(root: Path, zip_path: Path) -> Path:
    """`images/`·`labels/`·`stats.json` 을 `lane_seg/` 아래로 묶는다 (Colab 에서 풀면 /content/lane_seg/…)."""
    root, zip_path = Path(root), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # jpg 라 압축해도 거의 안 준다
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if p.is_file() and rel.parts[0] in ("images", "labels", "stats.json"):
                z.write(p, Path("lane_seg") / rel)
    return zip_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="VL4 2D_JSON 폴더")
    ap.add_argument("--images", type=Path, required=True, help="VS4 zip 또는 이미지 폴더")
    ap.add_argument("--out", type=Path, default=Path("data/lane_seg"))
    ap.add_argument("--zip", type=Path, default=None, help="만든 뒤 이 zip 으로 묶는다 (Drive parts/ 에 올릴 것)")
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N 장만 (시험용)")
    ap.add_argument("--preview", type=int, default=0, help="확인용 그림 N 장 (<out>/preview/)")
    a = ap.parse_args(argv)
    res = convert(a.labels, a.images, a.out, a.max_side, limit=a.limit, preview_n=a.preview)
    print(json.dumps(res["stats"], ensure_ascii=False, indent=1))
    if a.zip:
        print("zip:", zip_dir(a.out, a.zip), f"{a.zip.stat().st_size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
