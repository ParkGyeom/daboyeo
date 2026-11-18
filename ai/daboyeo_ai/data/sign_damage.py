"""표지판 파손 분류 v2 (handoff 25장): Roboflow Universe 공개 표지판 파손 사진을 KRID 학습에 더한다.

흐름 (Mac 에서 한 번 → Colab):
  1) `crops`  : Roboflow YOLO 형식 내보내기(`data/roboflow_signs/<project>/`) → 박스마다 KRID 와 같은 규칙
                (`facility_crop`: pad 0.1 · 긴 변 224 이하)으로 자르고, 원래 클래스를 damaged/normal/제외로 바꾼다.
                거의 같은 크롭(dHash 해밍 거리 ≤ DUP_HAMMING)은 출처를 가리지 않고 하나만 남긴다.
                그다음 블랙박스 크롭처럼 **작고 흐리게** 만든 사본(`degrade`)을 쓴다. 15% 는 `holdout`
                (같은 원본 사진의 크롭은 같은 쪽).
  2) `pack`   : 1) 결과를 `sign_rf.zip` 으로 묶는다 → Drive `daboyeo_data/parts/`.
  3) Colab 03 노트북: `build_cls_v2` 가 KRID `cls/` + `sign_rf.zip` 으로 Ultralytics classify 폴더를 만든다.
     **val·test 는 KRID 만** (블랙박스와 같은 도메인으로 판정). Roboflow 크롭은 train 에만 들어간다.

라벨 바꾸기 (`RF_LABEL`): bent·broken sheet·crack·rust·damaged → damaged, not-damaged·normal·healthy → normal.
세로로 길쭉한 박스(`MAX_TALL`)·아주 납작한 박스(`MAX_WIDE`)도 뺀다 (지름길 방지).
graffiti(낙서)는 **뺀다** — KRID 의 파손 표지판은 탈색·벗겨짐·휨이고 낙서는 파손 라벨에 없으며, 정상으로 넣기에도
표지판 면이 가려져 애매하다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .damage_cls import _rank, crop_type, read_type_table, square_input
from .yolo import SPLITS, Box, facility_crop

RF_LABEL = {
    "bent": "damaged", "broken-sheet": "damaged", "broken sheet": "damaged", "crack": "damaged",
    "rust": "damaged", "damaged": "damaged",
    "not-damaged": "normal", "normal": "normal", "healthy": "normal",
    "graffiti": None,
}
DUP_HAMMING = 4          # dHash 64bit 해밍 거리. 같은 사진의 다른 증강/재업로드를 묶는다
# 세로/가로 > 2.5 또는 가로/세로 > 4 인 박스는 뺀다. Roboflow 에서 이런 박스(기둥째 넘어진 표지판·기둥 조각)는
# 거의 다 damaged(174 대 5)라 '길쭉하면 파손' 지름길이 되고, KRID 표지판 크롭은 99% 가 세로/가로 2.5 이하다.
MAX_TALL, MAX_WIDE = 2.5, 4.0
HOLDOUT_PCT = 15         # Roboflow 크롭 중 학습에 넣지 않고 같은 도메인 정확도만 볼 비율
# 블랙박스(KRID) 표지판 크롭: 긴 변 중앙값 ~90 px, 하위 10% 짧은 변 ~35 px (handoff 25장)
DEGRADE = {"long_min": 40, "long_max": 160, "blur_max": 1.2, "jpeg_min": 45, "jpeg_max": 90}


@dataclass
class RfCrop:
    file: str          # sign_rf/<split>/<label>/rf_<src>_<i>.jpg (split = train|holdout)
    source: str        # Roboflow 프로젝트 slug
    image: str         # 원본 이미지 파일 이름
    box_index: int
    raw_class: str
    label: str
    w: int             # 원본 크롭(degrade 전) 크기
    h: int
    dhash: str
    dup_of: str = ""   # 중복이면 남긴 크롭의 file (그 크롭은 쓰지 않는다)


# ---------------------------------------------------------------- 읽기

def read_data_yaml_names(root: Path) -> list[str]:
    """Roboflow data.yaml 의 `names: [...]` (한 줄 리스트) → 클래스 이름들."""
    for line in (Path(root) / "data.yaml").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            body = line.split(":", 1)[1].strip()
            return [s.strip().strip("'\"") for s in body.strip("[]").split(",") if s.strip()]
    raise ValueError(f"no names in {root}/data.yaml")


def yolo_boxes(txt: Path, w: int, h: int) -> Iterator[tuple[int, Box]]:
    """YOLO txt (cls cx cy bw bh, 0~1) → (cls, 픽셀 Box). 폴리곤 줄(점 6개 이상)은 외접 사각형."""
    for line in Path(txt).read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c, nums = int(v[0]), [float(x) for x in v[1:]]
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:
            xs, ys = nums[0::2], nums[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        yield c, Box(c, x1 * w, y1 * h, x2 * w, y2 * h)


def iter_source(root: Path) -> Iterator[tuple[str, np.ndarray, int, str, Box]]:
    """Roboflow 내보내기 폴더 → (이미지 이름, 이미지, 박스 번호, 원래 클래스, 박스). 모든 split 을 읽는다."""
    names = read_data_yaml_names(root)
    for img_path in sorted(Path(root).glob("*/images/*")):
        txt = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not txt.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for i, (c, box) in enumerate(yolo_boxes(txt, w, h)):
            yield img_path.name, img, i, names[c], box


# ---------------------------------------------------------------- 중복·도메인 맞추기

def dhash(img_bgr: np.ndarray, size: int = 8) -> int:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def mark_duplicates(hashes: list[int], max_dist: int = DUP_HAMMING) -> list[int]:
    """각 항목이 앞선 어떤 항목의 중복인지 → 대표 인덱스 리스트 (자기 자신이면 대표). 앞에 온 것이 남는다."""
    rep: list[int] = []
    keep: list[int] = []
    for i, h in enumerate(hashes):
        r = next((k for k in keep if hamming(hashes[k], h) <= max_dist), None)
        if r is None:
            keep.append(i)
            rep.append(i)
        else:
            rep.append(r)
    return rep


def degrade(crop_bgr: np.ndarray, seed: str, cfg: dict | None = None) -> np.ndarray:
    """가까이서 찍은 선명한 크롭 → 블랙박스 크롭처럼 작고 흐리게 (seed 문자열마다 같은 결과).

    긴 변을 long_min~long_max(로그 균등)로 줄이고, 가우시안 블러(σ 0~blur_max), JPEG 품질 jpeg_min~max 로
    다시 압축한다. 크기는 작은 채로 둔다 — KRID 의 작은 크롭처럼 학습·추론 때 `square_input` 이 키운다.
    """
    c = {**DEGRADE, **(cfg or {})}
    rng = random.Random(int(hashlib.sha1(seed.encode()).hexdigest()[:12], 16))
    h, w = crop_bgr.shape[:2]
    target = float(np.exp(rng.uniform(np.log(c["long_min"]), np.log(c["long_max"]))))
    s = min(1.0, target / max(h, w))
    out = crop_bgr if s >= 1 else cv2.resize(
        crop_bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    sigma = rng.uniform(0.0, c["blur_max"])
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    q = rng.randint(c["jpeg_min"], c["jpeg_max"])
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else out


# ---------------------------------------------------------------- 1) crops

def build_rf_crops(
    sources: dict[str, Path], out: Path, degrade_crops: bool = True, holdout_pct: int = HOLDOUT_PCT,
) -> tuple[list[RfCrop], Counter[str]]:
    """Roboflow 폴더들 → `out/sign_rf/{train,holdout}/{damaged,normal}/*.jpg` + `out/sign_rf/manifest.csv`.

    sources: {slug: 폴더}. 중복은 manifest 에 `dup_of` 로 남기고 파일은 쓰지 않는다.
    """
    out = Path(out) / "sign_rf"
    if out.exists():
        shutil.rmtree(out)
    stats: Counter[str] = Counter()
    rows: list[RfCrop] = []
    crops: list[np.ndarray] = []
    for slug, root in sources.items():
        short = slug.split("/")[-1]
        for img_name, img, i, raw, box in iter_source(root):
            label = RF_LABEL.get(raw.lower(), "unknown")
            if label == "unknown":
                raise ValueError(f"{slug}: unknown class {raw!r} — RF_LABEL 에 추가할 것")
            if label is None:
                stats[f"excluded_{raw}"] += 1
                continue
            if box.height > MAX_TALL * box.width or box.width > MAX_WIDE * box.height:
                stats["dropped_aspect"] += 1
                continue
            crop = facility_crop(img, box)
            if crop is None:
                stats["dropped_small"] += 1
                continue
            stats[f"raw_{short}_{raw}"] += 1
            rows.append(RfCrop("", slug, img_name, i, raw, label, crop.shape[1], crop.shape[0], ""))
            crops.append(crop)
    hashes = [dhash(c) for c in crops]
    rep = mark_duplicates(hashes)
    # 같은 원본 사진(출처+이미지)의 크롭은 같은 쪽에. 대표가 같으면(중복) 대표를 따른다
    for k, (r, c) in enumerate(zip(rows, crops)):
        r.dhash = f"{hashes[k]:016x}"
        group = f"{rows[rep[k]].source}/{rows[rep[k]].image}"
        split = "holdout" if int(_rank(group, "daboyeo-sign-rf"), 16) % 100 < holdout_pct else "train"
        r.file = f"sign_rf/{split}/{r.label}/rf_{r.source.split('/')[-1]}_{k:05d}.jpg"
    for k, (r, c) in enumerate(zip(rows, crops)):
        if rep[k] != k:
            r.dup_of = rows[rep[k]].file
            stats["duplicate"] += 1
            continue
        if rows[rep[k]].label != r.label:  # 대표가 자기 자신이라 생길 수 없지만 방어
            stats["label_conflict"] += 1
        p = out.parent / r.file
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), degrade(c, r.file) if degrade_crops else c, [cv2.IMWRITE_JPEG_QUALITY, 95])
        stats[f"{r.file.split('/')[1]}_{r.label}"] += 1
    # 중복인데 라벨이 다른 쌍 (같은 사진을 서로 다르게 라벨한 경우) — 대표만 쓰지만 수는 남긴다
    stats["dup_label_conflict"] = sum(1 for k, r in enumerate(rows) if rep[k] != k and rows[rep[k]].label != r.label)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else ["file"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    return rows, stats


def pack(out: Path, zip_path: Path, sources_json: Path | None = None) -> Path:
    """`out/sign_rf/` → zip (Drive parts/ 에 올릴 것). 출처·라이선스(sources.json)를 같이 넣는다 (CC BY)."""
    root = Path(out)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # jpg 라 압축해도 거의 안 줄어든다
        for p in sorted((root / "sign_rf").rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root).as_posix())
        if sources_json is not None and Path(sources_json).exists():
            z.write(sources_json, "sign_rf/sources.json")
    return Path(zip_path)


# ---------------------------------------------------------------- 3) 학습 폴더 (Colab)

def _write_sq(src: Path, dst: Path, size: int) -> None:
    img = cv2.imread(str(src))
    if img is None:
        raise ValueError(f"cannot read {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), square_input(img, size))


def build_cls_v2(
    krid_cls: Path,
    rf_root: Path | None,
    dst: Path,
    facility: str = "sign",
    krid_normal_ratio: dict[str, float] | None = None,
    rf_normal_max_ratio: float | None = None,
    salt: str = "daboyeo-dmg",
    size: int = 224,
    types_table: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """KRID `cls/` (+ Roboflow `sign_rf/train`) → `dst/{train,val,test}/{normal,damaged}/*.png`.

    facility: "sign" 이면 KRID 크롭 중 표지판만, "all" 이면 가드레일+표지판 (v1b 와 같음).
    KRID normal 은 split 마다 `KRID damaged 수 × krid_normal_ratio[split]` 개를 해시 순으로 (v1b: 5/10/10).
    val·test 는 KRID 만. rf_root=None 이면 KRID 만 (v1b 다시 만들기). dst 는 지우고 새로 만든다.
    rf_normal_max_ratio: Roboflow normal 을 Roboflow damaged 의 몇 배까지 (None = 전부).
    """
    ratio = {"train": 5.0, "val": 10.0, "test": 10.0, **(krid_normal_ratio or {})}
    table = types_table if types_table is not None else read_type_table()
    krid_cls, dst = Path(krid_cls), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    out: dict[str, dict[str, int]] = {}

    def keep(p: Path) -> bool:
        return facility == "all" or crop_type(p.stem, table) == facility

    for split in SPLITS:
        damaged = sorted(p for p in (krid_cls / split / "damaged").glob("*.jpg") if keep(p))
        normal = sorted((p for p in (krid_cls / split / "normal").glob("*.jpg") if keep(p)),
                        key=lambda p: _rank(p.stem, salt))
        keep_n = normal[: int(round(len(damaged) * ratio[split]))]
        for label, files in (("damaged", damaged), ("normal", keep_n)):
            for p in files:
                _write_sq(p, dst / split / label / f"{p.stem}.png", size)
        out[split] = {"krid_damaged": len(damaged), "krid_normal": len(keep_n), "krid_normal_available": len(normal)}
    if rf_root is not None:
        rf_d = sorted((Path(rf_root) / "sign_rf" / "train" / "damaged").glob("*.jpg"))
        rf_n = sorted((Path(rf_root) / "sign_rf" / "train" / "normal").glob("*.jpg"), key=lambda p: _rank(p.stem, salt))
        if rf_normal_max_ratio is not None:
            rf_n = rf_n[: int(round(len(rf_d) * rf_normal_max_ratio))]
        for label, files in (("damaged", rf_d), ("normal", rf_n)):
            for p in files:
                _write_sq(p, dst / "train" / label / f"{p.stem}.png", size)
        out["train"].update(rf_damaged=len(rf_d), rf_normal=len(rf_n))
    return out


def rf_holdout_files(rf_root: Path) -> Iterable[tuple[Path, int]]:
    """Roboflow holdout 크롭 → (경로, 1=damaged)."""
    for label, y in (("damaged", 1), ("normal", 0)):
        for p in sorted((Path(rf_root) / "sign_rf" / "holdout" / label).glob("*.jpg")):
            yield p, y


# ---------------------------------------------------------------- CLI

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m daboyeo_ai.data.sign_damage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops", help="Roboflow 내보내기 폴더들 → sign_rf/ 크롭 + manifest.csv")
    c.add_argument("--rf-dir", type=Path, default=Path("data/roboflow_signs"),
                   help="sources.json 과 프로젝트 폴더들이 있는 곳")
    c.add_argument("--out", type=Path, default=Path("data/sign_damage_v2"))
    c.add_argument("--no-degrade", action="store_true")
    c.add_argument("--zip", type=Path, default=None, help="주면 sign_rf.zip 까지 만든다")
    args = ap.parse_args(argv)

    srcs = json.loads((args.rf_dir / "sources.json").read_text())
    sources = {s["slug"]: args.rf_dir / s["slug"].split("/")[1] for s in srcs}
    rows, stats = build_rf_crops(sources, args.out, degrade_crops=not args.no_degrade)
    shutil.copy(args.rf_dir / "sources.json", args.out / "sign_rf" / "sources.json")
    print(f"crops={len(rows)} -> {args.out / 'sign_rf'}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    if args.zip:
        pack(args.out, args.zip)
        print(f"zip -> {args.zip} ({args.zip.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
