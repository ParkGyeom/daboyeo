"""시설물 파손 분류(계획 2.7) 데이터 준비: 크롭별 시설물 종류 복원 + normal 줄이기.

`prepare_local` 이 만든 `krid.zip` 의 `cls/{split}/{normal,damaged}/krid_<이미지 stem>_<k>.jpg` 는
파일 이름에 시설물 종류(guardrail/sign)가 없다. k 는 `parse_krid(...).facilities` 의 순서이므로
(작아서 버린 크롭도 번호는 차지한다) 같은 라벨 JSON 을 다시 파싱하면 종류를 되찾을 수 있다.

라벨 zip 은 한국 IP 에서만 받을 수 있어서 Colab 이 다시 만들 수 없다. 그래서 Mac 에서 한 번
`python -m daboyeo_ai.data.damage_cls types ...` 로 이미지 단위 표 (`krid_crop_types.csv.gz`, stem → "gsg..")
를 만들어 레포에 둔다.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from daboyeo_contracts import CLASS_NAMES

from .sources import parse_krid
from .yolo import SPLITS

LABELS = ("normal", "damaged")
FACILITY_TYPES = ("guardrail", "sign")
_CODE = {"guardrail": "g", "sign": "s"}
_TYPE = {v: k for k, v in _CODE.items()}
TYPES_CSV = Path(__file__).with_name("krid_crop_types.csv.gz")


# ---------------------------------------------------------------- 크롭 → 시설물 종류

def krid_image_facilities(label_zip: Path, include_median: bool = False) -> Iterator[tuple[str, str, str]]:
    """라벨 zip → (이미지 stem, 종류 코드열 "gs..", 파손 여부열 "01..").

    `pipeline.process_krid` 와 같은 `parse_krid` 를 쓰므로 k 번째 글자가 크롭 `_<k>` 의 종류다.
    """
    from .pipeline import _read_json  # pipeline 은 무거운 import 가 없지만 순환을 피한다

    with zipfile.ZipFile(label_zip) as lz:
        for n in sorted(x for x in lz.namelist() if x.lower().endswith(".json")):
            li = parse_krid(_read_json(lz, n), include_median=include_median)
            if not li.facilities:
                continue
            types = "".join(_CODE[CLASS_NAMES[b.cls]] for b, _ in li.facilities)
            dmg = "".join("1" if d else "0" for _, d in li.facilities)
            yield PurePosixPath(li.file_name).stem, types, dmg


def split_crop_name(name: str) -> tuple[str, int]:
    """`krid_<stem>_<k>` (확장자 없이) → (stem, k)."""
    if not name.startswith("krid_"):
        raise ValueError(f"not a KRID crop name: {name}")
    stem, _, k = name[len("krid_"):].rpartition("_")
    return stem, int(k)


def crop_member_info(member: str) -> tuple[str, str, str] | None:
    """`cls/<split>/<label>/<name>.jpg` → (split, label, name). 다른 경로는 None."""
    parts = PurePosixPath(member).parts
    if len(parts) != 4 or parts[0] != "cls" or parts[1] not in SPLITS or parts[2] not in LABELS:
        return None
    if not parts[3].lower().endswith(".jpg"):
        return None
    return parts[1], parts[2], PurePosixPath(parts[3]).stem


def build_type_table(
    crop_members: Iterable[str], facilities: Iterable[tuple[str, str, str]],
) -> tuple[dict[str, str], Counter[str]]:
    """크롭 경로들과 라벨 표를 맞춰 {이미지 stem: 종류 코드열} 을 만든다 (크롭이 있는 이미지만).

    검증: 크롭 폴더(normal/damaged)와 라벨의 파손 여부가 다르면 `label_mismatch`, 라벨이 없으면
    `missing`, 같은 stem 이 라벨 zip 여러 개에서 다르게 나오면 `conflict` 를 센다. 셋 다 0 이어야
    k 순서가 `parse_krid` 와 맞는다는 뜻이다.
    """
    stats: Counter[str] = Counter()
    by_stem: dict[str, tuple[str, str]] = {}
    for stem, types, dmg in facilities:
        if stem in by_stem and by_stem[stem] != (types, dmg):
            stats["conflict"] += 1
        by_stem[stem] = (types, dmg)
    table: dict[str, str] = {}
    for m in crop_members:
        info = crop_member_info(m)
        if info is None:
            continue
        split, label, name = info
        stem, k = split_crop_name(name)
        got = by_stem.get(stem)
        if got is None or k >= len(got[0]):
            stats["missing"] += 1
            continue
        types, dmg = got
        if (dmg[k] == "1") != (label == "damaged"):
            stats["label_mismatch"] += 1
        table[stem] = types
        stats[f"{split}_{label}_{_TYPE[types[k]]}"] += 1
    return table, stats


def write_type_table(table: dict[str, str], path: Path = TYPES_CSV) -> Path:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["stem", "types"])
    for stem in sorted(table):
        w.writerow([stem, table[stem]])
    # mtime=0: 같은 내용이면 같은 바이트 (git 에서 쓸데없는 변경이 안 생기게)
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    return Path(path)


def read_type_table(path: Path = TYPES_CSV) -> dict[str, str]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return {r["stem"]: r["types"] for r in csv.DictReader(f)}


def crop_type(name: str, table: dict[str, str]) -> str | None:
    """크롭 이름(확장자 유무 상관없음) → 'guardrail' / 'sign' / None(표에 없음)."""
    try:
        stem, k = split_crop_name(PurePosixPath(name).stem if name.endswith(".jpg") else name)
    except ValueError:
        return None
    types = table.get(stem)
    return _TYPE[types[k]] if types is not None and k < len(types) else None


# ---------------------------------------------------------------- normal 줄이기 (YOLO-cls 폴더)

def _rank(name: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}:{name}".encode()).hexdigest()


def square_input(crop_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """`facility_crop` 결과 → 분류 모델 입력 크기의 정사각형 (비율 유지, 남는 곳은 회색 114).

    Ultralytics classify 는 짧은 변을 imgsz 로 맞춘 뒤 가운데를 잘라서, 가로로 긴 가드레일 크롭은 대부분이
    잘려 나간다. 그래서 학습 폴더에 넣을 때 미리 정사각형으로 만들고 (`subsample_cls`), 추론
    (`DamageClassifier`)도 같은 함수를 쓴다. 정사각형 size 입력이면 Ultralytics 의 resize·center crop 은
    그대로 통과한다.
    """
    from ..inference.onnx_detector import letterbox

    return letterbox(crop_bgr, size)[0]


def subsample_cls(
    src: Path,
    dst: Path,
    normal_ratio: dict[str, float] | None = None,
    salt: str = "daboyeo-dmg",
    size: int = 224,
) -> dict[str, dict[str, int]]:
    """`src/{split}/{normal,damaged}/*.jpg` → `dst/{train,val,test}/{normal,damaged}/*.png` (Ultralytics classify 구조).

    damaged 는 전부, normal 은 split 마다 `damaged 수 × normal_ratio[split]` 개만 파일 이름 해시 순으로 고른다
    (다시 실행해도 같은 결과). 기본은 train 5배, val·test 10배. 각 크롭은 `square_input` 으로 size×size 로 만들어
    PNG(무손실)로 쓴다. dst 는 지우고 새로 만든다.
    """
    ratio = {"train": 5.0, "val": 10.0, "test": 10.0, **(normal_ratio or {})}
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    out: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        damaged = sorted((src / split / "damaged").glob("*.jpg"))
        normal = sorted((src / split / "normal").glob("*.jpg"), key=lambda p: _rank(p.stem, salt))
        keep_n = normal[: int(round(len(damaged) * ratio[split]))]
        for label, files in (("damaged", damaged), ("normal", keep_n)):
            d = dst / split / label
            d.mkdir(parents=True, exist_ok=True)
            for p in files:
                img = cv2.imread(str(p))
                if img is None:
                    raise ValueError(f"cannot read {p}")
                cv2.imwrite(str(d / f"{p.stem}.png"), square_input(img, size))
        out[split] = {"damaged": len(damaged), "normal": len(keep_n), "normal_available": len(normal)}
    return out


# ---------------------------------------------------------------- CLI (Mac 에서 한 번)

def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m daboyeo_ai.data.damage_cls")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("types", help="krid.zip 크롭 + KRID 라벨 zip → 시설물 종류 표 (csv.gz)")
    t.add_argument("--krid-zip", type=Path, required=True, help="prepare_local 이 만든 parts/krid.zip")
    t.add_argument("--labels", type=Path, required=True, help="KRID 라벨 zip 들이 있는 폴더 (하위 폴더까지 찾음)")
    t.add_argument("--out", type=Path, default=TYPES_CSV)
    t.add_argument("--include-median", action="store_true", help="prepare_local 을 중앙분리대 포함으로 돌렸을 때만")
    args = ap.parse_args(argv)

    label_zips = sorted(args.labels.rglob("*.zip"))

    def facilities() -> Iterator[tuple[str, str, str]]:
        for z in label_zips:
            print(f"  {z.name}", flush=True)
            yield from krid_image_facilities(z, args.include_median)

    with zipfile.ZipFile(args.krid_zip) as kz:
        members = kz.namelist()
    table, stats = build_type_table(members, facilities())
    write_type_table(table, args.out)
    print(f"label zips={len(label_zips)} images={len(table)} -> {args.out}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    bad = stats["conflict"] + stats["missing"] + stats["label_mismatch"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_main())
