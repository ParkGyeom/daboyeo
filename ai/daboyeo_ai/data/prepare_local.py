"""AI Hub 학습 데이터를 로컬(한국 안의 Mac 등)에서 준비한다 — 노트북 01 의 2~4단계 대체.

AI Hub 는 해외 IP 의 다운로드를 막아서 Colab 에서는 받을 수 없다. 여기서 라벨·이미지 zip 을 받아
YOLO 형식으로 바꾸고 `parts/krid.zip`, `parts/road.zip` 을 만든다. 그 두 파일을 Google Drive
`daboyeo_data/parts/` 에 올리면 Colab 의 `02_train_detector` 가 그대로 쓴다.

    uv run python -m daboyeo_ai.data.prepare_local --out data/aihub

API 키는 환경변수 또는 레포 최상위 `.env` 의 `AIHUB_APIKEY` 에서 읽는다 (출력하지 않는다).
중간에 끊겨도 다시 실행하면 `progress.json` 에 기록된 zip 은 건너뛴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import aihub, pipeline, selection, sources
from .aihub import AihubFile
from .selection import Job
from .yolo import CropWriter, YoloDetWriter

EXIT_ROAD_FORMAT = 2  # 도로장애물 라벨이 COCO 형식이 아님 → KRID 만 만들고 멈춤


@dataclass(frozen=True)
class Settings:
    krid_stride: int = selection.KRID_STRIDE
    krid_include_median: bool = selection.KRID_INCLUDE_MEDIAN
    road_stride: int = selection.ROAD_STRIDE
    road_negative_ratio: float = selection.ROAD_NEGATIVE_RATIO
    max_images_per_zip: int = selection.MAX_IMAGES_PER_ZIP


def convert_job(job: Job, label_zip: Path, image_zip: Path, out: Path, s: Settings = Settings()) -> Counter[str]:
    """받은 라벨·이미지 zip 한 쌍 → out/det (YOLO), KRID 는 out/cls (파손 분류 크롭) 도."""
    det = YoloDetWriter(out / "det")
    if job.kind == "krid":
        crops = CropWriter(out / "cls")
        stats = pipeline.process_krid(label_zip, image_zip, det, crops, stride=s.krid_stride,
                                      max_images=s.max_images_per_zip, include_median=s.krid_include_median)
        stats.update(crops.stats)
    else:
        stats = pipeline.process_road_obstacle(label_zip, image_zip, det, stride=s.road_stride,
                                               max_images=s.max_images_per_zip,
                                               negative_ratio=s.road_negative_ratio)
    stats.update(det.stats)
    return stats


# ---------------------------------------------------------------- API 키

def _env_candidates(start: Path) -> list[Path]:
    """start 부터 레포 최상위까지의 .env. git worktree 안이면 원래 체크아웃의 .env 도 본다."""
    out = []
    for d in (start, *start.parents):
        out.append(d / ".env")
        git = d / ".git"
        if git.is_dir():
            break
        if git.is_file():  # worktree: "gitdir: <main>/.git/worktrees/<name>"
            text = git.read_text(errors="replace").strip()
            if text.startswith("gitdir:"):
                gitdir = Path(text.split(":", 1)[1].strip())
                if gitdir.parent.name == "worktrees":
                    out.append(gitdir.parent.parent.parent / ".env")
            break
    return out


def _read_dotenv(path: Path, name: str) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip("'\"") or None
    return None


def load_apikey(start: Path | None = None, name: str = "AIHUB_APIKEY") -> str:
    if os.environ.get(name):
        return os.environ[name]
    candidates = _env_candidates((start or Path.cwd()).resolve())
    for p in candidates:
        if p.is_file() and (v := _read_dotenv(p, name)):
            return v
    raise SystemExit(f"{name} not set: put `{name}=...` in the repo-root .env or export it "
                     f"(looked in: {', '.join(str(p) for p in candidates)})")


# ---------------------------------------------------------------- 진행 기록

class Progress:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def __contains__(self, part: str) -> bool:
        return part in self.data

    def done(self, part: str, stats: dict) -> None:
        self.data[part] = stats
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)


def _gb(files) -> float:
    return sum(f.size_bytes for f in files) / 1024**3


def _download_all(ds: int, files: list[AihubFile], apikey: str, dest: Path, what: str) -> dict[str, Path]:
    out = {}
    t = time.time()
    for i, f in enumerate(files, 1):
        print(f"  [{what} {i}/{len(files)}] {f.name} ({f.size})", flush=True)
        out[f.path] = aihub.download(ds, f, apikey, dest)
    print(f"  {what}: {len(files)} zips, {_gb(files):.2f} GB in {time.time() - t:.0f}s", flush=True)
    return out


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: 엑셀에서 한글이 안 깨짐
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- 도로장애물 라벨 형식 확인

def coco_like_problem(label_zip: Path, sample: int = 20) -> str | None:
    """앞쪽 JSON 몇 개를 parse_coco_like 로 실제로 파싱해 보고, 되면 None, 안 되면 이유."""
    with zipfile.ZipFile(label_zip) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".json")][:sample]
        if not names:
            return "no .json in label zip"
        for n in names:
            js = pipeline._read_json(zf, n)
            if not isinstance(js, dict):
                return f"{n}: top-level JSON is {type(js).__name__}, not an object"
            try:
                sources.parse_coco_like(js)
            except Exception as e:  # noqa: BLE001 — 어떤 형식 차이든 사람에게 보여 주고 멈춘다
                return f"{n}: parse_coco_like failed: {type(e).__name__}: {e}"
    return None


def check_road_format(peek_zips: dict[str, Path], report: Path) -> list[str]:
    problems, text = [], []
    for title, z in peek_zips.items():
        peek = pipeline.peek_label_zip(z, max_chars=8000)
        text.append(f"=== {title}: {z.name} ===\n{peek}\n")
        if (p := coco_like_problem(z)) is not None:
            problems.append(f"{title}: {p}")
    report.write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    return problems


# ---------------------------------------------------------------- parts zip

def build_part_zip(work_dirs: list[Path], dest: Path) -> Counter[str]:
    """여러 작업 폴더(det/, cls/)를 zip 하나로. 02_train_detector 는 zip 의 det/ 만 /content/dataset 에 푼다."""
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    tmp = dest.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:  # jpg 라 다시 압축해도 거의 안 줄어든다
        for d in work_dirs:
            for p in sorted(d.rglob("*")):
                if not p.is_file():
                    continue
                arc = p.relative_to(d).as_posix()
                if arc in seen:
                    stats["duplicate_skipped"] += 1
                    continue
                seen.add(arc)
                zf.write(p, arc)
                if arc.startswith("det/images/"):
                    stats[f"det_images_{arc.split('/')[2]}"] += 1
                elif arc.startswith("cls/"):
                    stats["cls_crops"] += 1
    tmp.replace(dest)
    return stats


# ---------------------------------------------------------------- main

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m daboyeo_ai.data.prepare_local", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/aihub"), help="작업·결과 폴더 (기본 data/aihub)")
    ap.add_argument("--krid-zips", nargs="*", default=list(selection.KRID_IMAGE_ZIPS),
                    help="받을 KRID 이미지 zip 이름 (기본: 노트북 설정과 같은 4개)")
    ap.add_argument("--krid-labels", choices=("all", "selected"), default="all",
                    help="all: KRID 라벨 전부 받아 집계 CSV 작성 (기본) / selected: 고른 zip 의 라벨만")
    ap.add_argument("--road-pothole-zips", type=int, default=selection.ROAD_POTHOLE_ZIPS)
    ap.add_argument("--road-crack-zips", type=int, default=selection.ROAD_CRACK_ZIPS)
    ap.add_argument("--road-types", nargs="*", default=list(selection.ROAD_TYPES))
    ap.add_argument("--no-road", action="store_true", help="도로장애물(179)은 건너뛴다")
    ap.add_argument("--select-only", action="store_true", help="라벨 다운로드·집계·선택까지만 (이미지 zip 은 안 받음)")
    ap.add_argument("--max-images-per-zip", type=int, default=selection.MAX_IMAGES_PER_ZIP)
    ap.add_argument("--keep-raw", action="store_true", help="변환 후 원본 이미지 zip 을 지우지 않는다")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out: Path = args.out.resolve()
    labels_dir, raw_dir, work_dir, parts_dir = out / "labels", out / "raw", out / "work", out / "parts"
    for d in (labels_dir, work_dir, parts_dir):
        d.mkdir(parents=True, exist_ok=True)
    apikey = load_apikey()
    settings = Settings(max_images_per_zip=args.max_images_per_zip)
    print(f"out: {out}  (free disk: {shutil.disk_usage(out).free / 1024**3:.0f} GB)")

    # 1. 파일 목록
    krid_files = aihub.fetch_tree(selection.KRID_KEY)
    krid = selection.krid_jobs(krid_files, args.krid_zips)
    krid_labels = selection.krid_label_files(krid_files) if args.krid_labels == "all" else [j.label for j in krid]
    road = not args.no_road and (args.road_pothole_zips or args.road_crack_zips)
    road_files = aihub.fetch_tree(selection.ROAD_KEY) if road else []
    road_labels = selection.road_label_files(road_files)

    # 2. 라벨 다운로드 + KRID 집계
    print(f"\n## labels: KRID {len(krid_labels)} zips {_gb(krid_labels):.2f} GB"
          + (f", road {len(road_labels)} zips {_gb(road_labels):.2f} GB" if road else ""))
    label_paths = _download_all(selection.KRID_KEY, krid_labels, apikey, labels_dir / "krid", "KRID labels")
    krid_total: Counter[str] = Counter()
    for f in krid_labels:
        krid_total += pipeline.count_label_zip(label_paths[f.path], "krid")
    states, rows = selection.krid_count_table(krid_total)
    write_csv(out / "label_counts_krid.csv", ["name", *states], rows)

    # 3. 도로장애물: 라벨 형식 확인 → 집계 → 선택
    jobs = list(krid)
    road_problem: list[str] = []
    if road:
        toa = [f for f in road_labels if "/1.TOA/" in f.path]
        crack = [f for f in road_labels if "/2.CRACK/" in f.path]
        peek = {"TOA (pothole)": min(toa, key=lambda f: f.size_bytes),
                "CRACK": min(crack, key=lambda f: f.size_bytes)}
        peek_paths = _download_all(selection.ROAD_KEY, list(peek.values()), apikey, labels_dir / "road", "road label peek")
        print("\n## 도로장애물 label structure")
        road_problem = check_road_format({t: peek_paths[f.path] for t, f in peek.items()},
                                         out / "road_label_structure.txt")
    if road and not road_problem:
        label_paths |= _download_all(selection.ROAD_KEY, road_labels, apikey, labels_dir / "road", "road labels")
        by_path = {f.path: f for f in road_files}
        road_rows = [
            selection.road_count_row(f, pipeline.count_label_zip(label_paths[f.path], "coco"),
                                     selection.road_image_for(f, by_path), tuple(args.road_types))
            for f in road_labels
        ]
        write_csv(out / "label_counts_road.csv", list(road_rows[0]), road_rows)
        if sum(r["pothole"] for r in road_rows) == 0:
            road_problem = [("0 potholes counted — fix sources.road_obstacle_class / parse_coco_like "
                             "using road_label_structure.txt")]
        else:
            picked = selection.pick_road_rows(road_rows, args.road_pothole_zips, args.road_crack_zips)
            jobs += selection.road_jobs(picked, by_path)

    print("\n## selected image zips")
    for j in jobs:
        print(f"  {j.kind:4} {j.image.size:>7}  {j.image.path}")
    todo = [j for j in jobs if j.part not in Progress(out / "progress.json")]
    print(f"  total {_gb(j.image for j in jobs):.1f} GB, still to do {_gb(j.image for j in todo):.1f} GB")

    # 4. 이미지 zip 하나씩: 받기 → 변환 → 원본 삭제
    if not args.select_only:
        progress = Progress(out / "progress.json")
        for i, j in enumerate(jobs, 1):
            job_out = work_dir / j.part
            if j.part in progress and job_out.exists():
                print(f"[{i}/{len(jobs)}] skip (done): {j.part}")
                continue
            print(f"[{i}/{len(jobs)}] {j.image.name} ({j.image.size}) downloading...", flush=True)
            t = time.time()
            img_zip = aihub.download(j.ds, j.image, apikey, raw_dir)
            t_dl = time.time() - t
            shutil.rmtree(job_out, ignore_errors=True)
            stats = convert_job(j, label_paths[j.label.path], img_zip, job_out, settings)
            if not args.keep_raw:
                shutil.rmtree(raw_dir / str(j.image.key), ignore_errors=True)
            progress.done(j.part, dict(stats, image=j.image.path, download_s=round(t_dl),
                                       seconds=round(time.time() - t)))
            print(f"    done in {time.time() - t:.0f}s (download {t_dl:.0f}s): {dict(stats)}", flush=True)

        # 5. parts zip
        print("\n## parts")
        for kind in ("krid", "road"):
            dirs = [work_dir / j.part for j in jobs if j.kind == kind and (work_dir / j.part).exists()]
            if not dirs:
                continue
            dest = parts_dir / f"{kind}.zip"
            stats = build_part_zip(dirs, dest)
            print(f"  {dest}  {dest.stat().st_size / 1024**2:.0f} MB  {dict(stats)}")
        print(f"\nUpload everything in {parts_dir} to Google Drive daboyeo_data/parts/, then run "
              f"02_train_detector with PARTS = None.")
        print(f"Label counts: {out / 'label_counts_krid.csv'}"
              + (f", {out / 'label_counts_road.csv'}" if (out / "label_counts_road.csv").exists() else ""))

    if road_problem:
        print("\n!! 도로장애물 skipped — label format is not what parse_coco_like expects:", file=sys.stderr)
        for p in road_problem:
            print(f"   {p}", file=sys.stderr)
        print(f"   structure: {out / 'road_label_structure.txt'}", file=sys.stderr)
        return EXIT_ROAD_FORMAT
    return 0


if __name__ == "__main__":
    sys.exit(main())
