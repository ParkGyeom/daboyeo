"""받을 AI Hub zip 고르기 — 노트북 01 과 로컬 CLI(`prepare_local`)가 같이 쓴다.

- KRID(71786): 정해 둔 이미지 zip 목록 + 이름 규칙(TS→TL, VS→VL)으로 짝 라벨 zip.
- 도로장애물(179): 라벨 zip 마다 포트홀·크랙 수를 세고, 짝 이미지 zip 의 'GB 당 개수' 상위를 고른다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .aihub import AihubFile
from .sources import CRACK, POTHOLE, road_obstacle_class

KRID_KEY, ROAD_KEY = 71786, 179

# ---- 기본 설정 (노트북 01 설정 셀과 같은 값)
# 전국 도로시설물(KRID): 서울 시연과 비슷한 '도시부' + 검증셋 일부
KRID_IMAGE_ZIPS = (
    "TS_2.국도_02.도시부.zip",    # 9 GB
    "TS_3.지방도_02.도시부.zip",  # 0.7 GB
    "VS_2.국도_02.도시부.zip",    # 1 GB
    "VS_3.지방도_02.도시부.zip",  # 79 MB
)
KRID_STRIDE = 2              # 연속 프레임 중 몇 장에 1장 쓸지
KRID_INCLUDE_MEDIAN = False  # 중앙분리대도 guardrail 로 넣을지
# 도로장애물/표면: 라벨 집계 후 '포트홀 수 / GB' 가 높은 zip 부터 고른다
ROAD_TYPES = ("Mainroad", "Frontback")  # 국도·지방도, 도심 골목 (Highway·Industrialroads 제외)
ROAD_POTHOLE_ZIPS = 3
ROAD_CRACK_ZIPS = 2
ROAD_STRIDE = 5              # 1분 영상의 연속 프레임이라 5장에 1장
ROAD_NEGATIVE_RATIO = 0.1    # 보수완료 포트홀·맨홀만 있는 배경 이미지 비율
MIN_ROAD_IMAGE_GB = 0.1     # 이보다 작은 이미지 zip 은 고르지 않는다 (사실상 빈 zip)
MAX_IMAGES_PER_ZIP = 4000    # zip 하나에서 쓸 최대 이미지 수 (클래스 균형용)

KRID_LABEL_DIR = "/02.라벨링데이터/"
ROAD_LABEL_DIR, ROAD_IMAGE_DIR = "1.Annotations", "2.Images_new_211012"


@dataclass(frozen=True)
class Job:
    kind: str  # "krid" | "road"
    ds: int
    image: AihubFile
    label: AihubFile

    @property
    def part(self) -> str:
        """progress.json 키이자 작업 폴더 이름. 겹치지 않게 fileSn 을 붙인다."""
        return f"{self.kind}_{self.image.key}_{self.image.name.removesuffix('.zip')}"


def krid_label_files(files: list[AihubFile]) -> list[AihubFile]:
    return [f for f in files if KRID_LABEL_DIR in f.path]


def road_label_files(files: list[AihubFile]) -> list[AihubFile]:
    return [f for f in files if f"/{ROAD_LABEL_DIR}/" in f.path]


def krid_label_for(image_name: str) -> str:
    """`TS_2.국도_02.도시부.zip` → `TL_2.국도_02.도시부.zip` (VS → VL)."""
    for src, dst in (("TS_", "TL_"), ("VS_", "VL_")):
        if image_name.startswith(src):
            return dst + image_name[len(src):]
    raise ValueError(f"not a KRID image zip name (TS_/VS_): {image_name}")


def krid_jobs(files: list[AihubFile], image_zips=KRID_IMAGE_ZIPS) -> list[Job]:
    by_name = {f.name: f for f in files}
    missing = [n for n in image_zips if n not in by_name or krid_label_for(n) not in by_name]
    if missing:
        raise KeyError(f"KRID file tree has no image/label zip for: {missing}")
    return [Job("krid", KRID_KEY, by_name[n], by_name[krid_label_for(n)]) for n in image_zips]


def krid_count_table(counter: Counter[str]) -> tuple[list[str], list[dict]]:
    """count_label_zip(..., 'krid') 합계 → (상태 열 이름들, [{name, <상태>: 개수, ...}]) — '정상' 많은 순."""
    table: dict[str, Counter[str]] = {}
    for k, v in counter.items():
        if "|" not in k:
            continue  # '_json_files' 같은 집계용 키
        name, state = k.split("|", 1)
        table.setdefault(name, Counter())[state] += v
    states = sorted({s for c in table.values() for s in c}, key=lambda s: (s != "정상", s))
    rows = [{"name": n, **{s: c[s] for s in states}} for n, c in table.items()]
    rows.sort(key=lambda r: (-r.get("정상", 0), r["name"]))
    return states, rows


def road_image_for(label: AihubFile, files_by_path: dict[str, AihubFile]) -> AihubFile | None:
    return files_by_path.get(label.path.replace(ROAD_LABEL_DIR, ROAD_IMAGE_DIR))


def road_count_row(label: AihubFile, counts: Counter[str], image: AihubFile | None,
                   road_types=ROAD_TYPES) -> dict:
    """라벨 zip 하나의 집계(count_label_zip(..., 'coco')) → label_counts_road.csv 한 줄."""
    cls: Counter = Counter()
    for name, n in counts.items():
        if name != "_json_files":
            cls[road_obstacle_class(name)] += n
    return dict(
        label=label.path,
        kind="CRACK" if "/2.CRACK/" in label.path else "TOA",
        road=next((t for t in road_types if t in label.path), "other"),
        pothole=cls[POTHOLE], crack=cls[CRACK], negative=cls["negative"],
        json_files=counts["_json_files"],
        image=image.path if image else None,
        image_gb=image.size_bytes / 1024**3 if image else None,
    )


def pick_road_rows(rows: list[dict], n_pothole=ROAD_POTHOLE_ZIPS, n_crack=ROAD_CRACK_ZIPS,
                   min_image_gb: float = MIN_ROAD_IMAGE_GB) -> list[dict]:
    """대상 도로 종류 + 이미지 zip 이 있는 것 중 TOA 는 포트홀/GB, CRACK 은 크랙/GB 상위 n 개.

    이미지 zip 이 아주 작으면(예: 실제 179 의 4 KB 짜리 C_Frontback_B02) 라벨만 있고 이미지는 거의 없는데
    'GB 당 개수' 만 터무니없이 커지므로 min_image_gb 미만은 뺀다.
    """
    cand = [r for r in rows if r["road"] != "other" and r["image"] and (r["image_gb"] or 0) >= min_image_gb]

    def top(kind: str, field: str, n: int) -> list[dict]:
        rs = [r for r in cand if r["kind"] == kind and r[field] > 0]
        rs.sort(key=lambda r: -r[field] / r["image_gb"])  # 안정 정렬: 동점이면 입력 순서
        return rs[:n]

    return top("TOA", "pothole", n_pothole) + top("CRACK", "crack", n_crack)


def road_jobs(rows: list[dict], files_by_path: dict[str, AihubFile]) -> list[Job]:
    return [Job("road", ROAD_KEY, files_by_path[r["image"]], files_by_path[r["label"]]) for r in rows]
