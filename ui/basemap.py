"""인터넷 없이 쓰는 지도 배경 (docs/DESIGN.md 7장, B10).

오픈스트리트맵 공식 타일은 미리 받을 수 없어서(이용 정책), 시연 구간 주변의 도로 데이터만
Overpass API 로 한 번 받아 두고 대시보드에 도로선을 직접 그린다.
데이터 출처: © OpenStreetMap contributors (ODbL). 화면에 출처를 표시한다.

  받기(인터넷 필요, 한 번만): .venv/bin/python -m ui.basemap fetch
  저장 위치: data/basemap/<구간>.json, 구간마다 한 파일 (저장소에 올리지 않는다)
  이미 받은 구간은 건너뛴다. 공용 서버가 바빠서 실패한 구간은 다시 실행하면 받는다
"""

from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASEMAP_DIR = ROOT / "data" / "basemap"
ATTRIBUTION = "지도 데이터 © OpenStreetMap contributors (ODbL)"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "daboyeo-demo/0.1 (student project; offline road basemap)"
MARGIN_M = 500

# docs/DATA.md 3장의 시연 구간 후보
SEGMENTS = [
    "0401_light_rainy_smooth:03000071-03000182",
    "0401_night_clear_smooth:06000388-06000490",
    "0401_light_clear_smooth:07000006-07000040",
    "0401_light_clear_smooth:01000128-01000201",
    "0401_night_clear_smooth:06000016-06000115",
]
HIGHWAYS = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link"


def segment_bbox(segment: str, frames_root: str | Path = ROOT / "data/roadmark/VS4") -> tuple[float, float, float, float] | None:
    """구간 프레임들의 GPS 범위에 여유를 더한 (남, 서, 북, 동)."""
    from edge.blackbox.frames import list_frames, load_gps

    fixes = [g for f in list_frames(frames_root, segment) if (g := load_gps(f))]
    if not fixes:
        return None
    lat0 = sum(g.lat for g in fixes) / len(fixes)
    dlat = MARGIN_M / 111_320
    dlon = MARGIN_M / (111_320 * math.cos(math.radians(lat0)))
    return (min(g.lat for g in fixes) - dlat, min(g.lon for g in fixes) - dlon,
            max(g.lat for g in fixes) + dlat, max(g.lon for g in fixes) + dlon)


def _file(segment: str) -> Path:
    return BASEMAP_DIR / (segment.replace(":", "_") + ".json")


def fetch() -> None:
    import httpx

    BASEMAP_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for seg in SEGMENTS:
        if _file(seg).exists():
            print(f"{seg}: 이미 받음")
            continue
        bbox = segment_bbox(seg)
        if bbox is None:
            print(f"{seg}: 프레임이 없어 건너뜀")
            continue
        q = f'[out:json][timeout:60];way["highway"~"^({HIGHWAYS})$"]({",".join(f"{v:.6f}" for v in bbox)});out geom;'
        for attempt in range(3):  # 공용 서버가 바쁘면 502·504·429 를 돌려준다. 기다렸다 다시 시도한다
            r = httpx.post(OVERPASS_URL, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=120)
            if r.status_code not in (429, 502, 503, 504):
                break
            print(f"{seg}: 서버가 바쁨({r.status_code}), {20 * (attempt + 1)}초 뒤 다시 시도")
            time.sleep(20 * (attempt + 1))
        if r.status_code != 200:
            print(f"{seg}: 실패({r.status_code}). 나중에 다시 실행한다")
            failed.append(seg)
            continue
        roads = []
        for el in r.json().get("elements", []):
            if el.get("type") == "way" and el.get("geometry"):
                roads.append({
                    "id": el["id"],
                    "highway": el.get("tags", {}).get("highway", ""),
                    "coords": [[round(p["lat"], 6), round(p["lon"], 6)] for p in el["geometry"]],
                })
        _file(seg).write_text(json.dumps({"attribution": ATTRIBUTION, "bbox": bbox, "roads": roads}, ensure_ascii=False))
        print(f"{seg}: 도로 {len(roads)}개 저장 ({_file(seg).stat().st_size / 1024:.0f}KB)")
        time.sleep(2)  # 공용 서버에 부담을 주지 않게 요청 사이를 띄운다
    if failed:
        sys.exit(f"받지 못한 구간 {len(failed)}개. 잠시 뒤 다시 실행한다")


def load_roads(bbox: tuple[float, float, float, float] | None = None) -> list[dict]:
    """받아 둔 도로. bbox(남, 서, 북, 동)를 주면 그 안에 걸치는 것만. 파일이 없으면 빈 목록."""
    roads = _all_roads()
    if bbox is None:
        return roads
    s, w, n, e = bbox
    return [r for r in roads if any(s <= la <= n and w <= lo <= e for la, lo in r["coords"])]


@functools.lru_cache(maxsize=1)
def _all_roads() -> list[dict]:
    """파일은 프로세스마다 한 번만 읽는다. 대시보드가 5초마다 다시 그려도 파일을 다시 읽지 않는다.
    도로를 새로 받았으면 화면을 다시 띄운다(scripts/demo.sh down → up)."""
    seen, roads = set(), []
    for f in sorted(BASEMAP_DIR.glob("*.json")) if BASEMAP_DIR.exists() else []:
        for r in json.loads(f.read_text(encoding="utf-8"))["roads"]:
            if r["id"] not in seen:  # 구간이 겹치면 같은 도로가 두 번 들어 있다
                seen.add(r["id"])
                roads.append(r)
    return roads


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        sys.path.insert(0, str(ROOT))
        fetch()
    else:
        print(__doc__)
