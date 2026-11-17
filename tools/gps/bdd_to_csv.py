"""BDD100K 주행 기록(info JSON)을 가상 블랙박스 GPS CSV 로 바꾼다 (docs/plan.md 4.6, P2).

GPS 가 없는 유튜브 영상은 경로를 재구성(tools/gps/reconstruct.py)하지만, BDD100K 는 휴대폰이 잰
실제 GPS 가 영상마다 있다. 같은 CSV(`t_s,lat,lon,speed_kmh`, contracts 의 GpsTrack)로 바꾸면
블랙박스·서버는 그대로 돈다 — "실제 블랙박스 GPS 로 바꿔도 다른 부분은 그대로"(PRD 8.5).

info JSON (영상 한 개당 한 파일, `bdd100k/info/<train|val>/<영상 id>.json`):

    {"filename": "<영상 id>.mov", "startTime": 1503828163000, "endTime": 1503828203000,
     "gps": [{"timestamp": 1503828163000, "latitude": 40.7, "longitude": -73.9,
              "speed": 8.3, "course": 120.0, "accuracy": 5.0, ...}, ...],
     "locations": [...], "accelerometer": [...], "gyro": [...]}

확인한 것: 위치 목록 키 `gps`, 좌표 `latitude`·`longitude` (공식 코드 bdd100k/vis/trajectory.py),
`filename`·`startTime`·`endTime` (github.com/bdd100k/bdd100k discussions/340), GPS 1 Hz · 영상 29.97 fps
(bdd100k-discuss 메일링 리스트). 실제 파일로 확인 전 가정: 시각은 epoch 밀리초(초 단위면 자동 인식),
`speed` 는 m/s 이고 음수는 '모름'(휴대폰 위치 API 관례), `startTime` 이 영상 첫 프레임 시각.
`gps` 가 없으면 `locations` 를 쓴다.

t_s = (GPS 시각 - startTime) 초. 영상 시작 전·끝 뒤 점도 남긴다 (영상 시작 직전 점이 있어야 t=0 을
보간할 수 있다; 블랙박스는 경로 범위 밖 프레임을 건너뛴다). 좌표가 없거나 (0,0) 인 점, 시각이 앞 점보다
늦지 않은 점은 버린다.

  uv run python -m tools.gps.bdd_to_csv data/bdd100k/info/b1c66a42-6f7d68ca.json
  uv run python -m tools.gps.bdd_to_csv data/bdd100k/info/*.json --out-dir data/bdd100k/videos
  # → <out-dir>/<영상 id>.gps.csv (기본: JSON 과 같은 폴더). 영상 옆에 두면 run.py 가 --gps 없이 찾는다

공통 함수는 tools/gps/reconstruct.py 와 아직 나누지 않았다 (따로 만드는 중). 겹치는 것이 생기면 합친다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from daboyeo_contracts import GpsPoint, GpsTrack

MS_THRESHOLD = 1e11  # 이보다 크면 epoch 밀리초 (1e11 초 ≈ 서기 5138년, 1e11 ms ≈ 1973년)
MPS_TO_KMH = 3.6


def _to_seconds(v: float) -> float:
    return v / 1000.0 if abs(v) > MS_THRESHOLD else float(v)


def _float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN → None


def info_points(info: dict) -> list[dict]:
    """위치 목록. `gps` 가 비었거나 없으면 `locations`."""
    for key in ("gps", "locations"):
        pts = info.get(key)
        if isinstance(pts, list) and pts:
            return pts
    raise ValueError("info JSON 에 위치 목록(gps / locations)이 없다")


def info_to_track(info: dict) -> GpsTrack:
    """info JSON 하나 → GpsTrack (t_s 는 영상 시작 기준 초)."""
    raw = info_points(info)
    start = _float(info.get("startTime"))
    if start is None:
        first = _float(raw[0].get("timestamp"))
        if first is None:
            raise ValueError("startTime 도 첫 점의 timestamp 도 없다")
        start = first
    t0 = _to_seconds(start)

    points: list[GpsPoint] = []
    for p in raw:
        ts, lat, lon = _float(p.get("timestamp")), _float(p.get("latitude")), _float(p.get("longitude"))
        if ts is None or lat is None or lon is None or (lat == 0 and lon == 0):
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        t_s = _to_seconds(ts) - t0
        if points and t_s <= points[-1].t_s:
            continue
        speed = _float(p.get("speed"))
        points.append(GpsPoint(t_s, lat, lon, speed * MPS_TO_KMH if speed is not None and speed >= 0 else None))
    if len(points) < 2:
        raise ValueError(f"쓸 수 있는 위치가 {len(points)}개뿐이다 (2개 이상 필요)")
    return GpsTrack(points)


def convert(info_path: Path, out_dir: Path | None = None) -> tuple[Path, GpsTrack]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    track = info_to_track(info)
    stem = Path(info.get("filename") or info_path.name).stem
    out = (out_dir or info_path.parent) / f"{stem}.gps.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(out)
    return out, track


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BDD100K info JSON → GPS CSV (t_s,lat,lon,speed_kmh)")
    ap.add_argument("info", nargs="+", type=Path, help="info JSON 파일 (여러 개 가능)")
    ap.add_argument("--out-dir", type=Path, default=None, help="CSV 를 둘 폴더. 기본: JSON 과 같은 폴더")
    args = ap.parse_args(argv)

    failed = 0
    for path in args.info:
        try:
            out, track = convert(path, args.out_dir)
        except (OSError, ValueError) as e:
            print(f"{path}: {e}", file=sys.stderr)
            failed += 1
            continue
        print(f"{out}: 점 {len(track.points)}개, {track.start_s:.1f}~{track.end_s:.1f}초, {track.length_m():.0f} m")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
