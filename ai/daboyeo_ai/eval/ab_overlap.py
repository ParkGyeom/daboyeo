"""두 클립(다른 날 같은 길) 사이 같은 결함 맞추기 — 정답 결함·v3 탐지를 GPS 좌표로 바꿔 거리로 짝짓는다 (handoff 20장).

    # 1) 짝 후보 sheet (레포 밖 data/ab_overlap/, 눈으로 같은 결함인지 확인)
    uv run python -m daboyeo_ai.eval.ab_overlap sheets
    # 2) 숫자 (정답 짝 거리 + v3 탐지를 서버처럼 20 m 로 묶었을 때 두 대가 같이 본 결함)
    uv run python -m daboyeo_ai.eval.ab_overlap run --json experiments/ab_overlap.json

- 위치: 정답 결함은 구간 가운데 시각, 탐지는 그 프레임 시각의 GPS (블랙박스도 프레임 시각 GPS 를 그대로 보낸다).
  결함은 실제로 카메라 앞 몇~수십 m 에 있지만, 두 클립 모두 같은 규칙이라 대부분 상쇄된다. `--ahead-m` 으로 앞으로 민다.
- GPS CSV 끝을 조금 넘는 시각은 마지막 구간 방향·속도로 늘인다 (`MAX_EXTRAP_S` 까지, `extrap` 표시).
- 탐지 → 관측: `edge/blackbox/track_merge.TrackMerger` (읽기 전용으로 부른다)로 트랙을 합쳐 트랙마다 최고 신뢰도 프레임 1건.
- 서버 묶기: `server/dedup.ingest` 와 같은 규칙 — 같은 유형·반경 안에서 가장 가까운 결함에 붙이고, 결함 위치는 관측 평균.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from daboyeo_contracts import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[3]
CLIPS = {"A": "gn_a_2050-2510", "B": "gn_b_3320-3620"}
ROAD_TYPES = ("pothole", "crack")  # 서버로 보내는 도로 결함 (시설물은 화면에만)
RADII = (20.0, 30.0, 50.0)
MAX_EXTRAP_S = 8.0  # gn_b B_C11 (클립 끝, GPS 끝 6초 뒤) 까지


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class Fix:
    lat: float
    lon: float
    extrap: bool = False


class Track:
    """GPS CSV (t_s,lat,lon[,speed_kmh]) — 시각 → 위치, 경로를 따라 앞으로 민 위치."""

    def __init__(self, rows: list[tuple[float, float, float]]):
        if len(rows) < 2:
            raise ValueError("track needs >= 2 points")
        a = np.array(rows, dtype=float)
        self.t, self.lat, self.lon = a[:, 0], a[:, 1], a[:, 2]
        step = [haversine_m(*self._ll(i - 1), *self._ll(i)) for i in range(1, len(a))]
        self.s = np.concatenate([[0.0], np.cumsum(step)])  # 경로를 따라 간 거리 (m)

    @classmethod
    def from_csv(cls, path: Path) -> Track:
        with open(path) as f:
            return cls([(float(r["t_s"]), float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)])

    def _ll(self, i: int) -> tuple[float, float]:
        return float(self.lat[i]), float(self.lon[i])

    def dist_at(self, t: float) -> tuple[float, bool] | None:
        """시각 t 의 경로 거리. 끝을 MAX_EXTRAP_S 안으로 넘으면 끝 구간 속도로 늘인다."""
        t0, t1 = self.t[0], self.t[-1]
        if t0 <= t <= t1:
            return float(np.interp(t, self.t, self.s)), False
        if t1 < t <= t1 + MAX_EXTRAP_S:
            v = (self.s[-1] - self.s[-2]) / (self.t[-1] - self.t[-2])
            return float(self.s[-1] + v * (t - t1)), True
        if t0 - MAX_EXTRAP_S <= t < t0:
            v = (self.s[1] - self.s[0]) / (self.t[1] - self.t[0])
            return float(self.s[0] - v * (t0 - t)), True
        return None

    def at_dist(self, s: float) -> tuple[float, float]:
        """경로 거리 s 의 위치 (양 끝 밖은 끝 구간 방향으로 늘인다)."""
        if s > self.s[-1] or s < 0:
            i, j = (len(self.s) - 2, len(self.s) - 1) if s > self.s[-1] else (0, 1)
            seg = self.s[j] - self.s[i]
            f = (s - self.s[i]) / seg if seg else 0.0
            return (float(self.lat[i] + f * (self.lat[j] - self.lat[i])),
                    float(self.lon[i] + f * (self.lon[j] - self.lon[i])))
        # 멈춘 구간은 s 가 같은 점이 여럿이라 고유한 s 로 보간한다
        s_u, idx = np.unique(self.s, return_index=True)
        return float(np.interp(s, s_u, self.lat[idx])), float(np.interp(s, s_u, self.lon[idx]))

    def fix(self, t: float, ahead_m: float = 0.0) -> Fix | None:
        d = self.dist_at(t)
        if d is None:
            return None
        lat, lon = self.at_dist(d[0] + ahead_m)
        return Fix(lat, lon, d[1])


def read_gt(path: Path, types=ROAD_TYPES) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [{"id": r["defect_id"], "type": r["defect_type"], "t0": float(r["t_start_s"]), "t1": float(r["t_end_s"]),
             "amb": r["memo"].startswith("amb"), "memo": r["memo"]} for r in rows if r["defect_type"] in types]


def locate(events: list[dict], track: Track, ahead_m: float = 0.0) -> list[dict]:
    out = []
    for e in events:
        f = track.fix((e["t0"] + e["t1"]) / 2, ahead_m)
        out.append(e | ({"lat": None, "lon": None, "extrap": False} if f is None else
                        {"lat": round(f.lat, 7), "lon": round(f.lon, 7), "extrap": f.extrap}))
    return out


def pair_events(a: list[dict], b: list[dict], max_m: float = max(RADII), same_type: bool = True) -> list[dict]:
    """A·B 정답 결함 쌍 중 max_m 안인 것 (거리 순). 위치 없는 결함은 뺀다."""
    out = []
    for x in a:
        for y in b:
            if x["lat"] is None or y["lat"] is None or (same_type and x["type"] != y["type"]):
                continue
            d = haversine_m(x["lat"], x["lon"], y["lat"], y["lon"])
            if d <= max_m:
                out.append({"a": x["id"], "b": y["id"], "type_a": x["type"], "type_b": y["type"], "dist_m": round(d, 1)})
    return sorted(out, key=lambda p: p["dist_m"])


def nearest_counterpart(a: list[dict], b: list[dict]) -> dict[str, dict]:
    """A 결함마다 B 의 같은 유형 가장 가까운 결함과 거리."""
    out = {}
    for x in a:
        best = None
        for y in b:
            if x["lat"] is None or y["lat"] is None or x["type"] != y["type"]:
                continue
            d = haversine_m(x["lat"], x["lon"], y["lat"], y["lon"])
            if best is None or d < best[1]:
                best = (y["id"], d)
        out[x["id"]] = None if best is None else {"id": best[0], "dist_m": round(best[1], 1)}
    return out


@dataclass
class Det:
    type: str
    confidence: float
    bbox: tuple[float, float, float, float]


def observations(raw: dict[str, list], thr: dict[str, float], track: Track, device: str,
                 types=ROAD_TYPES, merge: bool = True) -> list[dict]:
    """프레임별 탐지 캐시({"t": [[cls, conf, x1, y1, x2, y2], ...]}) → 블랙박스가 보낼 관측 (시각·위치·신뢰도)."""
    from edge.blackbox.track_merge import TrackMerger  # 정현우 담당 코드, 읽기 전용으로 부른다

    frames = sorted((float(k), v) for k, v in raw.items())
    hits = []
    if merge:
        m = TrackMerger()
        tracks = []
        for t, ds in frames:
            dets = [Det(CLASS_NAMES[d[0]], d[1], tuple(d[2:6])) for d in ds
                    if CLASS_NAMES[d[0]] in types and d[1] >= thr[CLASS_NAMES[d[0]]]]
            tracks += m.update(dets, t)
        tracks += m.flush()
        hits = [(tr.best_t, tr.best, tr.hits) for tr in tracks]
    else:
        hits = [(t, Det(CLASS_NAMES[d[0]], d[1], tuple(d[2:6])), 1) for t, ds in frames for d in ds
                if CLASS_NAMES[d[0]] in types and d[1] >= thr[CLASS_NAMES[d[0]]]]
    out = []
    for t, d, n in sorted(hits, key=lambda h: h[0]):
        f = track.fix(t)
        if f is None:
            continue  # GPS 구간 밖: 블랙박스도 위치 없음으로 건너뛴다
        out.append({"device": device, "type": d.type, "t": round(t, 3), "conf": round(d.confidence, 3),
                    "lat": f.lat, "lon": f.lon, "frames": n})
    return out


def server_group(obs: list[dict], radius_m: float) -> list[dict]:
    """server/dedup.ingest 규칙: 도착 순서대로, 같은 유형·radius_m 안의 가장 가까운 결함에 붙인다. 위치 = 관측 평균."""
    defects: list[dict] = []
    for o in obs:
        near = [(haversine_m(o["lat"], o["lon"], d["lat"], d["lon"]), d) for d in defects if d["type"] == o["type"]]
        near = [x for x in near if x[0] <= radius_m]
        if near:
            d = min(near, key=lambda x: x[0])[1]
        else:
            d = {"type": o["type"], "obs": []}
            defects.append(d)
        d["obs"].append(o)
        d["lat"] = sum(x["lat"] for x in d["obs"]) / len(d["obs"])
        d["lon"] = sum(x["lon"] for x in d["obs"]) / len(d["obs"])
    for d in defects:
        d["devices"] = sorted({x["device"] for x in d["obs"]})
    return defects


def _thr(model_json: Path) -> dict[str, float]:
    return json.loads(model_json.read_text())["class_conf_draft"]


def summarize_groups(defects: list[dict]) -> dict:
    multi = [d for d in defects if len(d["devices"]) >= 2]
    return {"defects": len(defects), "by_type": {t: sum(d["type"] == t for d in defects) for t in ROAD_TYPES},
            "two_devices": len(multi),
            "two_device_defects": [{"type": d["type"], "lat": round(d["lat"], 6), "lon": round(d["lon"], 6),
                                    "obs": [{"device": o["device"], "t": o["t"], "conf": o["conf"]} for o in d["obs"]]}
                                   for d in multi]}


def run(args) -> dict:
    tracks = {k: Track.from_csv(args.videos / f"{v}.gps.csv") for k, v in CLIPS.items()}
    gt = {k: locate(read_gt(args.gt / f"{v}.csv"), tracks[k], args.ahead_m) for k, v in CLIPS.items()}
    pairs = pair_events(gt["A"], gt["B"])
    res: dict = {
        "clips": CLIPS, "ahead_m": args.ahead_m,
        "gps_span_s": {k: [float(tr.t[0]), float(tr.t[-1])] for k, tr in tracks.items()},
        "gt": {k: [{x: e[x] for x in ("id", "type", "t0", "t1", "amb", "lat", "lon", "extrap")} for e in v]
               for k, v in gt.items()},
        "gt_nearest_a_to_b": nearest_counterpart(gt["A"], gt["B"]),
        "gt_pairs_within": {f"{r:g}m": [p for p in pairs if p["dist_m"] <= r] for r in RADII},
    }
    thr = _thr(args.model_json)
    obs = {}
    for k, v in CLIPS.items():
        cache = json.loads((args.cache / "_".join(v.split("_")[:2]) / "raw.json").read_text())  # gn_a/raw.json
        obs[k] = observations(cache["raw"]["v3"], thr, tracks[k], k, merge=not args.no_merge)
    allobs = obs["B"] + obs["A"]  # 촬영 순서 (gn_b 2022-01 → gn_a 2022 여름, tools/video/sources.csv) = 도착 순서
    res["detections"] = {
        "model": "v3", "thr": thr, "merge": not args.no_merge,
        "obs": {k: {"n": len(v), "by_type": {t: sum(o["type"] == t for o in v) for t in ROAD_TYPES}}
                for k, v in obs.items()},
        "server": {f"{r:g}m": summarize_groups(server_group(allobs, r)) for r in RADII},
        "obs_list": {k: [{x: (round(o[x], 6) if x in ("lat", "lon") else o[x]) for x in o if x != "device"}
                         for o in v] for k, v in obs.items()},
    }
    return res


def sheets(args) -> None:
    """짝 후보마다 A(위)·B(아래) 결함 구간 처음·가운데·끝 프레임을 나란히 → data/ab_overlap/pair_<a>_<b>.jpg."""
    import cv2

    tracks = {k: Track.from_csv(args.videos / f"{v}.gps.csv") for k, v in CLIPS.items()}
    gt = {k: {e["id"]: e for e in locate(read_gt(args.gt / f"{v}.csv"), tracks[k], args.ahead_m)}
          for k, v in CLIPS.items()}
    pairs = pair_events(list(gt["A"].values()), list(gt["B"].values()), args.max_m, same_type=False)
    args.out.mkdir(parents=True, exist_ok=True)
    caps = {k: cv2.VideoCapture(str(args.videos / f"{v}.mp4")) for k, v in CLIPS.items()}

    def grab(k: str, t: float):
        caps[k].set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = caps[k].read()
        f = f if ok else np.zeros((1080, 1920, 3), np.uint8)
        f = cv2.resize(f, (640, 360))
        cv2.putText(f, f"{k} t={t:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        return f

    for p in pairs:
        rows = []
        for k, eid in (("A", p["a"]), ("B", p["b"])):
            e = gt[k][eid]
            rows.append(np.hstack([grab(k, t) for t in (e["t0"], (e["t0"] + e["t1"]) / 2, e["t1"])]))
        img = np.vstack(rows)
        cv2.putText(img, f"{p['a']} vs {p['b']}  {p['dist_m']} m", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2)
        cv2.imwrite(str(args.out / f"pair_{p['a']}_{p['b']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{p['a']:6s} {p['b']:6s} {p['dist_m']:6.1f} m")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "sheets"):
        p = sub.add_parser(name)
        p.add_argument("--videos", type=Path, default=ROOT / "data/videos", help="클립과 <클립>.gps.csv 가 있는 곳")
        p.add_argument("--gt", type=Path, default=ROOT / "experiments/gt_events")
        p.add_argument("--ahead-m", type=float, default=0.0, help="결함 위치를 경로를 따라 앞으로 미는 거리")
    r, s = sub.choices["run"], sub.choices["sheets"]
    r.add_argument("--cache", type=Path, default=ROOT / "data/demo_clip_eval", help="demo_clip run 의 탐지 캐시")
    r.add_argument("--model-json", type=Path, default=ROOT / "ai/models/detector_v3.json")
    r.add_argument("--no-merge", action="store_true", help="트랙 합치기 없이 탐지마다 관측 (비교용)")
    r.add_argument("--json", type=Path, default=None)
    s.add_argument("--out", type=Path, default=ROOT / "data/ab_overlap")
    s.add_argument("--max-m", type=float, default=max(RADII))
    args = ap.parse_args(argv)
    if args.cmd == "sheets":
        sheets(args)
        return
    res = run(args)
    for k in ("A", "B"):
        print(k, " ".join(f"{e['id']}@{e['lat']},{e['lon']}{'*' if e['extrap'] else ''}" for e in res["gt"][k]))
    print("정답 짝:", json.dumps(res["gt_pairs_within"], ensure_ascii=False))
    print("관측:", res["detections"]["obs"])
    for r_, s_ in res["detections"]["server"].items():
        print(r_, {k: s_[k] for k in ("defects", "by_type", "two_devices")})
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
        print(f"→ {args.json}")


if __name__ == "__main__":
    main()
