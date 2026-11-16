"""중복 정리 거리 기준 실험 (계획 4.5, 오픈 이슈 #2 반경 확정 근거).

정답을 아는 관측을 거리 기준(반경)을 바꿔 가며 서버의 중복 정리(server/dedup.py)에 넣어 결함이 몇 건이 되는지 센다.

  나뉨: 실제 결함 하나가 서버에서 둘 이상으로 갈라진 수 (반경이 좁거나 GPS 오차가 클 때)
  합쳐짐: 서버 결함 하나에 실제 결함 둘 이상이 섞인 수 (반경이 넓거나 결함이 가까이 있을 때)
  바르게 묶임: 실제 결함 중 서버 결함 하나로만 묶이고 그 결함에 다른 실제 결함이 섞이지 않은 비율

입력 두 가지:
  (기본) 합성: 결함 30개를 반지름 400m 안에 흩뿌리고 차량 3대가 각각 GPS 오차를 갖고 한 번씩 본 것처럼 만든다
         (server/fake.py 의 make_plan). GPS 오차 0~10m × 반경, 씨앗 5개 평균.
  --replay <bandwidth.py 결과 JSON>: 가상 블랙박스가 실제로 보낸 관측(위치·유형·영상 시각)을 그대로 다시 넣는다.
         가짜 탐지(--fake-every)로 잰 결과면 영상 시각으로 정답 결함을 알 수 있어 나뉨·합쳐짐도 센다.
         모델 탐지 결과면 정답이 없어 결함 수만 센다.

주의: DB 를 비운다. 측정이 끝나면 시연용 시민·차량을 다시 넣고 비운 상태로 둔다.
  uv run python experiments/dedup_distance.py --yes
  uv run python experiments/dedup_distance.py --yes --replay experiments/bandwidth_gn_a_demo_fake.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datetime import datetime, timedelta, timezone  # noqa: E402

from server import db, dedup, rules  # noqa: E402
from server.fake import make_plan, observation  # noqa: E402

START = datetime(2026, 11, 20, 10, 0, tzinfo=timezone(timedelta(hours=9)))

DEFECTS = 30  # 반지름 400m 안에 30개: 가장 가까운 결함까지 평균 약 65m, 일부는 20~40m 안에 붙어 있다
DEVICES = 3
RADIUS_M = 400
NOISES = [0, 3, 5, 7, 10]
DISTANCES = [10, 15, 20, 30, 40]
SEEDS = [1, 2, 3, 4, 5]


def ingest_all(items: list[tuple[str, int | None, dict]], distance: float) -> dict:
    """items: (차량 이름, 정답 결함 번호 또는 None, fake.make_plan 모양의 관측). 넣은 순서대로 묶는다."""
    for t in rules.DEDUP_DISTANCE_M:
        rules.DEDUP_DISTANCE_M[t] = distance
    db.reset_db()
    with db.connect() as conn:
        trips = {}
        for label in sorted({it[0] for it in items}):
            vid = db.demo_vehicle_id(label)
            tid = conn.execute("INSERT INTO trips (vehicle_id, started_at) VALUES (%s, %s) RETURNING id",
                               (vid, START)).fetchone()["id"]
            trips[label] = (tid, vid)
        truth_to_server: dict[int, set] = defaultdict(set)
        server_to_truth: dict = defaultdict(set)
        for label, truth, it in items:
            r = dedup.ingest(conn, observation(it, *trips[label], START), None, None)
            server_to_truth[r.defect_id].add(truth)
            if truth is not None:
                truth_to_server[truth].add(r.defect_id)
    out = {"observations": len(items), "defects": len(server_to_truth)}
    if truth_to_server:
        out["split"] = sum(len(s) - 1 for s in truth_to_server.values())
        out["merged"] = sum(1 for t in server_to_truth.values() if len(t) > 1)
        out["correct"] = sum(1 for s in truth_to_server.values()
                             if len(s) == 1 and len(server_to_truth[next(iter(s))]) == 1) / len(truth_to_server)
        out["truth"] = len(truth_to_server)
    return out


def synthetic(noise: float, distance: float, seed: int) -> dict:
    rng = random.Random(seed)
    plan = make_plan(37.5154, 126.9521, DEFECTS, DEVICES, noise, RADIUS_M, ["pothole"], rng)
    items = [(label, truth, it) for label, its in plan.items() for truth, it in enumerate(its)]
    rng.shuffle(items)  # 여러 차량이 섞여서 들어오는 상황
    return ingest_all(items, distance)


def replay_items(path: Path) -> tuple[list, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    obs = d.get("server", {}).get("observations_list")
    if not obs:
        sys.exit(f"{path} 에 관측 목록이 없다 (bandwidth.py 를 --dry-run 없이 서버로 보내며 돌린 결과여야 한다)")
    fake = d.get("fake")
    items = []
    for o in obs:
        truth = None
        if fake:
            k, r = divmod(o["t_s"] - fake["first"], fake["every"])
            truth = int(k) if r < fake["visible"] + 0.5 else None
        items.append((o["vehicle"], truth, {"defect_type": o["type"], "confidence": o["confidence"], "bbox": None,
                                            "lat": o["lat"], "lon": o["lon"], "t_s": o["t_s"]}))
    return items, d


def cell(r: dict) -> str:
    if "correct" not in r:
        return f"{r['defects']:.1f}건"
    return f"{r['defects']:.1f}건 (나뉨 {r['split']:.1f}, 합쳐짐 {r['merged']:.1f}, 바르게 {r['correct'] * 100:.0f}%)"


def mean(rs: list[dict]) -> dict:
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="DB 를 비워도 된다")
    ap.add_argument("--replay", default=None, help="experiments/bandwidth.py 결과 JSON")
    ap.add_argument("--distances", default=",".join(map(str, DISTANCES)), help="비교할 반경(m), 쉼표로")
    a = ap.parse_args()
    if not a.yes:
        sys.exit("DB 를 비운다. 괜찮으면 --yes 를 붙여 다시 실행한다")
    distances = [float(x) for x in a.distances.split(",")]
    saved = dict(rules.DEDUP_DISTANCE_M)
    db.init_db()
    try:
        if a.replay:
            items, d = replay_items(Path(a.replay))
            print(f"입력: {d['video']} ({d['video_duration_s']}초), 기기 {len(d['devices'])}대, GPS {d['gps']}, "
                  f"탐지 {d['detector']}")
            if d.get("note"):
                print(f"메모: {d['note']}")
            print(f"관측 {len(items)}건" + (f", 정답 결함 {len({t for _, t, _ in items if t is not None})}개"
                                           if d.get("fake") else " (정답 없음: 결함 수만)"))
            print("| 반경 | 결함 수 | 중복 정리 비율 (1 − 결함 ÷ 관측) |")
            print("|---|---|---|")
            for dist in distances:
                r = ingest_all(items, dist)
                print(f"| {dist:g}m | {cell(r)} | {(1 - r['defects'] / r['observations']) * 100:.1f}% |")
        else:
            print(f"합성: 결함 {DEFECTS}개, 기기 {DEVICES}대, 반지름 {RADIUS_M}m, 씨앗 {len(SEEDS)}개 평균")
            print("| GPS 오차(m) | " + " | ".join(f"반경 {d:g}m" for d in distances) + " |")
            print("|---|" + "---|" * len(distances))
            for noise in NOISES:
                cells = [cell(mean([synthetic(noise, d, s) for s in SEEDS])) for d in distances]
                print(f"| {noise} | " + " | ".join(cells) + " |")
    finally:
        rules.DEDUP_DISTANCE_M.update(saved)
        db.reset_db()
    print("DB 를 비웠다 (시민·차량은 다시 넣었다)")


if __name__ == "__main__":
    main()
