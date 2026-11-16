"""중복 정리 거리 기준 실험 (B8, docs/DESIGN.md 8장).

정답을 아는 가짜 데이터로 잰다. 결함 K개를 흩뿌리고 차량 3대가 각각 GPS 오차를 갖고 한 번씩 본 것처럼 만든 뒤,
거리 기준을 바꿔 가며 서버의 중복 정리(server/dedup.py)에 넣어 결함이 몇 건이 되는지 센다.

  나뉨: 실제 결함 하나가 서버에서 둘 이상으로 갈라진 수 (거리 기준이 좁거나 GPS 오차가 클 때)
  합쳐짐: 서버 결함 하나에 실제 결함 둘 이상이 섞인 수 (거리 기준이 넓거나 결함이 가까이 있을 때)

주의: DB 를 비운다. 실행: uv run python experiments/dedup_distance.py --yes
"""

from __future__ import annotations

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


def run_once(noise: float, distance: float, seed: int) -> tuple[int, int, int]:
    rng = random.Random(seed)
    plan = make_plan(37.5154, 126.9521, DEFECTS, DEVICES, noise, RADIUS_M, ["pothole"], rng)
    items = [(label, truth, it) for label, its in plan.items() for truth, it in enumerate(its)]
    rng.shuffle(items)  # 여러 차량이 섞여서 들어오는 상황
    rules.DEDUP_DISTANCE_M["pothole"] = distance
    db.reset_db()
    with db.connect() as conn:
        trips = {}
        for label in plan:
            vid = db.demo_vehicle_id(label)
            tid = conn.execute("INSERT INTO trips (vehicle_id, started_at) VALUES (%s, %s) RETURNING id",
                               (vid, START)).fetchone()["id"]
            trips[label] = (tid, vid)
        truth_to_server: dict[int, set] = defaultdict(set)
        server_to_truth: dict = defaultdict(set)
        for label, truth, it in items:
            r = dedup.ingest(conn, observation(it, *trips[label], START), None, None)
            truth_to_server[truth].add(r.defect_id)
            server_to_truth[r.defect_id].add(truth)
    split = sum(len(s) - 1 for s in truth_to_server.values())
    merged = sum(1 for t in server_to_truth.values() if len(t) > 1)
    return len(server_to_truth), split, merged


def main() -> None:
    if "--yes" not in sys.argv:
        sys.exit("DB 를 비운다. 괜찮으면 --yes 를 붙여 다시 실행한다")
    db.init_db()
    print(f"결함 {DEFECTS}개, 기기 {DEVICES}대, 반지름 {RADIUS_M}m, 씨앗 {len(SEEDS)}개 평균")
    print("| GPS 오차(m) | " + " | ".join(f"기준 {d}m" for d in DISTANCES) + " |")
    print("|---|" + "---|" * len(DISTANCES))
    for noise in NOISES:
        cells = []
        for d in DISTANCES:
            rs = [run_once(noise, d, s) for s in SEEDS]
            n = sum(r[0] for r in rs) / len(rs)
            split = sum(r[1] for r in rs) / len(rs)
            merged = sum(r[2] for r in rs) / len(rs)
            cells.append(f"{n:.1f}건 (나뉨 {split:.1f}, 합쳐짐 {merged:.1f})")
        print(f"| {noise} | " + " | ".join(cells) + " |")
    db.reset_db()
    print("DB 를 비웠다 (시민·차량은 다시 넣었다)")


if __name__ == "__main__":
    main()
