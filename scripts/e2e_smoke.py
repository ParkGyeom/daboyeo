"""첫 end-to-end 통합 테스트 (계획 2.8 · 10장 3번, 수용 기준 AC-1 · AC-4 · AC-6). 11/21 부터 매일 1회.

한 번 실행으로:
  1. 시연 DB(daboyeo)를 건드리지 않게 같은 PostGIS 컨테이너에 따로 둔 DB(기본 daboyeo_e2e)를 만들고 비운다
     (server/db.py 의 init_db · reset_db 를 그대로 쓴다: 001_init.sql + 002~ 적용, 시연 시민·차량 3대를 넣음).
  2. 그 DB 로 서버를 포트 8010 에 띄우고 /health 를 기다린다. 끝나면 반드시 끈다.
  3. 가상 블랙박스 3대를 edge.blackbox.variants 로 동시에 돌린다 (scripts/demo.sh replay 와 같은 호출).
     탐지기는 replay (모델 추론 없이 기록대로), 배속 0, 주행 종료까지.
  4. 블랙박스 출력과 DB 를 비교해 숫자 표를 출력한다. 하나라도 실패하면 종료 코드 1, 파일·Docker 가 없으면 2.

  uv run python scripts/e2e_smoke.py
  uv run python scripts/e2e_smoke.py --record data/replay/gn_b_3320-3620.v3tiles.jsonl --video data/videos/gn_b_3320-3620.mp4

판정:
  AC-1 4종 각 1건 이상. 포트홀·균열은 DB 결함 수. 가드레일·표지판은 블랙박스가 서버로 보내지 않는다
       (edge/blackbox/detect.py 의 SENT_TYPES, 시설물은 대체안대로 화면에만) → 블랙박스가 본 탐지 기록에서
       같은 유형이 GAP_S 초 안에 이어지면 1건으로 세어 '블랙박스 탐지 건수'로 판정한다.
  AC-4 블랙박스가 보낸 관측 수(각 기기 '보낸 관측 N건' 줄의 합) == DB observations 수, 전송 실패 0, 주행 종료 3대.
  AC-6 라벨(experiments/gt_events/<영상>.csv, amb 제외) 중 2대 이상이 관측한 '공통 결함'이 DB 결함 하나로
       묶인 비율. 관측은 같은 유형이고 영상 시각(t_s)이 라벨 구간 ±LABEL_TOL_S 안이면 그 라벨로 본다.
       --ac6-min (기본 0.5, 제안: 묶기가 망가지면 잡히는 바닥값. plan 의 AC-6 은 "비율 제시"만 요구) 이상이면 통과.

기록 파일이 없으면 onnx 로 한 번 만든다 (시연 설정: v3 + 도로 조각 2, 탐지 주기 10):
  uv run python -m edge.blackbox.run --video data/videos/gn_b_3320-3620.mp4 --model ai/models/detector_v3.onnx \\
      --detect-every 10 --speed 0 --dry-run --live-fps 0 --no-clip --keep-dir "" \\
      --record data/replay/gn_b_3320-3620.v3tiles.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VIDEO = "data/videos/gn_b_3320-3620.mp4"
DEFAULT_RECORD = "data/replay/gn_b_3320-3620.v3tiles.jsonl"
DEFAULT_DB = "daboyeo_e2e"
DEMO_DB = "daboyeo"
DEVICES = 3
GAP_S = 1.0  # edge/blackbox/track_merge.py 의 GAP_S 와 같게: 이만큼 안 보이면 다른 탐지
LABEL_TOL_S = 0.5  # 라벨 구간 앞뒤 여유 (handoff 18장 정답 창 ±0.5초)
ROAD_TYPES = ("pothole", "crack")  # 서버로 가는 노면 결함 (lane_wear 는 시연에서 끔)
FACILITY_TYPES = ("guardrail", "sign")  # 탐지 클래스 이름 (서버 유형은 guardrail_damage · sign_damage)
KO = {"pothole": "포트홀", "crack": "균열", "guardrail": "가드레일", "sign": "표지판", "lane_wear": "차선 마모"}

# variants.py 가 붙이는 "[bb-01]" 머리
_DONE = re.compile(r"^\[(?P<dev>[^\]]+)\] 끝\. .*보낸 관측 (?P<sent>\d+)건 .*전송 실패 (?P<failed>\d+)")
_ENDED = re.compile(r"^\[(?P<dev>[^\]]+)\] 주행 종료: 시민에게 보여줄 결함 (?P<n>\d+)건")
_SENT = re.compile(r"^\[(?P<dev>[^\]]+)\]\s+\S+: (?P<type>[a-z_]+) [0-9.]+.* → 결함 ")


# ---------- 순수 함수 (scripts/tests/test_e2e_smoke.py) ----------

def parse_blackbox_output(text: str) -> dict[str, dict]:
    """variants.py 출력 → 기기별 {"sent", "failed", "ended", "findings", "types"}.

    sent·failed 는 run.py 가 끝날 때 찍는 '보낸 관측 N건 … 전송 실패 M' (Sender.sent · Sender.failed).
    types 는 서버가 결함 번호를 돌려준 관측 줄('→ 결함 …')을 유형별로 센 것 (sent 와 같아야 한다).
    """
    out: dict[str, dict] = defaultdict(lambda: {"sent": None, "failed": None, "ended": False, "findings": None,
                                                "types": Counter()})
    for line in text.splitlines():
        if m := _DONE.match(line):
            out[m["dev"]]["sent"], out[m["dev"]]["failed"] = int(m["sent"]), int(m["failed"])
        elif m := _ENDED.match(line):
            out[m["dev"]]["ended"], out[m["dev"]]["findings"] = True, int(m["n"])
        elif m := _SENT.match(line):
            out[m["dev"]]["types"][m["type"]] += 1
    return dict(out)


def record_events(lines: list[str], gap_s: float = GAP_S) -> Counter:
    """탐지 기록(JSONL, 첫 줄 머리) → 유형별 탐지 건수. 같은 유형이 gap_s 초 안에 다시 나오면 같은 1건."""
    last: dict[str, float] = {}
    events: Counter = Counter()
    rows = sorted((json.loads(ln) for ln in lines[1:] if ln.strip()), key=lambda r: float(r["t"]))
    for r in rows:
        t = float(r["t"])
        for k in {d[0] for d in r["dets"]}:
            if k not in last or t - last[k] > gap_s:
                events[k] += 1
            last[k] = t
    return events


@dataclass(frozen=True)
class Label:
    id: str
    t0: float
    t1: float
    type: str
    amb: bool


def load_labels(text: str) -> list[Label]:
    """정답 라벨 CSV (defect_id,t_start_s,t_end_s,defect_type,memo, # 주석 줄 허용)."""
    rows = csv.DictReader(ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#"))
    return [Label(r["defect_id"], float(r["t_start_s"]), float(r["t_end_s"]), r["defect_type"],
                  (r.get("memo") or "").startswith("amb")) for r in rows]


def match_label(defect_type: str, t_s: float, labels: list[Label], tol: float = LABEL_TOL_S) -> str | None:
    """같은 유형이고 구간 ±tol 안인 라벨. 여럿이면 구간 가운데가 가장 가까운 것."""
    hits = [lb for lb in labels if lb.type == defect_type and lb.t0 - tol <= t_s <= lb.t1 + tol]
    if not hits:
        return None
    return min(hits, key=lambda lb: abs((lb.t0 + lb.t1) / 2 - t_s)).id


def ac6_summary(obs: list[tuple[str, str, float, str]], labels: list[Label], tol: float = LABEL_TOL_S) -> dict:
    """obs: (차량, 유형, t_s, 결함 id). amb 라벨은 뺀다 (amb 에 붙은 관측은 '애매'로 따로 센다).

    공통 결함 = 2대 이상이 관측한 라벨. 하나로 묶임 = 그 라벨의 관측(모든 차량)이 결함 1개에만 들어감.
    """
    strict = [lb for lb in labels if not lb.amb]
    amb = [lb for lb in labels if lb.amb]
    vehicles: dict[str, set] = defaultdict(set)
    defects: dict[str, set] = defaultdict(set)
    unmatched = amb_obs = 0
    for vehicle, dtype, t_s, defect_id in obs:
        lid = match_label(dtype, t_s, strict, tol)
        if lid is None:
            if match_label(dtype, t_s, amb, tol) is not None:
                amb_obs += 1
            else:
                unmatched += 1
            continue
        vehicles[lid].add(vehicle)
        defects[lid].add(defect_id)
    common = sorted(lid for lid, v in vehicles.items() if len(v) >= 2)
    merged = [lid for lid in common if len(defects[lid]) == 1]
    return {
        "labels": len([lb for lb in strict if lb.type in {o[1] for o in obs} | set(ROAD_TYPES)]),
        "seen": len(vehicles),
        "common": len(common),
        "merged": len(merged),
        "ratio": (len(merged) / len(common)) if common else None,
        "split": {lid: len(defects[lid]) for lid in common if len(defects[lid]) > 1},
        "unmatched_obs": unmatched,
        "amb_obs": amb_obs,
    }


def judge(devices: dict[str, dict], expected_devices: list[str], db_obs: int, db_by_type: Counter,
          facility_events: Counter, ac6: dict, ac6_min: float) -> list[tuple[str, str, str, str, bool]]:
    """판정 표의 줄: (기준, 항목, 값, 통과 조건, 통과)."""
    rows = []
    for t in ROAD_TYPES:
        n = db_by_type.get(t, 0)
        rows.append(("AC-1", f"{KO[t]} 결함 (DB)", str(n), "≥ 1", n >= 1))
    for t in FACILITY_TYPES:
        n = facility_events.get(t, 0)
        rows.append(("AC-1", f"{KO[t]} 탐지 (블랙박스, 서버 안 감)", str(n), "≥ 1", n >= 1))
    leaked = db_by_type.get("guardrail_damage", 0) + db_by_type.get("sign_damage", 0)
    rows.append(("AC-1", "시설물 결함 (DB, SENT_TYPES 확인)", str(leaked), "= 0", leaked == 0))

    missing = [d for d in expected_devices if devices.get(d, {}).get("sent") is None]
    sent = sum(devices[d]["sent"] for d in expected_devices if d not in missing)
    failed = sum(devices[d]["failed"] for d in expected_devices if d not in missing)
    ended = sum(1 for d in expected_devices if devices.get(d, {}).get("ended"))
    per_dev = " · ".join(f"{d} {devices[d]['sent']}" for d in expected_devices if d not in missing)
    rows.append(("AC-4", "끝까지 돈 블랙박스", f"{len(expected_devices) - len(missing)}/{len(expected_devices)}",
                 "전부", not missing))
    rows.append(("AC-4", "주행 종료", f"{ended}/{len(expected_devices)}", "전부", ended == len(expected_devices)))
    rows.append(("AC-4", f"보낸 관측 ({per_dev})", str(sent), "> 0", sent > 0))
    rows.append(("AC-4", "DB 저장 관측", str(db_obs), f"= {sent}", not missing and db_obs == sent))
    rows.append(("AC-4", "전송 실패", str(failed), "= 0", failed == 0))

    rows.append(("AC-6", "라벨 결함 중 관측됨 (amb 제외)", f"{ac6['seen']}/{ac6['labels']}", "-", True))
    rows.append(("AC-6", "2대 이상이 본 공통 결함", str(ac6["common"]), "≥ 1", ac6["common"] >= 1))
    ratio = ac6["ratio"]
    rows.append(("AC-6", "공통 결함 중 하나로 묶임", "-" if ratio is None else
                 f"{ac6['merged']}/{ac6['common']} ({ratio * 100:.0f}%)", f"≥ {ac6_min * 100:.0f}%",
                 ratio is not None and ratio >= ac6_min))
    return rows


def format_table(rows: list[tuple[str, str, str, str, bool]]) -> str:
    out = ["| 기준 | 항목 | 값 | 통과 조건 | 결과 |", "|---|---|---|---|---|"]
    out += [f"| {a} | {b} | {c} | {d} | {'통과' if ok else '**실패**'} |" for a, b, c, d, ok in rows]
    return "\n".join(out)


# ---------- 실행 ----------

def _say(msg: str) -> None:
    print(msg, flush=True)


def _db_url(name: str) -> str:
    # 아직 DATABASE_URL 을 바꾸기 전: 시연 DB 주소에서 이름만 바꾼다
    from psycopg import conninfo

    from server import config

    info = conninfo.conninfo_to_dict(config.DATABASE_URL)
    return conninfo.make_conninfo(**{**info, "dbname": name})


def _wait(check, timeout: float, what: str) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if check():
            return
        time.sleep(1)
    raise TimeoutError(f"{what} 이(가) {timeout:.0f}초 안에 준비되지 않았다")


def _port_busy(port: int) -> bool:
    import socket

    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def ensure_docker_db(url: str) -> int:
    """PostGIS 컨테이너를 띄우고 연결될 때까지 기다린다. 실패하면 종료 코드 2 를 돌려준다 (성공은 0)."""
    import psycopg
    from psycopg import conninfo

    admin = conninfo.make_conninfo(**{**conninfo.conninfo_to_dict(url), "dbname": "postgres"})

    def up() -> bool:
        try:
            with psycopg.connect(admin, connect_timeout=2):
                return True
        except psycopg.OperationalError:
            return False

    if up():
        return 0
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        _say("오류: DB 에 닿지 않고 Docker 도 꺼져 있다. Docker Desktop 을 켠 뒤 다시 실행한다")
        return 2
    _say("DB 띄우는 중 (docker compose up -d db)...")
    subprocess.run(["docker", "compose", "up", "-d", "db"], cwd=ROOT, check=False, capture_output=True)
    try:
        _wait(up, 120, "DB")
    except TimeoutError as e:
        _say(f"오류: {e} (docker compose logs db)")
        return 2
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="주행 영상 (GPS 는 옆의 <영상 이름>.gps.csv)")
    ap.add_argument("--gps", default=None, help="GPS CSV. 없으면 영상 옆의 <영상 이름>.gps.csv")
    ap.add_argument("--record", default=DEFAULT_RECORD, help="탐지 기록 JSONL (replay)")
    ap.add_argument("--labels", default=None, help="정답 라벨 CSV. 없으면 experiments/gt_events/<영상 이름>.csv")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"쓸 DB 이름 (시작할 때 비운다). 시연 DB '{DEMO_DB}' 는 안 된다")
    ap.add_argument("--port", type=int, default=8010, help="서버 포트 (시연 서버 8000 과 다르게)")
    ap.add_argument("--ac6-min", type=float, default=0.5, help="AC-6 공통 결함 중 하나로 묶인 비율의 통과 기준 (제안)")
    ap.add_argument("--timeout", type=float, default=900, help="블랙박스 3대가 끝나기를 기다릴 최대 초")
    ap.add_argument("--out", default=None, help="로그·사진을 둘 폴더. 없으면 server_data/e2e/<시각>/")
    a = ap.parse_args(argv)

    def path(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else ROOT / q

    video = path(a.video)
    gps = path(a.gps) if a.gps else video.with_name(video.stem + ".gps.csv")
    record = path(a.record)
    labels_path = path(a.labels) if a.labels else ROOT / "experiments" / "gt_events" / f"{video.stem}.csv"
    missing = [f"{what}: {p}" for what, p in (("영상", video), ("GPS", gps), ("탐지 기록", record),
                                               ("정답 라벨", labels_path)) if not p.exists()]
    if missing:
        _say("필요한 파일이 없다:\n  " + "\n  ".join(missing))
        if not record.exists():
            _say("탐지 기록은 이 스크립트 맨 위 설명의 onnx --record 명령으로 만든다")
        return 2
    if a.db == DEMO_DB:
        _say(f"오류: 시연 DB '{DEMO_DB}' 는 비우지 않는다. --db 로 다른 이름을 준다")
        return 2
    rec_lines = [ln for ln in record.read_text(encoding="utf-8").splitlines() if ln.strip()]
    head = json.loads(rec_lines[0])
    every = int(head["detect_every"])

    out = path(a.out) if a.out else ROOT / "server_data" / "e2e" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    url = _db_url(a.db)
    code = ensure_docker_db(url)
    if code:
        return code
    if _port_busy(a.port):
        _say(f"오류: 포트 {a.port} 을(를) 이미 쓰고 있다. --port 로 다른 포트를 준다")
        return 2

    # 이 프로세스의 server.* 도 e2e DB 를 보게 한다 (server.config 는 처음 불릴 때 읽는다)
    os.environ["DATABASE_URL"] = url
    import importlib

    from server import config

    importlib.reload(config)
    from server import db

    importlib.reload(db)
    db.init_db()
    db.reset_db()
    _say(f"DB {a.db}: 스키마 적용·비움 (시민·차량 3대 다시 넣음)")

    env = {**os.environ, "DATABASE_URL": url, "SNAPSHOT_DIR": str(out / "snapshots"), "CLIP_DIR": str(out / "clips"),
           # 결함 주소(네이버 Reverse Geocoding)를 받지 않는다: 바깥으로 요청을 보내지 않고, 키가 없어도 같은 결과
           "NAVER_MAP_KEY_ID": "", "NAVER_MAP_CLIENT_SECRET": "", "PYTHONUNBUFFERED": "1"}
    server_log = (out / "server.log").open("w")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.main:app", "--port", str(a.port)],
                              cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)
    base = f"http://localhost:{a.port}"
    try:
        import httpx

        def healthy() -> bool:
            if server.poll() is not None:
                raise RuntimeError(f"서버가 바로 꺼졌다. {out / 'server.log'} 를 본다")
            try:
                return httpx.get(f"{base}/health", timeout=2).status_code == 200
            except httpx.HTTPError:
                return False

        _wait(healthy, 60, "서버")
        _say(f"서버 준비됨 {base} (DB {a.db})")

        cmd = [sys.executable, "-u", "-m", "edge.blackbox.variants", "--devices", str(DEVICES), "--video", str(video),
               "--gps", str(gps), "--detector", "replay", "--record", str(record), "--detect-every", str(every),
               "--speed", "0", "--server", base,
               # 시연 화면(server_data/live)·블랙박스 보관 폴더를 건드리지 않는다
               "--live-fps", "0", "--keep-dir", ""]
        _say(f"가상 블랙박스 {DEVICES}대: replay {record.name} (탐지 주기 {every}), 배속 0 ...")
        t0 = time.time()
        try:
            bb = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"}, capture_output=True,
                                text=True, timeout=a.timeout, check=False)
            bb_out, bb_code = bb.stdout + bb.stderr, bb.returncode
        except subprocess.TimeoutExpired as e:
            bb_out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            bb_code = -1
            _say(f"블랙박스가 {a.timeout:.0f}초 안에 끝나지 않았다")
        (out / "blackbox.log").write_text(bb_out, encoding="utf-8")
        _say(f"블랙박스 끝 (종료 코드 {bb_code}, {time.time() - t0:.0f}초). 로그 {out / 'blackbox.log'}")

        with db.connect() as conn:
            db_obs = conn.execute("SELECT count(*) AS n FROM observations").fetchone()["n"]
            obs_types = Counter({r["t"]: r["n"] for r in conn.execute(
                "SELECT defect_type::text AS t, count(*) AS n FROM observations GROUP BY 1")})
            db_by_type = Counter({r["t"]: r["n"] for r in conn.execute(
                "SELECT defect_type::text AS t, count(*) AS n FROM defects GROUP BY 1")})
            multi = conn.execute("SELECT count(*) FILTER (WHERE vehicle_count >= 2) AS m, count(*) AS n "
                                 "FROM defects").fetchone()
            per_vehicle = {r["label"]: r["n"] for r in conn.execute(
                "SELECT v.label, count(*) AS n FROM observations o JOIN vehicles v ON v.id = o.vehicle_id GROUP BY 1")}
            obs_rows = [(r["label"], r["t"], float(r["t_s"]), str(r["defect_id"])) for r in conn.execute(
                "SELECT v.label, o.defect_type::text AS t, o.t_s, o.defect_id FROM observations o "
                "JOIN vehicles v ON v.id = o.vehicle_id")]
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
        _say("서버 끔")

    devices = parse_blackbox_output(bb_out)
    expected = [f"bb-0{i}" for i in range(1, DEVICES + 1)]
    facility = record_events(rec_lines)
    labels = load_labels(labels_path.read_text(encoding="utf-8"))
    ac6 = ac6_summary(obs_rows, labels)
    rows = judge(devices, expected, db_obs, db_by_type, facility, ac6, a.ac6_min)
    if bb_code != 0:
        rows.append(("-", "블랙박스 종료 코드", str(bb_code), "= 0", False))

    _say("")
    _say(f"## e2e smoke — {video.name}, replay {record.name} ({head.get('model', '?')}), 기기 {DEVICES}대, DB {a.db}")
    _say("")
    _say(format_table(rows))
    _say("")
    _say("참고 (판정 아님)")
    _say("  기기별 보낸/저장 관측: " + " · ".join(
        f"{d} {devices.get(d, {}).get('sent')}/{per_vehicle.get(d, 0)}" for d in expected))
    _say("  유형별 관측 (DB): " + ", ".join(f"{KO.get(t, t)} {n}" for t, n in sorted(obs_types.items())))
    _say("  유형별 결함 (DB): " + ", ".join(f"{KO.get(t, t)} {n}" for t, n in sorted(db_by_type.items())))
    _say("  블랙박스 탐지 건수 (기록, 1대분): " + ", ".join(f"{KO.get(t, t)} {n}" for t, n in sorted(facility.items())))
    n_def = multi["n"]
    if n_def:
        _say(f"  결함 {n_def}건 중 2대 이상이 본 결함 {multi['m']}건 ({multi['m'] / n_def * 100:.0f}%), "
             f"중복 정리 비율 (1 − 결함 ÷ 관측) {(1 - n_def / db_obs) * 100:.1f}%" if db_obs else "")
    _say(f"  라벨에 안 맞는 관측 {ac6['unmatched_obs']}건 (헛것 후보), amb 라벨에 맞는 관측 {ac6['amb_obs']}건")
    if ac6["split"]:
        _say("  나뉜 공통 결함 (라벨: 결함 수): " + ", ".join(f"{k}: {v}" for k, v in sorted(ac6["split"].items())))
    _say(f"  로그·사진: {out}")
    ok = all(r[4] for r in rows)
    _say("")
    _say("결과: 통과" if ok else "결과: 실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
