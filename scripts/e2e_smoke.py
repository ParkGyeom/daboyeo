"""첫 end-to-end 통합 테스트 (계획 2.8 · 10장 3번, 수용 기준 AC-1 · AC-4 · AC-6). 11/21 부터 매일 1회.

영상 2개(시연 클립 gn_b · gn_a) × 각 3대 = 가상 블랙박스 6대를 동시에 돌리고 DB 를 확인한다. 한 번 실행으로:
  1. DB 를 만들고 비운다 (server/db.py 의 init_db · reset_db 그대로: 001_init.sql + 002~ 적용, 시연 시민·차량 3대).
     기본은 시연 DB(daboyeo)를 건드리지 않는 따로 둔 DB daboyeo_e2e. --db daboyeo 로 시연 DB 에서도 돌릴 수 있는데,
     그때는 비우기 전에 확인을 받는다 (--yes 면 묻지 않음). 영상 2개째의 차량 bb-04~06 (시민 D~F) 은 이 스크립트가 넣는다.
  2. 그 DB 로 서버를 포트 8010 에 띄우고 /health 를 기다린다. 끝나면 반드시 끈다.
  3. 영상마다 3대를 edge.blackbox.run 으로 동시에 돌린다 (6대 동시). 기기별 조건(GPS 오차·밝기·시작 시점)은
     edge/blackbox/variants.py 의 VARIANTS 그대로, 기기 이름만 영상 2개째에서 bb-04~06. 탐지기는 replay
     (모델 추론 없이 기록대로), 배속 0, 주행 종료까지.
  4. 블랙박스 출력과 DB 를 비교해 숫자 표를 출력한다. 하나라도 실패하면 종료 코드 1, 파일·Docker 가 없거나 확인을 받지 못하면 2.

  uv run python scripts/e2e_smoke.py
  uv run python scripts/e2e_smoke.py --videos gn_b_3320-3620           # 영상 하나만 (3대)
  uv run python scripts/e2e_smoke.py --db daboyeo --yes                # 시연 DB 에서 (비운다)
  uv run python scripts/e2e_smoke.py --data-root ~/daboyeo             # data/ 가 다른 폴더(작업 폴더 밖)에 있을 때

영상 <이름> 마다 쓰는 파일: data/videos/<이름>.mp4 · data/videos/<이름>.gps.csv · data/replay/<이름>.v3tiles.jsonl
(--data-root 아래) · experiments/gt_events/<이름>.csv (레포).

판정:
  AC-1 4종 각 1건 이상. 포트홀·균열은 DB 결함 수 (6대 합). 가드레일·표지판은 블랙박스가 서버로 보내지 않는다
       (edge/blackbox/detect.py 의 SENT_TYPES, 시설물은 대체안대로 화면에만) → run.py 가 끝날 때 찍는
       '[bb-xx] 보내지 않은 탐지 guardrail N · sign M (트랙, 합치기 1초)' 줄을 영상마다 첫 차량(1대분)에서 읽어 더한다.
       같은 영상 3대를 다 더하면 같은 시설물을 세 번 세게 되기 때문이다.
  AC-4 블랙박스가 보낸 관측 수(각 기기 '보낸 관측 N건' 줄의 합, 6대) == DB observations 수, 전송 실패 0, 주행 종료 6대.
  AC-6 (같은 영상) 라벨(experiments/gt_events, amb 제외) 중 그 영상 2대 이상이 관측한 '공통 결함'이 DB 결함
       하나로 묶인 비율. 관측은 같은 유형이고 영상 시각(t_s)이 라벨 구간 ±LABEL_TOL_S 안이면 그 라벨로 본다.
       두 영상을 합친 비율이 --ac6-min (기본 0.5, 제안: 묶기가 망가지면 잡히는 바닥값. plan 의 AC-6 은 "비율 제시"만 요구) 이상이면 통과.
       (다른 영상) 두 영상의 관측이 함께 들어간 DB 결함 수. 두 영상의 GPS 경로가 묶기 반경 안으로 만나지 않으면
       0 이어야 통과 (잘못 묶임). 경로가 겹치면 같은 실제 결함일 수 있어 판정하지 않고 수와 라벨만 보인다.

탐지 기록이 없으면 onnx 로 한 번 만든다 (시연 설정: v3 + 도로 조각 2, 탐지 주기 10). 없을 때 명령을 출력한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_VIDEOS = ("gn_b_3320-3620", "gn_a_2050-2510")
RECORD_SUFFIX = ".v3tiles.jsonl"
DEFAULT_DB = "daboyeo_e2e"
DEMO_DB = "daboyeo"
PER_VIDEO = 3  # 영상마다 기기 수 (variants.py 의 VARIANTS 3가지)
LABEL_TOL_S = 0.5  # 라벨 구간 앞뒤 여유 (handoff 18장 정답 창 ±0.5초)
DEDUP_M = 30.0  # server/rules.py 의 노면 결함 묶기 반경 (11/17 확정). 경로가 이만큼 안으로 만나면 '겹친다'
ROAD_TYPES = ("pothole", "crack")  # 서버로 가는 노면 결함 (lane_wear 는 시연에서 끔)
FACILITY_TYPES = ("guardrail", "sign")  # 탐지 클래스 이름 (서버 유형은 guardrail_damage · sign_damage)
KO = {"pothole": "포트홀", "crack": "균열", "guardrail": "가드레일", "sign": "표지판", "lane_wear": "차선 마모"}
# 영상 2개째 차량. 시연 시민(bb-01~03, server/db.py DEMO)과 같은 방식으로 넣는다
EXTRA_DEVICES = [("시민 D", "bb-04"), ("시민 E", "bb-05"), ("시민 F", "bb-06")]
RECORD_CMD = ("uv run python -m edge.blackbox.run --video {video} --model ai/models/detector_v3.onnx "
              "--detect-every 10 --speed 0 --dry-run --live-fps 0 --no-clip --keep-dir \"\" --record {record}")

_DONE = re.compile(r"^\[(?P<dev>[^\]]+)\] 끝\. .*보낸 관측 (?P<sent>\d+)건 .*전송 실패 (?P<failed>\d+)")
_ENDED = re.compile(r"^\[(?P<dev>[^\]]+)\] 주행 종료: 시민에게 보여줄 결함 (?P<n>\d+)건")
# run.py 끝 줄 (PR #67): "[bb-01] 보내지 않은 탐지 guardrail 25 · sign 136 (트랙, 합치기 1초)" 또는 "… 탐지 없음 (…)"
_UNSENT = re.compile(r"^\[(?P<dev>[^\]]+)\] 보내지 않은 탐지 (?P<body>.+?) \(트랙")
_SENT = re.compile(r"^\[(?P<dev>[^\]]+)\]\s+\S+: (?P<type>[a-z_]+) [0-9.]+.* → 결함 ")


# ---------- 순수 함수 (scripts/tests/test_e2e_smoke.py) ----------

def device_ids(n_videos: int, per_video: int = PER_VIDEO) -> list[list[str]]:
    """영상별 기기 이름. 영상 1개째 bb-01~03, 2개째 bb-04~06."""
    return [[f"bb-{v * per_video + i + 1:02d}" for i in range(per_video)] for v in range(n_videos)]


def prefix_line(line: str, device: str) -> str:
    """run.py 출력 줄에 기기 머리를 붙인다 (variants.py 와 같이: 이미 '[' 로 시작하면 그대로)."""
    return line if line.startswith("[") else f"[{device}]{line}"


def parse_blackbox_output(text: str) -> dict[str, dict]:
    """기기 머리('[bb-01]')가 붙은 run.py 출력 → 기기별 {"sent", "failed", "ended", "findings", "types", "unsent"}.

    sent·failed 는 run.py 가 끝날 때 찍는 '보낸 관측 N건 … 전송 실패 M' (Sender.sent · Sender.failed).
    types 는 서버가 결함 번호를 돌려준 관측 줄('→ 결함 …')을 유형별로 센 것 (sent 와 같아야 한다).
    unsent 는 '보내지 않은 탐지 …' 줄의 유형별 트랙 수 (run.py 결과 dict 의 unsent_tracks). 줄이 없으면 None.
    """
    out: dict[str, dict] = defaultdict(lambda: {"sent": None, "failed": None, "ended": False, "findings": None,
                                                "types": Counter(), "unsent": None})
    for line in text.splitlines():
        if m := _DONE.match(line):
            out[m["dev"]]["sent"], out[m["dev"]]["failed"] = int(m["sent"]), int(m["failed"])
        elif m := _ENDED.match(line):
            out[m["dev"]]["ended"], out[m["dev"]]["findings"] = True, int(m["n"])
        elif m := _UNSENT.match(line):
            body = m["body"].strip()
            out[m["dev"]]["unsent"] = Counter() if body == "없음" else Counter(
                {k: int(n) for k, n in (part.strip().rsplit(" ", 1) for part in body.split("·"))})
        elif m := _SENT.match(line):
            out[m["dev"]]["types"][m["type"]] += 1
    return dict(out)


def facility_tracks(devices: dict[str, dict], first_devices: list[str]) -> tuple[Counter, list[str]]:
    """영상마다 첫 차량의 '보내지 않은 탐지' 트랙 수를 더한다. (합, 그 줄이 없는 기기)."""
    total: Counter = Counter()
    missing = []
    for d in first_devices:
        u = devices.get(d, {}).get("unsent")
        if u is None:
            missing.append(d)
        else:
            total += u
    return total, missing


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
    """영상 하나의 관측 obs: (차량, 유형, t_s, 결함 id). amb 라벨은 뺀다 (amb 에 붙은 관측은 '애매'로 따로 센다).

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
        "labels": len([lb for lb in strict if lb.type in ROAD_TYPES]),
        "seen": len(vehicles),
        "common": len(common),
        "merged": len(merged),
        "ratio": (len(merged) / len(common)) if common else None,
        "split": {lid: len(defects[lid]) for lid in common if len(defects[lid]) > 1},
        "unmatched_obs": unmatched,
        "amb_obs": amb_obs,
    }


def cross_video_defects(obs_by_video: dict[str, list[tuple[str, str, float, str]]],
                        labels_by_video: dict[str, list[Label]], tol: float = LABEL_TOL_S) -> dict[str, dict]:
    """두 영상 이상의 관측이 함께 들어간 결함 → {결함 id: {영상: [라벨 id 또는 '-' (라벨 없음)]}}."""
    by_defect: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for video, obs in obs_by_video.items():
        strict = [lb for lb in labels_by_video.get(video, []) if not lb.amb]
        for _, dtype, t_s, defect_id in obs:
            by_defect[defect_id][video].add(match_label(dtype, t_s, strict, tol) or "-")
    return {d: {v: sorted(ls) for v, ls in vs.items()} for d, vs in by_defect.items() if len(vs) >= 2}


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def min_track_distance_m(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """두 GPS 경로(위도, 경도 점)의 가장 가까운 거리 (점끼리)."""
    return min(_haversine_m(p, q) for p in a for q in b)


def judge(devices: dict[str, dict], expected_devices: list[str], db_obs: int, db_by_type: Counter,
          facility_events: Counter, ac6_by_video: dict[str, dict], ac6_min: float,
          cross: dict[str, dict] | None = None, tracks_overlap: bool = True,
          facility_missing: list[str] | None = None) -> list[tuple[str, str, str, str, bool]]:
    """판정 표의 줄: (기준, 항목, 값, 통과 조건, 통과). 판정하지 않는 줄은 통과 조건 '-' 에 통과 True."""
    rows = []
    for t in ROAD_TYPES:
        n = db_by_type.get(t, 0)
        rows.append(("AC-1", f"{KO[t]} 결함 (DB)", str(n), "≥ 1", n >= 1))
    for t in FACILITY_TYPES:
        n = facility_events.get(t, 0)
        rows.append(("AC-1", f"{KO[t]} 탐지 트랙 (보내지 않은 탐지, 영상당 1대)", str(n), "≥ 1", n >= 1))
    if facility_missing:
        rows.append(("AC-1", "'보내지 않은 탐지' 줄이 없는 기기", ", ".join(facility_missing), "없음", False))
    leaked = db_by_type.get("guardrail_damage", 0) + db_by_type.get("sign_damage", 0)
    rows.append(("AC-1", "시설물 결함 (DB, SENT_TYPES 확인)", str(leaked), "= 0", leaked == 0))

    missing = [d for d in expected_devices if devices.get(d, {}).get("sent") is None]
    sent = sum(devices[d]["sent"] for d in expected_devices if d not in missing)
    failed = sum(devices[d]["failed"] for d in expected_devices if d not in missing)
    ended = sum(1 for d in expected_devices if devices.get(d, {}).get("ended"))
    n = len(expected_devices)
    rows.append(("AC-4", "끝까지 돈 블랙박스", f"{n - len(missing)}/{n}", "전부", not missing))
    rows.append(("AC-4", "주행 종료", f"{ended}/{n}", "전부", ended == n))
    rows.append(("AC-4", f"보낸 관측 ({n}대 합)", str(sent), "> 0", sent > 0))
    rows.append(("AC-4", "DB 저장 관측", str(db_obs), f"= {sent}", not missing and db_obs == sent))
    rows.append(("AC-4", "전송 실패", str(failed), "= 0", failed == 0))

    for video, s in ac6_by_video.items():
        ratio = "-" if s["ratio"] is None else f"{s['ratio'] * 100:.0f}%"
        value = f"{s['seen']}/{s['labels']} · {s['common']} · {s['merged']} ({ratio})"
        rows.append(("AC-6", f"{video}: 관측된 라벨 · 공통 · 묶임", value, "-", True))
    common = sum(s["common"] for s in ac6_by_video.values())
    merged = sum(s["merged"] for s in ac6_by_video.values())
    rows.append(("AC-6", "같은 영상 2대 이상이 본 공통 결함 (합)", str(common), "≥ 1", common >= 1))
    rows.append(("AC-6", "공통 결함 중 하나로 묶임 (합)",
                 f"{merged}/{common} ({merged / common * 100:.0f}%)" if common else "-",
                 f"≥ {ac6_min * 100:.0f}%", common >= 1 and merged / common >= ac6_min))
    if cross is not None:
        if tracks_overlap:
            rows.append(("AC-6", "다른 영상끼리 묶인 결함 (경로 겹침: 판정 안 함)", str(len(cross)), "-", True))
        else:
            rows.append(("AC-6", "다른 영상끼리 잘못 묶인 결함 (경로 안 겹침)", str(len(cross)), "= 0", not cross))
    return rows


def format_table(rows: list[tuple[str, str, str, str, bool]]) -> str:
    out = ["| 기준 | 항목 | 값 | 통과 조건 | 결과 |", "|---|---|---|---|---|"]
    for a, b, c, d, ok in rows:
        res = "-" if d == "-" else ("통과" if ok else "**실패**")
        out.append(f"| {a} | {b} | {c} | {d} | {res} |")
    return "\n".join(out)


# ---------- 실행 ----------

def _say(msg: str) -> None:
    print(msg, flush=True)


def _db_url(name: str) -> tuple[str, str]:
    """(e2e 에 쓸 DB 주소, 지금 설정의 시연 DB 이름). DATABASE_URL 을 바꾸기 전에 부른다."""
    from psycopg import conninfo

    from server import config

    info = conninfo.conninfo_to_dict(config.DATABASE_URL)
    return conninfo.make_conninfo(**{**info, "dbname": name}), info.get("dbname", DEMO_DB)


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


def _read_track(path: Path) -> list[tuple[float, float]]:
    with path.open(encoding="utf-8") as f:
        return [(float(r["lat"]), float(r["lon"])) for r in csv.DictReader(f)]


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


def confirm_demo_db(name: str, yes: bool) -> bool:
    """시연 DB 를 비워도 되는지. --yes 면 묻지 않고, 터미널이 아니면(예: cron) 거절한다."""
    if yes:
        return True
    if not sys.stdin.isatty():
        _say(f"오류: 시연 DB '{name}' 를 비운다. 괜찮으면 --yes 를 붙여 다시 실행한다")
        return False
    _say(f"주의: 시연 DB '{name}' 의 결함·신고·문자를 모두 지우고 시민 D~F·차량 bb-04~06 을 더한다 "
         "(되돌리려면 scripts/demo.sh reset).")
    try:
        return input("계속하려면 yes 를 입력: ").strip().lower() == "yes"
    except EOFError:
        return False


def add_extra_devices(db, n: int) -> None:
    """영상 2개째 이후의 차량(bb-04~)을 시연 차량과 같은 방식(고정 UUID)으로 넣는다. 여러 번 불러도 안전."""
    with db.connect() as conn:
        for i, (name, label) in enumerate(EXTRA_DEVICES[:n], start=4):
            uid, vid = db.demo_user_id(label), db.demo_vehicle_id(label)
            conn.execute("INSERT INTO users (id, name, phone) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                         (uid, name, f"010-0000-{i:04d}"))
            conn.execute("INSERT INTO vehicles (id, owner_user_id, label) VALUES (%s, %s, %s) "
                         "ON CONFLICT (id) DO NOTHING", (vid, uid, label))


def run_devices(jobs: list[tuple[str, list[str]]], timeout: float) -> tuple[str, dict[str, int]]:
    """(기기, 명령) 들을 동시에 돌려 기기 머리를 붙인 출력과 기기별 종료 코드를 돌려준다."""
    from edge.blackbox.variants import _NOISE

    lock = threading.Lock()
    lines: list[str] = []
    procs = []

    def pump(p: subprocess.Popen, dev: str) -> None:
        for line in p.stdout:
            if not any(n in line for n in _NOISE):
                with lock:
                    lines.append(prefix_line(line, dev))

    for dev, cmd in jobs:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        t = threading.Thread(target=pump, args=(p, dev), daemon=True)
        t.start()
        procs.append((dev, p, t))
    end = time.time() + timeout
    codes = {}
    for dev, p, _ in procs:
        try:
            codes[dev] = p.wait(timeout=max(1.0, end - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
            codes[dev] = -1
    for _, _, t in procs:
        t.join(timeout=2)
    return "".join(lines), codes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=",".join(DEFAULT_VIDEOS), help="영상 이름 (쉼표로). 영상마다 3대")
    ap.add_argument("--data-root", default=str(ROOT),
                    help="data/videos · data/replay 가 들어 있는 폴더 (기본: 이 레포)")
    ap.add_argument("--db", default=DEFAULT_DB,
                    help=f"쓸 DB 이름 (시작할 때 비운다). 기본 {DEFAULT_DB}. 시연 DB '{DEMO_DB}' 면 확인을 받는다")
    ap.add_argument("--yes", action="store_true", help="시연 DB 를 비울 때 묻지 않는다")
    ap.add_argument("--port", type=int, default=8010, help="서버 포트 (시연 서버 8000 과 다르게)")
    ap.add_argument("--ac6-min", type=float, default=0.5, help="AC-6 공통 결함 중 하나로 묶인 비율의 통과 기준 (제안)")
    ap.add_argument("--timeout", type=float, default=900, help="블랙박스가 모두 끝나기를 기다릴 최대 초")
    ap.add_argument("--out", default=None, help="로그·사진을 둘 폴더. 없으면 server_data/e2e/<시각>/")
    a = ap.parse_args(argv)

    data = Path(a.data_root).expanduser()
    videos = [v for v in a.videos.split(",") if v]
    if not videos or len(videos) > 1 + len(EXTRA_DEVICES) // PER_VIDEO:
        _say(f"오류: 영상은 1~{1 + len(EXTRA_DEVICES) // PER_VIDEO}개")
        return 2
    files = {v: {"video": data / "data" / "videos" / f"{v}.mp4", "gps": data / "data" / "videos" / f"{v}.gps.csv",
                 "record": data / "data" / "replay" / f"{v}{RECORD_SUFFIX}",
                 "labels": ROOT / "experiments" / "gt_events" / f"{v}.csv"} for v in videos}
    missing = [f"{v} {what}: {p}" for v, fs in files.items() for what, p in fs.items() if not p.exists()]
    if missing:
        _say("필요한 파일이 없다:\n  " + "\n  ".join(missing))
        for v, fs in files.items():
            if not fs["record"].exists() and fs["video"].exists():
                _say(f"{v} 탐지 기록 만들기 (시연 설정 v3 + 도로 조각):\n  "
                     + RECORD_CMD.format(video=fs["video"], record=fs["record"]))
        return 2
    heads = {}
    for v, fs in files.items():
        with fs["record"].open(encoding="utf-8") as f:
            heads[v] = json.loads(f.readline())

    url, demo_name = _db_url(a.db)
    if a.db in {DEMO_DB, demo_name} and not confirm_demo_db(a.db, a.yes):
        return 2
    code = ensure_docker_db(url)
    if code:
        return code
    if _port_busy(a.port):
        _say(f"오류: 포트 {a.port} 을(를) 이미 쓰고 있다. --port 로 다른 포트를 준다")
        return 2

    out = Path(a.out) if a.out else ROOT / "server_data" / "e2e" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    # 이 프로세스의 server.* 도 e2e DB 를 보게 한다 (server.config 는 처음 불릴 때 읽는다)
    os.environ["DATABASE_URL"] = url
    import importlib

    from server import config

    importlib.reload(config)
    from server import db

    importlib.reload(db)
    db.init_db()
    db.reset_db()
    add_extra_devices(db, PER_VIDEO * (len(videos) - 1))
    _say(f"DB {a.db}: 스키마 적용·비움 (시민·차량 {PER_VIDEO * len(videos)}대)")

    env = {**os.environ, "DATABASE_URL": url, "SNAPSHOT_DIR": str(out / "snapshots"), "CLIP_DIR": str(out / "clips"),
           # 결함 주소(네이버 Reverse Geocoding)를 받지 않는다: 바깥으로 요청을 보내지 않고, 키가 없어도 같은 결과
           "NAVER_MAP_KEY_ID": "", "NAVER_MAP_CLIENT_SECRET": "", "PYTHONUNBUFFERED": "1"}
    server_log = (out / "server.log").open("w")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.main:app", "--port", str(a.port)],
                              cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)
    base = f"http://localhost:{a.port}"
    ids = device_ids(len(videos))
    video_of = {d: v for v, devs in zip(videos, ids) for d in devs}
    try:
        import httpx

        from edge.blackbox.variants import VARIANTS, run_args

        def healthy() -> bool:
            if server.poll() is not None:
                raise RuntimeError(f"서버가 바로 꺼졌다. {out / 'server.log'} 를 본다")
            try:
                return httpx.get(f"{base}/health", timeout=2).status_code == 200
            except httpx.HTTPError:
                return False

        _wait(healthy, 60, "서버")
        _say(f"서버 준비됨 {base} (DB {a.db})")

        jobs = []
        for v, devs in zip(videos, ids):
            fs = files[v]
            for dev, variant in zip(devs, VARIANTS):
                jobs.append((dev, [
                    sys.executable, "-u", "-m", "edge.blackbox.run", "--video", str(fs["video"]), "--gps", str(fs["gps"]),
                    *run_args({**variant, "device_id": dev}), "--detector", "replay", "--record", str(fs["record"]),
                    "--detect-every", str(heads[v]["detect_every"]), "--speed", "0", "--server", base,
                    # 시연 화면(server_data/live)·블랙박스 보관 폴더를 건드리지 않는다
                    "--live-fps", "0", "--keep-dir", ""]))
        _say(f"가상 블랙박스 {len(jobs)}대 동시 (영상 {len(videos)}개 × {PER_VIDEO}대, replay, 배속 0) ...")
        t0 = time.time()
        bb_out, codes = run_devices(jobs, a.timeout)
        (out / "blackbox.log").write_text(bb_out, encoding="utf-8")
        _say(f"블랙박스 끝 ({time.time() - t0:.0f}초, 종료 코드 {codes}). 로그 {out / 'blackbox.log'}")

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
    expected = [d for devs in ids for d in devs]
    facility, facility_missing = facility_tracks(devices, [devs[0] for devs in ids])
    labels = {v: load_labels(files[v]["labels"].read_text(encoding="utf-8")) for v in videos}
    obs_by_video = {v: [o for o in obs_rows if video_of.get(o[0]) == v] for v in videos}
    ac6 = {v: ac6_summary(obs_by_video[v], labels[v]) for v in videos}
    cross = overlap = None
    if len(videos) >= 2:
        cross = cross_video_defects(obs_by_video, labels)
        tracks = {v: _read_track(files[v]["gps"]) for v in videos}
        gaps = {(x, y): min_track_distance_m(tracks[x], tracks[y]) for i, x in enumerate(videos) for y in videos[i + 1:]}
        overlap = any(g <= DEDUP_M for g in gaps.values())
    rows = judge(devices, expected, db_obs, db_by_type, facility, ac6, a.ac6_min, cross,
                 overlap if overlap is not None else True, facility_missing)
    bad = {d: c for d, c in codes.items() if c != 0}
    if bad:
        rows.append(("-", "블랙박스 종료 코드 0 아님", ", ".join(f"{d} {c}" for d, c in bad.items()), "없음", False))

    _say("")
    _say(f"## e2e smoke — 영상 {len(videos)}개 × {PER_VIDEO}대, DB {a.db}")
    for v, devs in zip(videos, ids):
        _say(f"- {v}: {devs[0]}~{devs[-1]}, replay {files[v]['record'].name} ({heads[v].get('model', '?')}, "
             f"탐지 주기 {heads[v]['detect_every']})")
    _say("")
    _say(format_table(rows))
    _say("")
    _say("참고 (판정 아님)")
    _say("  기기별 보낸/저장 관측: " + " · ".join(
        f"{d} {devices.get(d, {}).get('sent')}/{per_vehicle.get(d, 0)}" for d in expected))
    _say("  유형별 관측 (DB): " + ", ".join(f"{KO.get(t, t)} {n}" for t, n in sorted(obs_types.items())))
    _say("  유형별 결함 (DB): " + ", ".join(f"{KO.get(t, t)} {n}" for t, n in sorted(db_by_type.items())))
    _say("  보내지 않은 탐지 트랙 (기기별): " + " · ".join(
        f"{d} " + ("-" if devices.get(d, {}).get("unsent") is None else
                   ", ".join(f"{KO.get(t, t)} {n}" for t, n in sorted(devices[d]["unsent"].items())) or "없음")
        for d in expected))
    n_def = multi["n"]
    if n_def and db_obs:
        _say(f"  결함 {n_def}건 중 2대 이상이 본 결함 {multi['m']}건 ({multi['m'] / n_def * 100:.0f}%), "
             f"중복 정리 비율 (1 − 결함 ÷ 관측) {(1 - n_def / db_obs) * 100:.1f}%")
    for v, s in ac6.items():
        _say(f"  {v}: 라벨에 안 맞는 관측 {s['unmatched_obs']}건 (헛것 후보), amb 라벨 관측 {s['amb_obs']}건"
             + (", 나뉜 공통 결함 " + ", ".join(f"{k}→{n}개" for k, n in sorted(s["split"].items()))
                if s["split"] else ""))
    if cross is not None:
        _say("  두 영상 GPS 경로의 가장 가까운 거리: " + ", ".join(f"{x}↔{y} {g:.0f} m" for (x, y), g in gaps.items())
             + f" (묶기 반경 {DEDUP_M:g} m {'안: 경로가 겹친다' if overlap else '밖'})")
        for d, vs in cross.items():
            _say(f"  다른 영상끼리 묶인 결함 {d[:8]}: " + " / ".join(f"{v} {','.join(ls)}" for v, ls in vs.items()))
    _say(f"  로그·사진: {out}")
    ok = all(r[4] for r in rows)
    _say("")
    _say("결과: 통과" if ok else "결과: 실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
