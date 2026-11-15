"""결함 위치의 주소 (네이버 클라우드 Maps Reverse Geocoding).

결함이 처음 생기면 한 번 주소를 받아 defects.address 에 저장한다(server/sql/002_address.sql).
시연장에 인터넷이 없어도 저장해 둔 주소가 화면·신고서에 보인다.
받지 못하면(키 없음, 권한 없음, 인터넷 없음) 주소 없이 진행하고 RETRY_S 초 뒤에 다시 시도한다.

필요한 값(.env, 저장소에 올리지 않는다):
  NAVER_MAP_KEY_ID         지도와 같은 Client ID
  NAVER_MAP_CLIENT_SECRET  Client Secret. 콘솔의 Application 에서 Reverse Geocoding 을 켜야 한다
  NAVER_GEOCODE_URL        (선택) 호출 주소. 없으면 URLS 를 차례로 시도하고 되는 쪽을 기억한다
"""

from __future__ import annotations

import logging
import threading

import httpx
import psycopg

from server import config, db

log = logging.getLogger(__name__)

URLS = [
    "https://maps.apigw.ntruss.com/map-reversegeocode/v2/gc",
    "https://naveropenapi.apigw.ntruss.com/map-reversegeocode/v2/gc",
]
RETRY_S = 60        # 실패한 결함을 다시 시도하기까지 기다리는 시간
BATCH = 10          # 한 번에 채우는 결함 수
TIMEOUT_S = 3.0

_lock = threading.Lock()
_good_url: str | None = None


def keys() -> tuple[str, str] | None:
    kid = config.get("NAVER_MAP_KEY_ID", "").strip()
    secret = config.get("NAVER_MAP_CLIENT_SECRET", "").strip()
    return (kid, secret) if kid and secret else None


def format_address(body: dict) -> str | None:
    """Reverse Geocoding 응답을 "서울 강남구 강남대로 396" 꼴로. 도로명 주소가 없으면 지번 주소."""
    results = {r.get("name"): r for r in body.get("results", [])}
    for kind in ("roadaddr", "addr"):
        r = results.get(kind)
        if not r:
            continue
        region, land = r.get("region", {}), r.get("land") or {}
        a1 = region.get("area1", {})
        parts = [a1.get("alias") or a1.get("name"), region.get("area2", {}).get("name")]
        num = "-".join(n for n in (land.get("number1"), land.get("number2")) if n)
        if kind == "roadaddr":
            parts += [land.get("name"), num]
        else:
            parts += [region.get("area3", {}).get("name"), region.get("area4", {}).get("name"),
                      ("산 " if land.get("type") == "2" else "") + num]
        text = " ".join(p.strip() for p in parts if p and p.strip())
        if text:
            return text
    return None


def reverse(lat: float, lon: float) -> str | None:
    """위치의 주소. 받지 못하면 None (예외를 내지 않는다)."""
    global _good_url
    k = keys()
    if k is None:
        return None
    headers = {"x-ncp-apigw-api-key-id": k[0], "x-ncp-apigw-api-key": k[1]}
    params = {"coords": f"{lon},{lat}", "orders": "roadaddr,addr", "output": "json"}
    fixed = config.get("NAVER_GEOCODE_URL", "").strip()
    urls = [fixed] if fixed else ([_good_url] if _good_url else []) + [u for u in URLS if u != _good_url]
    for url in urls:
        try:
            r = httpx.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
        except httpx.HTTPError as e:
            log.info("주소 받기 실패(%s): %s", url, e)
            return None  # 인터넷이 없으면 다른 주소도 안 된다
        if r.status_code != 200:
            log.info("주소 받기 실패(%s): %s %s", url, r.status_code, r.text[:200])
            continue
        _good_url = url
        body = r.json()
        if body.get("status", {}).get("code") not in (0, 3):  # 3 = 결과 없음(바다 등)
            log.info("주소 받기 실패(%s): %s", url, body.get("status"))
            return None
        return format_address(body)
    return None


def fill_one(conn: psycopg.Connection, defect_id) -> str | None:
    """결함 하나의 주소를 지금 받아 저장한다(이미 있으면 그대로). 신고서를 만들 때 쓴다."""
    d = conn.execute(
        "SELECT address, ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lon FROM defects WHERE id = %s",
        (defect_id,),
    ).fetchone()
    if d is None or d["address"]:
        return d and d["address"]
    addr = reverse(d["lat"], d["lon"])
    conn.execute("UPDATE defects SET address = %s, address_tried_at = now() WHERE id = %s", (addr, defect_id))
    return addr


def fill_missing() -> int:
    """주소가 없는 결함(최근 RETRY_S 초 안에 시도한 것은 빼고)의 주소를 받아 저장한다. 채운 수.

    새 관측이 들어올 때와 관리자 지도가 목록을 부를 때 응답 뒤에 돈다. 이미 돌고 있으면 바로 끝낸다.
    """
    if keys() is None or not _lock.acquire(blocking=False):
        return 0
    try:
        with db.connect() as conn:
            rows = conn.execute(
                """SELECT id, ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lon FROM defects
                   WHERE address IS NULL
                     AND (address_tried_at IS NULL OR address_tried_at < now() - make_interval(secs => %s))
                   ORDER BY first_seen_at LIMIT %s""",
                (RETRY_S, BATCH),
            ).fetchall()
            n = 0
            for d in rows:
                addr = reverse(d["lat"], d["lon"])
                conn.execute("UPDATE defects SET address = %s, address_tried_at = now() WHERE id = %s",
                             (addr, d["id"]))
                conn.commit()
                n += addr is not None
            return n
    except psycopg.Error as e:
        log.warning("주소 저장 실패: %s", e)
        return 0
    finally:
        _lock.release()
