"""설정값. 환경 변수가 우선이고, 프로젝트 폴더의 .env 가 있으면 그 값을 기본으로 쓴다."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> dict[str, str]:
    env: dict[str, str] = {}
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


_DOTENV = _load_dotenv()


def get(name: str, default: str) -> str:
    return os.environ.get(name, _DOTENV.get(name, default))


# server/sql/001_init.sql 이 만드는 daboyeo DB 와 테이블 이름(users, defects 등)이 겹쳐서,
# 이 서버는 contracts 형식으로 옮기기 전까지 같은 DB 서버의 daboyeo_legacy DB 를 따로 쓴다 (없으면 만든다)
DATABASE_URL = get("DATABASE_URL", "postgresql://daboyeo:daboyeo@localhost:5432/daboyeo_legacy")
SNAPSHOT_DIR = ROOT / get("SNAPSHOT_DIR", "server_data/snapshots")
MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024
