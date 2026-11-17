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


DATABASE_URL = get("DATABASE_URL", "postgresql://daboyeo:daboyeo@localhost:5432/daboyeo")
SNAPSHOT_DIR = ROOT / get("SNAPSHOT_DIR", "server_data/snapshots")
MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024
# 모의 문자에 넣을 링크의 주소(화면 주소). scripts/demo.sh 가 화면 포트에 맞춰 준다
PUBLIC_BASE_URL = get("PUBLIC_BASE_URL", "http://localhost:8501").rstrip("/")
LINK_TTL_HOURS = int(get("LINK_TTL_HOURS", "72"))
CLIP_DIR = ROOT / get("CLIP_DIR", "server_data/clips")
MAX_CLIP_BYTES = 20 * 1024 * 1024  # 결함 구간 클립 (전후 2초, 480p H.264 면 보통 1MB 아래)
