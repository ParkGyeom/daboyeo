"""AI Hub 다운로드 클라이언트.

공식 aihubshell(v0.6)과 같은 엔드포인트를 쓴다:
- 파일 트리 (공개):   GET https://api.aihub.or.kr/info/{datasetkey}.do
- 다운로드 (승인 필요): GET https://api.aihub.or.kr/down/0.6/{datasetkey}.do?fileSn={filekey}
                       헤더 `apikey: <AI Hub API 키>` → tar 안에 분할 압축(*.zip.partN)

aihubshell 대신 직접 구현한 이유: 병합할 때 tar·part·zip 이 동시에 디스크에 남지 않게 하고,
macOS/Colab 차이(sort -V) 없이 동작하게 하려고.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://api.aihub.or.kr"
API_VERSION = "0.6"

_SIZE_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


@dataclass(frozen=True)
class AihubFile:
    path: str  # 트리 안의 전체 경로 (폴더/.../파일명)
    size: str  # 예: "11 GB"
    key: int  # fileSn

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def size_bytes(self) -> int:
        num, unit = self.size.split()
        return int(float(num) * _SIZE_UNITS[unit.upper()])


def parse_tree(text: str) -> list[AihubFile]:
    """`/info/{key}.do` 응답(트리 문자열)을 파일 목록으로 바꾼다."""
    files: list[AihubFile] = []
    stack: list[tuple[int, str]] = []  # (마커 열 위치, 폴더 이름)
    for line in text.splitlines():
        m = re.search(r"[├└]─", line)
        if not m:
            continue
        col = m.start()
        entry = line[m.end():].strip()
        while stack and stack[-1][0] >= col:
            stack.pop()
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) == 3 and parts[2].isdigit():
            path = "/".join([*(n for _, n in stack), parts[0]])
            files.append(AihubFile(path=path, size=parts[1], key=int(parts[2])))
        else:
            stack.append((col, entry))
    return files


def fetch_tree(datasetkey: int) -> list[AihubFile]:
    with urllib.request.urlopen(f"{BASE_URL}/info/{datasetkey}.do", timeout=60) as r:
        return parse_tree(r.read().decode("utf-8"))


def _natural_key(p: Path) -> int:
    return int(re.search(r"\.part(\d+)$", p.name).group(1))


def merge_parts(root: Path) -> list[Path]:
    """root 아래 `x.zip.part0, x.zip.part1, …` 을 `x.zip` 으로 합치고 part 파일을 지운다."""
    groups: dict[Path, list[Path]] = {}
    for p in root.rglob("*.part*"):
        if re.search(r"\.part\d+$", p.name):
            groups.setdefault(p.with_name(re.sub(r"\.part\d+$", "", p.name)), []).append(p)
    merged = []
    for target, parts in groups.items():
        parts.sort(key=_natural_key)
        with open(target, "wb") as out:
            for part in parts:
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out, 16 * 1024 * 1024)
                part.unlink()  # 합치는 즉시 지워서 디스크 사용량을 줄인다
        merged.append(target)
    return merged


def download(datasetkey: int, file: AihubFile, apikey: str, dest: Path) -> Path:
    """파일 1개를 받아 병합까지 끝낸 zip 경로를 돌려준다. 이미 있으면 다시 받지 않는다."""
    dest.mkdir(parents=True, exist_ok=True)
    existing = list(dest.rglob(file.name))
    if existing:
        return existing[0]

    tar_path = dest / f"download_{file.key}.tar"
    url = f"{BASE_URL}/down/{API_VERSION}/{datasetkey}.do?fileSn={file.key}"
    # curl: 이어받기(-C -) 지원, 진행률 표시. API 키는 헤더로만 보낸다.
    proc = subprocess.run(
        ["curl", "-L", "-C", "-", "-o", str(tar_path), "-H", f"apikey:{apikey}",
         "-w", "%{http_code}", "-sS", url],
        capture_output=True, text=True,
    )
    code = proc.stdout.strip()[-3:]
    if proc.returncode != 0 or code not in ("200", "206"):
        body = tar_path.read_text(errors="replace")[:500] if tar_path.exists() else ""
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(f"AI Hub download failed ({file.name}): curl={proc.returncode} http={code} {body}")

    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")
    tar_path.unlink()
    merge_parts(dest)

    found = list(dest.rglob(file.name))
    if not found:
        raise RuntimeError(f"{file.name} not found after extracting {tar_path.name}")
    return found[0]
