"""통신량 측정 (B8, docs/DESIGN.md 8장).

구간의 원본 프레임 용량 합과 가상 블랙박스가 실제로 보내는 용량(탐지 결과 JSON + 결함 주변 사진)을 비교한다.
서버로는 보내지 않고(--dry-run) 보낼 용량만 센다. 서버의 GET /api/stats 로 받은 용량과 같은 방식으로 센다.
원본 PNG 는 무손실이라 커서 비율이 과장된다. 그래서 "프레임 전체를 JPEG(품질 80)로 보냈다면"의 용량도 함께 비교한다.

  .venv/bin/python experiments/bandwidth.py --model runs/<모델>.onnx
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edge.blackbox.frames import list_frames  # noqa: E402
from edge.blackbox.run import drive, parse_args  # noqa: E402
from ui.basemap import SEGMENTS  # noqa: E402  시연 구간 후보 5곳


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    print("| 구간 | 프레임 | 원본 PNG | 전체 JPEG | 보낸 건수 | 보낸 용량 | PNG 대비 | JPEG 대비 |")
    print("|---|---|---|---|---|---|---|---|")
    tot_raw = tot_sent = tot_jpg = 0
    for seg in SEGMENTS:
        frames = list_frames(ROOT / "data/roadmark/VS4", seg)
        raw = sum(f.image_path.stat().st_size + (f.gps_path.stat().st_size if f.gps_path.exists() else 0) for f in frames)
        jpg = sum(len(cv2.imencode(".jpg", cv2.imread(str(f.image_path)), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]) for f in frames)
        with contextlib.redirect_stdout(io.StringIO()):
            r = drive(parse_args(["--segment", seg, "--model", a.model, "--speed", "0", "--dry-run", "--live-fps", "0"]))
        tot_raw += raw
        tot_jpg += jpg
        tot_sent += r["bytes_sent"]
        print(f"| {seg} | {len(frames)} | {raw / 1e6:.1f}MB | {jpg / 1e6:.1f}MB | {r['sent']} | {r['bytes_sent'] / 1e3:.1f}KB "
              f"| {r['bytes_sent'] / raw * 100:.3f}% | {r['bytes_sent'] / jpg * 100:.2f}% |", flush=True)
    print(f"| 합계 | | {tot_raw / 1e6:.1f}MB | {tot_jpg / 1e6:.1f}MB | | {tot_sent / 1e3:.1f}KB "
          f"| {tot_sent / tot_raw * 100:.3f}% | {tot_sent / tot_jpg * 100:.2f}% |")


if __name__ == "__main__":
    main()
