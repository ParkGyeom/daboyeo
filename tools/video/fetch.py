"""sources.csv 의 공개 주행 영상을 id 로 골라 받는다 (docs/plan.md 0.6).

    uv run python tools/video/fetch.py gn_a gn_b            # 받기 + 1080p/30fps 사본 + 시연 클립
    uv run python tools/video/fetch.py gn_a --max-height 1080 --no-normalize
    uv run python tools/video/fetch.py gn_a gn_b --clips-only   # 이미 받은 원본에서 클립만 다시

만드는 파일 (기본 --out data/videos):
    <id>.mp4                 원본 (영상만, 소리 없음). 화질은 --max-height 이하 최고
    <id>_1080p30.mp4         높이 1080 이하·30fps 로 맞춘 사본 (H.264)
    <id>_MMSS-MMSS.mp4       sources.csv demo_segment 구간. 원본 해상도, 재인코딩 없이 자름
    fetch_log.jsonl          파일마다 길이·fps·해상도·크기 (한 줄에 하나)

재인코딩 없이 자르면 클립은 시작 시각 직전의 키프레임에서 시작한다. 실제 시작 시각은
로그의 clip_start_in_source_s 에 적는다 (클립 t=0 ↔ 원본 이 시각). --exact-clips 를 주면
원본 해상도로 다시 인코딩해 정확한 시각에서 자른다.

영상은 YouTube 표준 라이선스라 저장소·공개 위치에 올리지 않는다 (.gitignore: /data/, *.mp4).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).resolve().with_name("sources.csv")


@dataclass(frozen=True)
class Source:
    id: str
    url: str
    demo_segment: str

    def segment(self) -> tuple[float, float] | None:
        """'강남대로 … 약 20:50~25:10' → (1250.0, 1510.0). 없으면 None."""
        m = re.search(r"(\d+):(\d{2})\s*~\s*(\d+):(\d{2})", self.demo_segment)
        if not m:
            return None
        a, b, c, d = map(int, m.groups())
        return a * 60 + b, c * 60 + d


def load_sources(path: Path = SOURCES) -> dict[str, Source]:
    with path.open(encoding="utf-8") as f:
        return {r["id"]: Source(r["id"], r["url"], r.get("demo_segment", "")) for r in csv.DictReader(f)}


def mmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 60:02d}{t % 60:02d}"


def clip_name(vid: str, start: float, end: float) -> str:
    return f"{vid}_{mmss(start)}-{mmss(end)}.mp4"


# ---------------------------------------------------------------- ffprobe / 로그

def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration,size",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    j = json.loads(out)
    s, fmt = j["streams"][0], j["format"]
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    return {
        "codec": s["codec_name"],
        "width": s["width"],
        "height": s["height"],
        "fps": round(num / den, 3) if den else None,
        "duration_s": round(float(fmt["duration"]), 3),
        "size_bytes": int(fmt["size"]),
        "size_mb": round(int(fmt["size"]) / 1e6, 1),
    }


def keyframe_at_or_before(path: Path, t: float) -> float:
    """t 이하에서 가장 늦은 키프레임 시각 (스트림 복사로 자를 때 실제 시작점)."""
    # 패킷의 K 플래그로 찾는다 (frame 출력에 -skip_frame nokey 는 걸러지지 않는다)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-read_intervals", f"{max(t - 30, 0)}%{t + 0.001}",
         "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    keys = []
    for line in out.splitlines():
        pts, _, flags = line.partition(",")
        if "K" in flags and pts not in ("", "N/A") and float(pts) <= t + 1e-3:
            keys.append(float(pts))
    return max(keys) if keys else 0.0


def log(out_dir: Path, vid: str, kind: str, path: Path, **extra) -> dict:
    rec = {"id": vid, "kind": kind, "file": path.name, **probe(path), **extra,
           "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    # 같은 파일의 예전 기록은 지우고 새 기록으로 바꾼다 (다시 실행해도 파일당 한 줄)
    log_path = out_dir / "fetch_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    lines = [ln for ln in lines if ln.strip() and json.loads(ln).get("file") != path.name]
    lines.append(json.dumps(rec, ensure_ascii=False))
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{vid}] {kind:10s} {path.name}: {rec['width']}x{rec['height']} {rec['fps']}fps "
          f"{rec['duration_s']}s {rec['size_mb']}MB {rec['codec']}"
          + "".join(f" {k}={v}" for k, v in extra.items()), flush=True)
    return rec


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y", *args], check=True)


# ---------------------------------------------------------------- 단계

def download(src: Source, out_dir: Path, max_height: int) -> Path:
    import yt_dlp  # uv 의존성 (pyproject.toml)

    dst = out_dir / f"{src.id}.mp4"
    if dst.exists():
        print(f"[{src.id}] 이미 있음: {dst}")
        return dst
    opts = {
        # 영상만 (블랙박스는 소리를 안 쓴다). VP9 우선: AV1 은 OpenCV 빌드에 따라 못 읽을 수 있다
        "format": f"bv*[height<={max_height}]",
        "format_sort": [f"res:{max_height}", "fps", "vcodec:vp9"],
        "outtmpl": str(out_dir / f"{src.id}.%(ext)s"),
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "noplaylist": True,
        "continuedl": True,
        # 큰 파일을 한 요청으로 받으면 중간에 HTTP 403 이 난다 → 10MB 씩 나눠 받고 재시도
        "http_chunk_size": 10 * 1024 * 1024,
        "retries": 10,
        "fragment_retries": 10,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([src.url])
    if not dst.exists():
        raise SystemExit(f"[{src.id}] 다운로드 결과 {dst} 가 없다")
    return dst


def normalize(src: Source, orig: Path, out_dir: Path) -> Path:
    dst = out_dir / f"{src.id}_1080p30.mp4"
    if not dst.exists():
        tmp = dst.with_suffix(".part.mp4")
        ffmpeg("-i", str(orig), "-an",
               "-vf", "scale=-2:'min(1080,ih)':flags=lanczos,fps=30",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(tmp))
        tmp.rename(dst)
    return dst


def cut_clip(src: Source, orig: Path, out_dir: Path, exact: bool) -> tuple[Path, dict] | None:
    seg = src.segment()
    if seg is None:
        print(f"[{src.id}] demo_segment 없음 — 클립 건너뜀")
        return None
    start, end = seg
    dst = out_dir / clip_name(src.id, start, end)
    if exact:
        ffmpeg("-ss", str(start), "-i", str(orig), "-t", str(end - start), "-an",
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(dst))
        actual = float(start)
    else:
        actual = keyframe_at_or_before(orig, start)
        # 실제 시작 키프레임부터 원래 끝 시각까지 복사
        ffmpeg("-ss", f"{actual:.3f}", "-i", str(orig), "-t", f"{end - actual:.3f}", "-an",
               "-c", "copy", "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(dst))
    extra = {
        "segment_s": [start, end],
        "clip_start_in_source_s": round(actual, 3),
        "start_offset_s": round(actual - start, 3),
        "reencoded": exact,
    }
    return dst, extra


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+", help="sources.csv 의 id (예: gn_a gn_b)")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "videos")
    ap.add_argument("--max-height", type=int, default=2160, help="원본으로 받을 최대 높이 (기본 2160)")
    ap.add_argument("--no-normalize", action="store_true", help="1080p/30fps 사본을 만들지 않음")
    ap.add_argument("--no-clip", action="store_true", help="demo_segment 클립을 만들지 않음")
    ap.add_argument("--clips-only", action="store_true", help="받지 않고 이미 있는 원본에서 클립만")
    ap.add_argument("--exact-clips", action="store_true", help="클립을 다시 인코딩해 정확한 시각에서 자름")
    args = ap.parse_args(argv)

    sources = load_sources()
    unknown = [i for i in args.ids if i not in sources]
    if unknown:
        ap.error(f"sources.csv 에 없는 id: {unknown} (있는 id: {list(sources)})")
    args.out.mkdir(parents=True, exist_ok=True)

    for vid in args.ids:
        src = sources[vid]
        orig = args.out / f"{vid}.mp4"
        if args.clips_only:
            if not orig.exists():
                raise SystemExit(f"[{vid}] 원본 {orig} 가 없다")
        else:
            orig = download(src, args.out, args.max_height)
            log(args.out, vid, "original", orig, url=src.url)
            if not args.no_normalize:
                log(args.out, vid, "normalized", normalize(src, orig, args.out))
        if not args.no_clip:
            got = cut_clip(src, orig, args.out, args.exact_clips)
            if got:
                log(args.out, vid, "clip", got[0], **got[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
