# 다보여

블랙박스 탑재형 AI 도로 결함 탐지 및 간편 신고 시스템 — 2026 세종대 컴퓨터공학과 학술제 MVP.
요구사항은 [PRD.md](PRD.md), 실행 계획은 [docs/plan.md](docs/plan.md), 협업 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md), 탐지 모델 사용법은 [docs/ai-handoff.md](docs/ai-handoff.md).

## 시작하기

필요: [uv](https://docs.astral.sh/uv/), Docker

```bash
uv sync                    # Python 3.11 가상환경 + 워크스페이스 설치
uv run pytest              # 계약·스키마 테스트
docker compose up -d db    # PostgreSQL + PostGIS (스키마는 server/sql 에서 자동 생성)
```

스키마를 바꾼 뒤 DB를 새로 만들려면 `docker compose down -v && docker compose up -d db`.

## 구조

| 폴더 | 담당 | 내용 |
|---|---|---|
| `contracts/` | 공동 | **인터페이스 계약** — 클래스 순서, `Detection`/`Detector`, `ObservationIn`, GPS CSV(`GpsTrack`). 바꿀 때는 두 사람 모두 확인 |
| `ai/` | 박겸 | 데이터셋 준비, YOLO 학습(Colab), ONNX 변환·추론, 차선 마모, 평가·속도 측정 |
| `edge/` | 정현우 | 가상 블랙박스 (프레임 추출·품질 판별·분석 주기·전송) |
| `server/` | 정현우 | FastAPI + PostGIS. `server/sql/001_init.sql` 이 스키마 |
| `ui/` | 정현우 | Streamlit 시민 화면·관리자 지도 |
| `tools/video`, `tools/gps` | 정현우 | 공개 주행 영상 목록, GPS 경로 재구성 |
| `scripts/` | 공동 | 다중 차량 실행, end-to-end 확인, 시연 초기화 |
| `experiments/` | 공동 | 실험 결과 CSV·그래프, `gt_events/` 수동 라벨 |
| `data/` | — | 영상·GPS·클립 (git 제외) |

새 구성요소(예: `server/`)에 Python 패키지를 만들면 루트 `pyproject.toml` 의 `[tool.uv.workspace] members` 에 추가한다.

## 계약 요약

```python
from daboyeo_contracts import CLASS_NAMES, Detection, Detector, GpsTrack, ObservationIn

CLASS_NAMES                      # ("pothole", "crack", "guardrail", "sign") — YOLO class_id 순서, 바꾸면 모델 재학습
Detection.from_class_id(0, conf=0.9, bbox=(x1, y1, x2, y2))   # bbox 는 0~1 정규화
GpsTrack.from_csv("v1.gps.csv").at(12.5)                       # CSV: t_s,lat,lon[,speed_kmh]
```

- 탐지기 구현은 `detect(frame_bgr) -> list[Detection]` 하나만 제공하고, 전처리·NMS는 안에서 끝낸다.
- DB enum 과 `DefectType`/`RiskLevel` 은 `tests/test_schema_matches_contracts.py` 가 맞는지 확인한다.

## 영상 데이터

공개 주행 영상은 라이선스 때문에 레포에 올리지 않는다 (`.gitignore` 에서 `*.mp4` 제외). 출처는 `tools/video/sources.csv` 에 기록하고 발표 자료에 표기한다.
