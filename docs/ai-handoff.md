# AI → 백엔드 인계: 탐지 모델 사용법

> 박겸 → 정현우. 계획서 [docs/plan.md](plan.md) 4장 "인터페이스 계약", 작업 1.6(11/18 인터페이스 맞춤)용.
> 이 문서는 **AI 쪽에서 넘기는 것과 쓰는 법**만 다룬다. 가상 블랙박스·서버 구현은 정현우 담당이라 여기서 정하지 않는다.

## 1. 한눈에

```python
from daboyeo_ai.inference import OnnxDetector

det = OnnxDetector("ai/models/detector_v1.onnx", threads=1)
detections = det.detect(frame_bgr)        # OpenCV 로 읽은 BGR 프레임 그대로
for d in detections:
    d.class_name   # "pothole" | "crack" | "guardrail" | "sign"
    d.conf         # 0~1
    d.bbox         # (x1, y1, x2, y2) — 프레임 크기로 나눈 0~1 값
    d.defect_type  # DefectType.POTHOLE 등 — 서버 전송용 결함 유형
```

전처리(리사이즈·패딩·정규화)와 NMS 는 `OnnxDetector` 안에서 끝난다. 프레임을 가공해서 넘길 필요 없다.

## 2. 설치

```bash
git clone https://github.com/ParkGyeom/daboyeo.git && cd daboyeo
uv sync          # contracts + ai 패키지, onnxruntime·opencv 포함 (Python 3.11)
```

uv 를 안 쓰면: `pip install -e contracts -e ai`

## 3. 모델 파일 받기

| 파일 | 위치 | 비고 |
|---|---|---|
| `detector_v1.onnx` | 박겸 Google Drive `daboyeo_data/models/` → 받아서 **`ai/models/`** 에 둔다 | git 에 안 올림 (`.gitignore`) |
| `detector_v1.json` | 같은 곳 | 클래스 순서, 입력 크기, 성능, 학습 데이터 정보 |

| 버전 | 클래스 | 상태 |
|---|---|---|
| **v1** | 출력은 4클래스지만 **학습된 건 pothole·crack 만** (공개 데이터 RDD2022) | 11/13 학습 중, 11/18 전 전달 |
| v2 | 4클래스 전부 (AI Hub 국내 도로·시설물 데이터 추가) | AI Hub 서버 복구 후 |

**v1 → v2 는 파일만 바꾸면 된다.** 출력 클래스 수·순서·입력 형식이 같아서 코드 변경이 없다.
v1 에서는 guardrail·sign 이 거의 나오지 않는 게 정상이다.

## 4. `OnnxDetector` 옵션

| 인자 | 기본 | 설명 |
|---|---|---|
| `model_path` | — | `.onnx` 경로 |
| `threads` | `1` | ONNX Runtime CPU 스레드 수. **PRD FR-1.6 의 "CPU 1~2코어 제한"을 여기서 건다** (1 또는 2) |
| `conf` | `0.25` | 이 값 미만 탐지는 버린다. 블랙박스가 기록할 최소 신뢰도 |
| `iou` | `0.45` | 같은 클래스 박스끼리 겹침 제거(NMS) 기준 |

- `conf` 는 **블랙박스 기록 기준**이다. 시민에게 보여줄지 정하는 **노출 기준 θ(오픈 이슈 #3)는 별개**로, 서버(F7)에서 따로 정한다.
- 입력 크기는 모델 파일에서 자동으로 읽는다 (v1: 640).
- 클래스 수가 계약(4개)과 다른 모델을 넣으면 생성할 때 바로 `ValueError` 가 난다.
- `det.raw(frame)` 은 `[(class_id, conf, x1, y1, x2, y2), ...]` 를 **픽셀 좌표**로 준다 (오버레이 그릴 때 편함).

## 5. 계약 패키지 (`daboyeo_contracts`)

두 사람이 같이 쓰는 형식. **바꿀 때는 서로 확인한다** ([contracts/](../contracts/daboyeo_contracts/__init__.py)).

| 이름 | 내용 |
|---|---|
| `CLASS_NAMES` | `("pothole", "crack", "guardrail", "sign")` — 모델 class_id 순서. 바꾸면 재학습 필요 |
| `Detection` | 탐지 1건 (위 1장). `damage`·`damage_conf` 는 시설물 파손 분류용으로 **v1 에서는 항상 `None`** |
| `Detector` | `detect(frame_bgr) -> list[Detection]` 프로토콜. 가짜·재생(replay) 탐지기를 만들 때 이 모양만 맞추면 `OnnxDetector` 와 바꿔 끼울 수 있다 |
| `DefectType`, `DETECTOR_TO_DEFECT` | 탐지 클래스 → 서버 결함 유형 (`guardrail` → `guardrail_damage`, `sign` → `sign_damage`) |
| `RISK_BY_TYPE` | 유형별 위험도 (PRD F4 표) |
| `ObservationIn` | `POST /api/v1/observations` JSON 메타 형식 (PRD 9장 observations) |
| `GpsTrack` | GPS CSV(`t_s,lat,lon[,speed_kmh]`) 읽기 + `at(t_s)` 보간 (FR-4.4) |

```python
from daboyeo_contracts import GpsTrack
track = GpsTrack.from_csv("data/gps/video1.gps.csv")
lat, lon = track.at(12.5)              # 영상 12.5초 지점 좌표
lat, lon = track.at(999, clamp=True)   # 범위 밖이면 끝점 (clamp 없으면 ValueError)
```

DB enum(`server/sql/001_init.sql`)과 `DefectType`·`RiskLevel` 이 어긋나면 `tests/test_schema_matches_contracts.py` 가 실패한다.

## 6. 확인된 것

- `OnnxDetector` 의 전처리·후처리는 Ultralytics 가 같은 ONNX 를 돌린 결과와 **박스·신뢰도가 완전히 같다** (사전학습 YOLO11n, 5/5, IoU 1.000).
- 학습 노트북이 v1 모델을 만들 때 val 10장으로 같은 비교를 다시 하고 결과를 `detector_v1.json` 의 `onnx_parity` 에 남긴다.
- CPU 속도는 `detector_v1.json` 의 `colab_cpu_speed` 에 참고값이 있다 (Colab CPU 기준). **정식 측정은 노트북 PC 에서** (계획 4.2).

## 7. 주의

- **OpenCV**: `ai` 패키지는 `opencv-python-headless` 를 쓴다 (서버·Colab 용, 창 띄우기 기능 없음). 블랙박스에서 `cv2.imshow` 로 오버레이 창을 띄우려면 GUI 가 되는 `opencv-python` 이 필요한데, **두 패키지를 같이 설치하면 충돌**한다. 화면 표시 방식을 정할 때 같이 맞추자.
- 세션 생성(`OnnxDetector(...)`)은 무거우니 한 번 만들어 재사용한다. 차량(가상 블랙박스)마다 하나씩 만드는 건 괜찮다.
- `detect()` 가 빈 리스트를 주는 프레임이 대부분이다.

## 8. 아직 AI 쪽에서 안 넘긴 것

| 항목 | 예정 | 비고 |
|---|---|---|
| v1 모델 파일 | 11/13~14 | 학습 끝나면 Drive 공유 |
| 시설물 파손 분류 (`Detection.damage` 채우기) | 11/19~20 (P1) | KRID 파손 라벨 사용. 11/20 유지 여부 결정 |
| 차선 마모 판별 | 11/19~21 | 입력·출력 형식은 만들기 전에 같이 정하기 |
| v2 모델 (4클래스) | AI Hub 데이터 확보 후 | 파일만 교체 |

## 9. AI 쪽이 받아야 하는 것 (계획 4장)

- **속도 측정 로그 형식**: 프레임·단계별 `{"frame", "t", "stage", "ran", "ms"}` JSONL — `bench_cpu`·`interval_sweep` 실험이 이 로그를 읽는다. 형식이 바뀌면 알려주기.
- **프레임 샘플링 방식**: 분석 주기(N프레임마다 탐지 등)를 어떻게 적용하는지.
- **시연 영상 + 수동 라벨** (`experiments/gt_events/`): 실제 주행 영상 평가에 쓴다.
