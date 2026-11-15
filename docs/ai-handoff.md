# AI → 백엔드 인계: 탐지 모델 사용법

> 박겸 → 정현우. 계획서 [docs/plan.md](plan.md) 4장 "인터페이스 계약", 작업 1.6(11/18 인터페이스 맞춤)용.
> 이 문서는 **AI 쪽에서 넘기는 것과 쓰는 법**만 다룬다. 가상 블랙박스·서버 구현은 정현우 담당이라 여기서 정하지 않는다.

## 1. 한눈에

```python
from daboyeo_ai.inference import OnnxDetector

det = OnnxDetector("ai/models/detector_v2.onnx", threads=1)
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
| `detector_v2.onnx` | 박겸 Google Drive `daboyeo_data/models/` → 받아서 **`ai/models/`** 에 둔다 | git 에 안 올림 (`.gitignore`) |
| `detector_v2.json` | 레포 `ai/models/` (Drive 에도 있음) | 클래스 순서, 입력 크기, 성능, 학습 데이터 정보 |

| 버전 | 클래스 | 상태 |
|---|---|---|
| **v2 (사용)** | 4클래스 전부 — RDD2022 + AI Hub 도로장애물(179) + 전국 도로시설물(KRID) | 11/15 완료 |
| v1 | 출력은 4클래스지만 **학습된 건 pothole·crack 만** (RDD2022) | v2 로 대체 |

**v1 → v2 는 파일만 바꾸면 된다.** 출력 클래스 수·순서·입력 형식이 같아서 코드 변경이 없다.

v2 test mAP50 (자세한 값은 `detector_v2.json`):

| | pothole | crack | guardrail | sign |
|---|---|---|---|---|
| 전체 test | 0.468 | 0.272 | 0.553 | 0.658 |
| RDD2022 test 만 | 0.418 | **0.584** | — | — |
| 도로장애물(179) test 만 | 0.573 | 0.001 | — | — |

- 크랙 성능은 **RDD2022 test 기준(0.584)** 으로 본다. 도로장애물(179) 크랙 라벨은 한 이미지에 짧은 선분 박스가 수백 개라, 모델이 크랙 전체를 큰 박스로 잡아도 평가에서 맞은 것으로 치지 않는다. 전체 test 의 0.272 는 그 영향이다.
- 같은 RDD2022 test 에서 v1 은 pothole 0.409 / crack 0.570 이라 v2 가 떨어진 클래스는 없다.

## 4. `OnnxDetector` 옵션

| 인자 | 기본 | 설명 |
|---|---|---|
| `model_path` | — | `.onnx` 경로 |
| `threads` | `1` | ONNX Runtime CPU 스레드 수. **PRD FR-1.6 의 "CPU 1~2코어 제한"을 여기서 건다** (1 또는 2) |
| `conf` | `0.25` | 이 값 미만 탐지는 버린다. 블랙박스가 기록할 최소 신뢰도. 숫자 하나 또는 클래스별 dict (아래 4-1) |
| `iou` | `0.45` | 같은 클래스 박스끼리 겹침 제거(NMS) 기준 |

- `conf` 는 **블랙박스 기록 기준**이다. 시민에게 보여줄지 정하는 **노출 기준 θ(오픈 이슈 #3)는 별개**로, 서버(F7)에서 따로 정한다.
- 입력 크기는 모델 파일에서 자동으로 읽는다 (v1: 640).
- 클래스 수가 계약(4개)과 다른 모델을 넣으면 생성할 때 바로 `ValueError` 가 난다.
- `det.raw(frame)` 은 `[(class_id, conf, x1, y1, x2, y2), ...]` 를 **픽셀 좌표**로 준다 (오버레이 그릴 때 편함).

### 4-1. 클래스별 신뢰도 기준 (제안 — 정현우 확인 필요, 오픈 이슈 #3)

`conf` 에 클래스별 dict 를 넘길 수 있다. 빠진 클래스는 0.25. 숫자 하나를 넘기는 예전 방식은 그대로 동작한다.

```python
from daboyeo_ai.inference import OnnxDetector, load_class_conf

det = OnnxDetector("ai/models/detector_v2.onnx", threads=1,
                   conf={"pothole": 0.22, "crack": 0.25, "guardrail": 0.26, "sign": 0.24})
# 또는 파일에서: experiments/conf_thresholds.json 의 "class_conf", 모델 카드의 "class_conf_draft"
det = OnnxDetector("ai/models/detector_v2.onnx", threads=1, conf=load_class_conf("experiments/conf_thresholds.json"))
det.class_conf   # [0.22, 0.25, 0.26, 0.24] — class_id 순서
det.conf         # 0.22 — 그중 최솟값 (예전 코드 호환)
```

초안 값 (2026-11-15, `uv run python -m daboyeo_ai.eval.conf_thresholds --model ... --parts-dir data/aihub/parts`).
test 사진에서 IoU 0.5 로 맞춘 **탐지 1건 단위** 정밀도(P)·재현율(R):

| 클래스 | 정밀도 ≥ 0.7 (`class_conf`) | 0.25 일 때 | 정밀도 ≥ 0.8 | 정밀도 ≥ 0.9 | 데이터 |
|---|---|---|---|---|---|
| pothole | **0.22** (P 0.70 / R 0.50) | P 0.72 / R 0.47 | 0.33 (0.82 / 0.40) | 0.48 (0.90 / 0.31) | 도로장애물(179) test 392장 (RDD2022 미포함) |
| crack | **0.25** (데이터 없음, 그대로) | — | — | — | RDD2022 test 필요 (179 크랙 라벨은 매칭 불가) |
| guardrail | **0.26** (0.70 / 0.57) | 0.69 / 0.58 | 0.42 (0.81 / 0.50) | 0.62 (0.90 / 0.37) | KRID test 1,165장 |
| sign | **0.24** (0.70 / 0.62) | 0.71 / 0.62 | 0.37 (0.81 / 0.56) | 0.56 (0.90 / 0.45) | KRID test 1,165장 |

- 정밀도 0.7 을 목표로 하면 네 클래스 모두 **지금의 0.25 와 거의 같다** — test 에서 0.25 가 이미 정밀도 약 0.7 이다.
  시민 화면에 헛것을 덜 보이려면 0.8 열(0.33 / 0.42 / 0.37)이 후보. 재현율을 6~8%p 잃는다.
- **어디에 쓰나 (정할 것)**: (가) 블랙박스 `conf` — 기준 미만은 아예 서버로 안 간다. (나) 서버 노출 기준 θ
  (`exposure.py`: `conf ≥ θ OR vehicle_count ≥ k`) — 블랙박스는 0.25 로 기록하고, 서버가 클래스별 θ 로 '한 대만 봐도
  보여줄지'를 정한다. θ 는 차량 한 대의 탐지만으로 보여주는 기준이라 (나)라면 0.8~0.9 열이 맞다. **k(최소 차량 수)는
  서버 값이라 여기서 정하지 않는다.**
- 한계: 학습 데이터와 같은 출처의 사진이다. 시연 영상에서는 정밀도가 달라지므로 #3(11/24)에서 시연 영상으로 다시 확인한다.
  crack·pothole 은 RDD2022 test(`data/aihub/parts/rdd2022.zip`)를 받으면 같은 명령으로 다시 낸다.

## 5. 계약 패키지 (`daboyeo_contracts`)

두 사람이 같이 쓰는 형식. **바꿀 때는 서로 확인한다** ([contracts/](../contracts/daboyeo_contracts/__init__.py)).

| 이름 | 내용 |
|---|---|
| `CLASS_NAMES` | `("pothole", "crack", "guardrail", "sign")` — 모델 class_id 순서. 바꾸면 재학습 필요 |
| `Detection` | 탐지 1건 (위 1장). `damage`·`damage_conf` 는 시설물 파손 분류용 — **파손 모델을 켰을 때만** guardrail·sign 에 채워진다 (9장). 안 켜면 항상 `None` |
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
- 학습 노트북이 모델을 만들 때마다 val 10장으로 같은 비교를 다시 하고 결과를 `detector_vN.json` 의 `onnx_parity` 에 남긴다 (v2: 66/66 일치).
- CPU 속도는 `detector_vN.json` 의 `colab_cpu_speed` 에 참고값이 있다 (Colab CPU 기준). **정식 측정은 노트북 PC 에서** (계획 4.2).

## 7. 주의

- **OpenCV**: `ai` 패키지는 `opencv-python-headless` 를 쓴다 (서버·Colab 용, 창 띄우기 기능 없음). 블랙박스에서 `cv2.imshow` 로 오버레이 창을 띄우려면 GUI 가 되는 `opencv-python` 이 필요한데, **두 패키지를 같이 설치하면 충돌**한다. 화면 표시 방식을 정할 때 같이 맞추자.
- 세션 생성(`OnnxDetector(...)`)은 무거우니 한 번 만들어 재사용한다. 차량(가상 블랙박스)마다 하나씩 만드는 건 괜찮다.
- `detect()` 가 빈 리스트를 주는 프레임이 대부분이다.

## 8. 차선 마모 (제안 — 정현우 확인 필요)

> 계획 2.6, PRD FR-1.3. 학습 모델이 아니라 OpenCV 영상 처리다. **입력·출력 형식은 아래가 제안이고, 11/18 인터페이스 맞춤 때 같이 확정한다.**
> **11/15 변경: 판정은 `LaneWearTracker` (상대 판정 + 연속 프레임 투표) 를 쓰고, 보내는 건 `j.confirmed` 일 때뿐이다.** `r.is_worn`(절대 기준 0.3)은 호환용으로 남겼지만 손 라벨에서 거의 맞지 않는다 (아래 표). 기준값(1.0)·투표(5 중 3)는 잠정 (TBD) — 시연 주행 영상으로 오픈 이슈 #6(11/22)에서 다시 확인한다.

```python
from daboyeo_ai.lane_wear import LaneWearMeter

meter = LaneWearMeter()               # 한 번 만들어 재사용 (threshold=0.3 잠정, roi=하단 육각형)
r = meter.measure(frame_bgr)          # OpenCV BGR 프레임 그대로, 1280×720 에서 약 4 ms (CPU)
r.lane_found        # 차선 선분을 찾았나
r.contrast          # (차선 밝기 − 바로 옆 노면 밝기) / 노면 밝기, 항상 > 0. 못 찾으면 None
r.is_worn           # lane_found 이고 contrast < threshold
r.confidence        # 0~1, 판정 신뢰도 → ObservationIn.confidence
r.lane_pixel_ratio  # 진단용: ROI 중 차선 화소 비율
r.n_segments        # 진단용: 차선으로 인정된 선분 수
r.contrast_p25      # 선분을 따라 행마다 잰 대비의 하위 25% — 페인트가 군데군데 벗겨지면 낮다 (못 찾으면 None)
r.paint_len         # 남은 차선 선분 길이 합 ÷ 화면 폭 — 지워져 토막 난 차선은 짧다 (못 찾으면 0)
```

**상대 판정 + 연속 프레임 투표 (권장).** 절대 대비는 마모보다 카메라·노면색·조명에 더 좌우된다 (KRID 선명 중앙값 0.47 < AI Hub 도로장애물 *마모* 중앙값 0.72). 그래서 같은 주행의 보통 수준과 비교하고, 한 프레임 판정은 절반쯤 오탐이라 **연속 프레임 5개 중 3개**가 마모일 때만 "마모 관측" 1건으로 확정한다 (11/15 r3.8).

```python
from daboyeo_ai.lane_wear import LaneWearMeter, LaneWearTracker

meter = LaneWearMeter()
tracker = LaneWearTracker()           # 주행(영상·차량) 하나에 하나. threshold=1.0 잠정, window=100, warmup=20,
                                      # confirm_k=3, confirm_n=5, cooldown=None, max_gap_s=5.0
j = tracker.update(meter.measure(frame_bgr), t=frame_time_s)   # t(초)는 선택 — 주면 끊김을 안다
j.ready       # 차선 프레임이 warmup(20)개 모였나. 그 전에는 판정 안 함 (is_worn=False)
j.baseline    # 이 주행의 보통 contrast_p25 (최근 차선 프레임 100개의 중앙값)
j.score       # (contrast_p25 ÷ baseline) × paint_len. 낮을수록 마모
j.is_worn     # 이 프레임 하나의 판정: ready 이고 score < 1.0 — 이것만으로 보내지 않는다
j.votes       # 투표 창(최근 판정 프레임 5개) 안의 마모 프레임 수
j.n_voted     # 투표 창에 든 판정 프레임 수 (≤ 5)
j.confirmed   # ★ 이 프레임에서 마모 관측 1건 확정 → 이때만 ObservationIn 1건
j.confidence  # 0~1 → ObservationIn.confidence
```

투표 규칙 (`LaneWearTracker`, 블랙박스 쪽 = 2단계):
- 투표 창 = 최근 **판정한** 프레임(차선 찾음 + warmup 뒤) 5개의 `is_worn`. `lane_found=False` 프레임은 창에 넣지도, 창을 비우지도 않는다 (앞차 가림·교차로 한두 프레임은 건너뛴다).
- 창 안 마모가 3개 이상이 되는 프레임에서 `confirmed=True`. 그 뒤로는 **창 안 마모가 0개가 될 때까지**(선명 판정 5개 연속) 다시 확정하지 않는다 → 마모 구간 하나에 1건. `cooldown=N` 을 주면 확정 뒤 판정 프레임 N개가 지나면 같은 구간에서도 다시 1건 (긴 마모 구간을 지점마다 보내고 싶을 때).
- 구간 끊김: `t` 를 주면 차선 찾은 프레임 사이가 5초(`max_gap_s`)를 넘을 때 투표 창을 비운다 (기준은 유지). `t` 를 안 주면 끊김을 모르므로, 영상이 바뀌거나 오래 멈췄으면 `tracker.reset()` (기준·투표 모두 비움).
- 블랙박스가 보내는 것: **`j.confirmed` 인 프레임마다** `ObservationIn(defect_type=DefectType.LANE_WEAR, bbox=None, confidence=j.confidence, ...)` 1건 (위치·시각은 그 프레임 것). `r.is_worn`·`j.is_worn` 으로는 보내지 않는다.
- 투표 효과는 손 라벨(서로 독립인 프레임)로 잴 수 없다 — **시연 주행 영상으로 확인한다** (#6, 11/22).

손 라벨 143장 비교 (선명 112 / 마모 31, 낮을수록 마모, `uv run python -m daboyeo_ai.eval.lane_wear_eval <labels.json> <frames>`):

| 점수 | AUC | best F1 | 2-fold F1 |
|---|---|---|---|
| contrast (절대, 지금 `r.is_worn`) | 0.56 | 0.37 | 0.32 |
| contrast ÷ 주행 중앙값 | 0.67 | 0.44 | 0.40 |
| paint_len | 0.74 | 0.51 | 0.49 |
| **score (상대 p25 × paint_len)** | **0.79** | **0.58** (기준 1.7) | **0.58** |
| score, tracker 를 순서대로 돌림 (warmup 뒤 111장) | 0.78 | 0.55 | 0.50 |

한 프레임 기준값별 (같은 도구 출력, 차선 찾은 218장 = `제외` 포함 / 선명 112장):

| 기준 (score <) | 마모로 본 비율 (218장) | 선명→마모 | 정밀도 | 재현율 |
|---|---|---|---|---|
| 0.6 | 19% | 4.5% | 0.55 | 0.19 |
| **1.0 (잠정, 11/15~)** | **28%** | **12%** | 0.41 | 0.29 |
| 1.7 (11/15 이전) | 41% | 18% | 0.51 | 0.68 |

1.7 에서는 너무 많이 마모로 불렀다. 정밀도는 기준값과 상관없이 0.5 안팎이라 **엄격함은 반복 관측에서 얻는다**: 1단계 기준 1.0 → 2단계 블랙박스 5 중 3 투표 → 3단계 서버 여러 차량 확정 (아래, 제안).

**3단계 — 서버 확정 규칙 (제안 — 정현우 확인 필요).** 한 지점(묶기 반경 안, #2)의 차선 마모는 **서로 다른 차량 2대 이상**에서 온 **마모 관측 3건 이상**일 때만 확정한다 (오픈 이슈 #6 "3회 이상 관측"과 같음). 한 차량이 같은 곳을 여러 번 지나며 낸 관측만으로는 확정하지 않는다 — 그 차 카메라·주행의 기준이 치우쳤을 수 있다. 서버 코드·스키마는 정현우 담당이라 AI 쪽에서는 고치지 않았다.

| 항목 | 제안 |
|---|---|
| 언제 부르나 | **품질 판별을 통과한 프레임**에서만, 탐지보다 긴 주기로 (PRD 8.3 — 주기 값은 `scheduler.py` 에서 정함). 주기가 길면 연속 5 프레임이 차지하는 거리도 길어진다 |
| tracker 를 어디에 두나 | **미정 — 정현우와 정할 것.** (가) 블랙박스가 주행(영상)마다 `LaneWearTracker` 하나를 두고 `j.confirmed` 일 때만 보낸다 — 간단, 서버 변경 없음, 투표까지 블랙박스에서 끝남. (나) 블랙박스는 `contrast_p25`·`paint_len` 을 보내고 서버가 차량별로 기준·투표를 한다 — 기준값을 서버에서 바꿀 수 있지만 `ObservationIn` 필드 추가(계약 변경)가 필요하다 |
| 서버 전송 | (가) 기준: `j.confirmed` 일 때 `ObservationIn(defect_type=DefectType.LANE_WEAR, bbox=None, confidence=j.confidence, ...)` 1건 |
| 확정 | 한 프레임으로 확정하지 않는다. 블랙박스 5 중 3 투표로 관측 1건 → 서버에서 차량 2대 이상 · 관측 3건 이상으로 지점 확정 (3단계 제안) |
| 못 찾음 | `lane_found=False` 면 `is_worn=False`. **완전히 지워진 차선과 차선 없는 도로는 구별 못 하므로** 마모로 보내지 않는다 |

주의:
- 야간·비·역광·그림자에서는 밝기 대비가 크게 흔들린다. 낮·맑은 품질 통과 프레임만 쓰는 걸 권장.
- ROI 기본값(11/15 변경)은 화면 높이 0.35~0.92 의 육각형 — 지평선이 화면 0.2~0.5 사이인 블랙박스(KRID·AI Hub 도로장애물 실제 프레임)에 맞췄고, 아래 8% 는 보닛·대시보드로 뺀다. 영상 화각이 다르면 `LaneWearMeter(roi=...)` 로 바꾼다 (정규화 좌표 다각형).
- 11/15 실제 프레임 300장 기준 차선을 찾는 비율은 약 73% (교차로·차선 없는 도로·앞차에 가려진 프레임은 `lane_found=False`). 1280×720 과 1920×1080 은 폭 640 으로 줄여서 재므로 결과가 같다.
- 상대 판정의 한계: 주행 전체가 마모 도로면 기준 자체가 낮아져 놓친다. 터널·야간 진입처럼 조명이 크게 바뀌면 `tracker.reset()` 하거나 새로 만든다. `paint_len` 은 보이는 차로 수·점선 여부에 따라 달라진다 (편도 1차로·점선 도로는 짧게 나와 마모 쪽으로 기운다).
- 위 숫자의 한계: 라벨은 썸네일로 붙여 잡음이 있고, 카메라가 두 종류뿐이며, KRID 101장은 여러 날·지역 프레임을 "한 주행"으로 묶었다. 기준 1.0 과 5 중 3 투표는 이 라벨(독립 프레임)로 고른·정한 값이라 시연 영상에서 다시 확인해야 한다.
- 보정 도구: `uv run python -m daboyeo_ai.eval.lane_wear_calib <프레임 폴더|영상> --overlay-dir out/` — 프레임별 contrast CSV, 백분위수, ROI·선분 오버레이.

## 9. 시설물 파손 분류 — **사용 안 함 (11/15 대체안 확정)**

> **이번 버전에서는 파손 판정을 하지 않는다.** 학습 결과(test F1 0.43, 정상 오탐률 15.7%)가 기준에 못 미쳐 PRD 대체안으로 확정했다.
> `damage_model` 을 넘기지 않으면 되고, 모델 파일도 배포하지 않는다. `Detection.damage` 는 항상 `None`, `ObservationIn.is_suspected` 는 항상 `false`.
> 가드레일·표지판은 **'시설물 탐지됨'(위치)** 으로만 다룬다. 아래는 나중에 다시 켤 때를 위한 기록이다.

> 계획 2.7, PRD FR-1.2. `OnnxDetector` 에 **선택 인자 하나**가 늘었다. 안 쓰면 지금과 완전히 같다.
> 블랙박스에서 켤지, 서버·화면에서 `damage` 를 어떻게 쓸지는 11/18 인터페이스 맞춤 때 같이 정한다.

```python
from daboyeo_ai.inference import OnnxDetector

det = OnnxDetector("ai/models/detector_v2.onnx", threads=1,
                   damage_model="ai/models/damage_cls_v1.onnx")   # 옆에 damage_cls_v1.json 이 꼭 있어야 한다
for d in det.detect(frame_bgr):
    if d.class_name in ("guardrail", "sign"):
        d.damage       # "suspected"(파손 의심) | "normal" | None(박스가 너무 작아 분류 안 함)
        d.damage_conf  # P(damaged) 0~1 — "normal" 이어도 채워진다 (판정 신뢰도가 아니라 파손 확률)
```

| 항목 | 내용 |
|---|---|
| 대상 | guardrail·sign 탐지만. pothole·crack 은 항상 `damage=None` (계약이 막음) |
| `"suspected"` 뜻 | `damage_conf >= threshold` (임계값은 `damage_cls_v1.json` 의 `threshold`, val 최대 F1). **확정이 아니라 '파손 의심'** — 화면에도 그렇게만 표시 (FR-1.2) |
| 신고 제안 | 한 대·한 프레임의 `suspected` 로 하지 않는다. **여러 차량이 같은 판단일 때만** (FR-7.2). 몇 대·몇 % 로 할지는 서버 쪽 규칙 (제안: 같은 결함 군집의 관측 중 suspected 비율) |
| 너무 작은 박스 | 짧은 변 24 px 미만이면 분류하지 않고 `None` (학습 크롭에도 없던 크기) |
| CPU | 탐지 → 분류가 같은 스레드 수(`threads`)로 **차례로** 돈다. 코어를 더 쓰지 않지만 시설물 박스 1개마다 시간이 늘어난다 (크롭 1장당 ms 는 `damage_cls_v1.json` 의 `colab_cpu_speed`) |
| 파일 | `damage_cls_v1.onnx` + `damage_cls_v1.json` 둘 다 `ai/models/` 에. json 에 클래스 순서·임계값·크롭 규칙이 있어서 **json 없이는 열리지 않는다** |

- 서버로 보낼 때 `ObservationIn` 에 `damage` 를 넣을 필드는 아직 없다. 넣으려면 `contracts/` 변경이라 둘이 같이 정한다 (제안: `damage: Literal["normal","suspected"] | None = None` 기본값 추가).
- 직접 부를 수도 있다: `DamageClassifier("ai/models/damage_cls_v1.onnx").prob_damaged(frame_bgr, (x1, y1, x2, y2))` (픽셀 좌표).

## 10. 아직 AI 쪽에서 안 넘긴 것

| 항목 | 예정 | 비고 |
|---|---|---|
| v1 모델 파일 | 완료 | v2 로 대체 |
| 시설물 파손 분류 모델 파일 | **넘기지 않음** | 11/15 대체안 확정 (9장) |
| 차선 마모 기준값·확정 규칙 | 11/22 (#6) | 측정 코드는 8장 (형식은 제안 상태). 기준값은 실제 프레임으로 보정 |
| v2 모델 (4클래스) | **완료 11/15** | Drive `models/detector_v2.onnx` (3장) |

## 11. AI 쪽이 받아야 하는 것 (계획 4장)

- **속도 측정 로그 형식**: 프레임·단계별 `{"frame", "t", "stage", "ran", "ms"}` JSONL — `bench_cpu`·`interval_sweep` 실험이 이 로그를 읽는다. 형식이 바뀌면 알려주기.
  주기 적용 전/후 로그를 하나씩 주면 단계별 실행 비율·ms·FPS·영상 1분당 계산 시간을 낸다: `uv run python -m daboyeo_ai.eval.bench_cpu log 전=before.jsonl 후=after.jsonl --out experiments/bench_cpu_log.csv`. 건너뛴 프레임도 `"ran": false` 로 한 줄씩 남겨 주기 (영상 길이·실행 비율 계산에 쓴다). `t` 가 없으면 `--video-fps 30` 을 준다.
- **프레임 샘플링 방식**: 분석 주기(N프레임마다 탐지 등)를 어떻게 적용하는지.
- **시연 영상 + 수동 라벨** (`experiments/gt_events/`): 실제 주행 영상 평가에 쓴다.
