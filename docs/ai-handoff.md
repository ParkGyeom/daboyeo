# AI → 백엔드 인계: 탐지 모델 사용법

> 박겸 → 정현우. 계획서 [docs/plan.md](plan.md) 4장 "인터페이스 계약", 작업 1.6(11/18 인터페이스 맞춤)용.
> 이 문서는 **AI 쪽에서 넘기는 것과 쓰는 법**만 다룬다. 가상 블랙박스·서버 구현은 정현우 담당이라 여기서 정하지 않는다.

## 1. 한눈에

```python
from daboyeo_ai.inference import OnnxDetector, load_class_conf

det = OnnxDetector("ai/models/detector_v3.onnx", threads=1,
                  conf=load_class_conf("ai/models/detector_v3.json"))   # 시연 모델 v3 (11/16 확정, 16장)
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
| `detector_v3.onnx` | 박겸 Google Drive `daboyeo_data/models/` → 받아서 **`ai/models/`** 에 둔다 | git 에 안 올림 (`.gitignore`) |
| `detector_v3.json` | 레포 `ai/models/` | 클래스 순서, 입력 크기, 성능, **클래스별 기준 `class_conf_draft`** (crack 0.42) |

| 버전 | 클래스 | 상태 |
|---|---|---|
| **v3 (사용, 11/16 확정)** | v2 에서 이어 학습 — 179 크랙 데이터를 고친 `road_v3` (15장) | 시연 클립에서 v2 가 못 잡는 균열을 잡음 (16장) |
| v2 | 4클래스 전부 — RDD2022 + AI Hub 도로장애물(179) + 전국 도로시설물(KRID) | v3 로 대체 |
| v1 | 출력은 4클래스지만 **학습된 건 pothole·crack 만** (RDD2022) | v2 로 대체 |

**v2 → v3 도 파일(과 기준)만 바꾸면 된다.** 출력 클래스 수·순서·입력 형식이 같아서 코드 변경이 없다.

v2 test mAP50 (자세한 값은 `detector_v2.json`):

| | pothole | crack | guardrail | sign |
|---|---|---|---|---|
| 전체 test | 0.468 | 0.272 | 0.553 | 0.658 |
| RDD2022 test 만 | 0.418 | **0.584** | — | — |
| 도로장애물(179) test 만 | 0.573 | 0.001 | — | — |

- (11/15) 179 크랙 0.001 의 진짜 원인은 179 크랙 이미지 zip 이 라벨과 다른 사진이었던 것 — 15장.
- 크랙 성능은 **RDD2022 test 기준(0.584)** 으로 본다. 도로장애물(179) 크랙 라벨은 한 이미지에 짧은 선분 박스가 수백 개라, 모델이 크랙 전체를 큰 박스로 잡아도 평가에서 맞은 것으로 치지 않는다. 전체 test 의 0.272 는 그 영향이다.
- 같은 RDD2022 test 에서 v1 은 pothole 0.409 / crack 0.570 이라 v2 가 떨어진 클래스는 없다.

## 4. `OnnxDetector` 옵션

| 인자 | 기본 | 설명 |
|---|---|---|
| `model_path` | — | `.onnx` 경로 |
| `threads` | `1` | ONNX Runtime CPU 스레드 수. **PRD FR-1.6 의 "CPU 1~2코어 제한"을 여기서 건다** (1 또는 2) |
| `conf` | `0.25` | 이 값 미만 탐지는 버린다. 블랙박스가 기록할 최소 신뢰도. 숫자 하나 또는 클래스별 dict (아래 4-1) |
| `iou` | `0.45` | 같은 클래스 박스끼리 겹침 제거(NMS) 기준 |
| `windows` | `None` | (선택) 프레임 일부를 잘라 한 번 더 본다. `road_tiles(cols=2)` = 전체 + 도로 좌우 2조각 → 모델 3번, 시간 약 3배. 14장 |

- `conf` 는 **블랙박스 기록 기준**이다. 시민에게 보여줄지 정하는 **노출 기준 θ(오픈 이슈 #3)는 별개**로, 서버(F7)에서 따로 정한다.
- 입력 크기는 모델 파일에서 자동으로 읽는다 (v1: 640).
- 클래스 수가 계약(4개)과 다른 모델을 넣으면 생성할 때 바로 `ValueError` 가 난다.
- `det.raw(frame)` 은 `[(class_id, conf, x1, y1, x2, y2), ...]` 를 **픽셀 좌표**로 준다 (오버레이 그릴 때 편함).

### 4-1. 클래스별 신뢰도 기준 (제안 — 정현우 확인 필요, 오픈 이슈 #3)

`conf` 에 클래스별 dict 를 넘길 수 있다. 빠진 클래스는 0.25. 숫자 하나를 넘기는 예전 방식은 그대로 동작한다.

```python
from daboyeo_ai.inference import OnnxDetector, load_class_conf

det = OnnxDetector("ai/models/detector_v2.onnx", threads=1,
                   conf={"pothole": 0.27, "crack": 0.29, "guardrail": 0.26, "sign": 0.24})
# 또는 파일에서: experiments/conf_thresholds.json 의 "class_conf", 모델 카드의 "class_conf_draft"
det = OnnxDetector("ai/models/detector_v2.onnx", threads=1, conf=load_class_conf("experiments/conf_thresholds.json"))
# 시연 모델 v3 (16장): 카드의 class_conf_draft = {"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}
det = OnnxDetector("ai/models/detector_v3.onnx", threads=1, conf=load_class_conf("ai/models/detector_v3.json"))
det.class_conf   # [0.27, 0.29, 0.26, 0.24] — class_id 순서
det.conf         # 0.24 — 그중 최솟값 (예전 코드 호환)
```

초안 값 (2026-11-15, `uv run python -m daboyeo_ai.eval.conf_thresholds --model ... --parts-dir data/aihub/parts`).
test 사진에서 IoU 0.5 로 맞춘 **탐지 1건 단위** 정밀도(P)·재현율(R):

| 클래스 | 정밀도 ≥ 0.7 (`class_conf`) | 0.25 일 때 | 정밀도 ≥ 0.8 | 정밀도 ≥ 0.9 | 데이터 |
|---|---|---|---|---|---|
| pothole | **0.27** (P 0.70 / R 0.36) | P 0.68 / R 0.38 | 0.39 (0.81 / 0.27) | 0.51 (0.90 / 0.19) | RDD2022 test 3,526장 + 도로장애물(179) test 392장 |
| crack | **0.29** (0.71 / 0.48) | 0.67 / 0.52 | 0.40 (0.80 / 0.39) | 0.55 (0.90 / 0.25) | RDD2022 test 3,526장 (179 크랙 라벨은 매칭 불가라 뺌) |
| guardrail | **0.26** (0.70 / 0.57) | 0.69 / 0.58 | 0.42 (0.81 / 0.50) | 0.62 (0.90 / 0.37) | KRID test 1,165장 |
| sign | **0.24** (0.70 / 0.62) | 0.71 / 0.62 | 0.37 (0.81 / 0.56) | 0.56 (0.90 / 0.45) | KRID test 1,165장 |

- 정밀도 0.7 을 목표로 하면 네 클래스 모두 **지금의 0.25 와 거의 같다** — test 에서 0.25 가 이미 정밀도 약 0.7 이다.
  시민 화면에 헛것을 덜 보이려면 0.8 열(0.39 / 0.40 / 0.42 / 0.37)이 후보. 재현율을 7~13%p 잃는다.
- **어디에 쓰나 (정할 것)**: (가) 블랙박스 `conf` — 기준 미만은 아예 서버로 안 간다. (나) 서버 노출 기준 θ
  (`exposure.py`: `conf ≥ θ OR vehicle_count ≥ k`) — 블랙박스는 0.25 로 기록하고, 서버가 클래스별 θ 로 '한 대만 봐도
  보여줄지'를 정한다. θ 는 차량 한 대의 탐지만으로 보여주는 기준이라 (나)라면 0.8~0.9 열이 맞다. **k(최소 차량 수)는
  서버 값이라 여기서 정하지 않는다.**
- 한계: 학습 데이터와 같은 출처의 사진이다. 시연 영상에서는 정밀도가 달라지므로 #3(11/24)에서 시연 영상으로 다시 확인한다.
  (11/15 RDD2022 test 를 넣어 다시 냄 — crack·pothole 값 갱신.)

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

### 8-1. 실측 재귀반사도(LTL-M)와 비교 (11/15, 결과: 기준·규칙 그대로)

AI Hub 「도로 로드마크 인식을 위한 주행 영상 데이터」(654) 검증 묶음 VS4.비온전(복합) + VL4 라벨. 프레임마다 차량에 단 이동식 측정기 값 하나(LTL-M, 데이터 페이지 표기 "로드마크 반사도 수치 (단위: mcd)" — 단위는 이것뿐이고 m²·lx 는 적혀 있지 않다. 값 범위로 보아 보통의 mcd/(m²·lx) 로 읽었다)가 있다. 이미지·LTL-M 이 모두 있는 프레임은 **1,996장**(낮 1,327 / 밤 669; LTL-M 2,792개 중 796개는 이미지가 없다). 약 1초 간격의 짧은 토막이라 warmup·5 중 3 투표는 못 재고 **한 프레임 판정**만 쟀다. 결과 집계 `experiments/lane_wear_roadmark.json`.

```bash
uv run python -m daboyeo_ai.eval.lane_wear_roadmark --ltlm <VS4>/LTL-M --gps <VS4>/GPS --labels <VL4>/2D_JSON \
    --images "<…>/VS4.비온전(복합).zip" --all-images --cache <레포 밖>/cache.json --out experiments/lane_wear_roadmark.json
```

정답 정하기:
- **기준값 근거**: 노면표시 최소 재귀반사성능 — 설치 시 백색 240 · 황색 150, **유지 최소 백색 100 · 황색 70** mcd/(m²·lx) ([도로교통법 시행규칙 별표 6 관련 문서](https://www.law.go.kr/flDownload.do?flSeq=79139155)). 지방도 현장 연구도 이 범위를 재도색 기준으로 쓴다 ([KoreaScience 2017](https://koreascience.kr/article/JAKO201734542314896.pdf)). 측정한 선의 색을 모르므로 **마모 = LTL-M < 100** (백색 기준), 70 은 민감도로 따로 봤다.
- **LTL-M < 30 은 뺐다 ("측정기 아래 페인트 없음")**: 값이 0~20(612개)과 50~190 두 덩어리로 갈리고 20~30 은 0개다. 닳은 페인트라면 연속으로 낮아져야 하는데, 낮은 값은 대부분 1~3 프레임짜리 토막으로 높은 값 사이에 끼어 있다 → 점선 빈칸·차로 변경으로 측정기가 페인트를 벗어난 것으로 본다 (추정). **장소의 성질도 아니다**: 다른 주행이 10 m 안 같은 곳을 지날 때, 한쪽이 30 미만이면 다른 쪽도 30 미만일 확률은 19% 로 기저 비율 14% 와 거의 같다 (149쌍). 닳은 페인트라면 되풀이돼야 한다. 30 은 매뉴얼·법 기준이 아니라 정현우가 11/12 에 온전 표본(100~160)과 손상 표본(0~16) 사이에서 고른 값이다 — 그 정의("30 미만 = 마모")로 잰 결과도 아래에 함께 둔다.
- **어느 선을 재는지는 데이터로 정할 수 없다 (결론).** 데이터 페이지에 설명이 없고, VL4 로 쪽마다 세 가지를 봤지만 어느 쪽도 뚜렷하지 않다:
  (1) 라벨 선 위치에서 잰 선 밝기 대비 vs LTL-M — 낮 왼쪽 0.07 / 오른쪽 −0.11; (2) 자기 차로 선까지 거리로 30 미만을 가르는 AUC — 낮 왼쪽 0.59 / 오른쪽 0.53, 밤 0.55 / 0.46; (3) 그 쪽에 라벨 선이 있을 때·없을 때 30 미만 비율 — 낮 양쪽 모두 24% vs 20% 로 같음, 밤만 왼쪽 선이 없을 때 31% vs 있을 때 20%. 왼쪽(중앙선 쪽) 선일 가능성을 약하게 가리키지만 근거로 쓸 수준은 아니다.

LTL-M 자체는 일관된다 (GPS 로 확인): 같은 주행 1~2초 이웃 프레임끼리 Spearman **0.90** (953쌍, 차이 중앙값 3), 다른 주행이 10 m 안 같은 곳을 지날 때 낮-낮 0.62 (21쌍). 그러나 **같은 곳을 밤에 재면 낮보다 평균 31 높다** (낮-밤 72쌍, 113 → 144, Spearman 0.33). 낮에 "마모"(< 100)가 26% 인데 밤은 2% 다. 그리고 **차선이 아닌 노면 밝기**(화면 아래 가운데 V 중앙값)가 LTL-M 과 강하게 같이 움직인다 — 낮 Spearman **0.72**, 낮·맑음 **0.91**, 밤 0.79. LTL-M 이 주변 밝기(일광)에 크게 영향받는 것으로 보이며(가설), 이 데이터에서는 깨끗한 "페인트 마모 정답"이 아니다.

우리 점수 (LTL-M ≥ 30, 차선 찾은 프레임, 낮을수록 마모):

| 묶음 | 점수 | n (마모) | AUC | best F1 | 2-fold F1 | Spearman(LTL-M) |
|---|---|---|---|---|---|---|
| 낮 | contrast | 984 (244) | 0.32 | 0.40 | 0.40 | −0.29 |
| 낮 | paint_len | 984 (244) | 0.63 | 0.43 | 0.41 | 0.34 |
| 낮 | **★ score (p25 ÷ 주행 중앙값 × paint_len)** | 984 (244) | **0.51** | 0.40 | 0.40 | 0.14 |
| 낮·맑음 | ★ score | 513 (120) | 0.45 | 0.38 | 0.37 | −0.00 |
| 낮·맑음 | contrast | 513 (120) | 0.20 | 0.38 | 0.37 | −0.55 |
| 밤 | ★ score | 523 (10) | 0.51 | 0.06 | 0.02 | −0.06 |

- **★ score 는 LTL-M 마모를 가르지 못한다 (AUC 0.51).** best F1 0.40 은 "전부 마모"라고 부를 때의 F1(마모 25% → 0.40)과 같다. 잠정 기준 1.0 에서 낮 프레임의 18% 를 마모로 부르고 정밀도 0.37 · 재현율 0.26.
- contrast 는 거꾸로 간다 (LTL-M 이 높을수록 낮다). 노면이 밝으면 (차선 − 노면)/노면 이 작아지는데, 그 노면 밝기가 LTL-M 과 같이 오르기 때문이다 (노면 밝기 vs contrast 낮 −0.47).
- 마모 기준을 70 으로 낮추면 (낮 마모 45개) ★ AUC 0.60 · paint_len 0.73 이지만 2-fold F1 은 0.21 · 0.18 로 쓸 만한 기준값이 안 나온다.
- **정현우 정의 (30 미만 = 마모, 30 이상 = 정상, 차선 찾은 프레임)**: ★ AUC 낮 0.62 · 낮·맑음 0.67 · 밤 0.72, contrast 0.66 · 0.74 · 0.76. 2-fold F1 은 ★ 0.39 · 0.41 · 0.45, contrast 0.41 · 0.52 · 0.50 (전부 마모라고 부를 때 F1 ≈ 0.37~0.40). 신호는 있지만 약하다 — 측정기가 페인트를 못 본 순간은 화면에서도 페인트가 덜 보인다(점선 구간) 정도로 본다.
- **정확도는 "전부 정상"보다 낮다** (잠정 1.0, 차선 못 찾은 프레임은 정상으로 봄): 30 미만 정의에서 낮 70.6% vs 전부 정상 76.9% (양성 23%, 정밀도 0.35 · 재현율 0.30), 밤 75.8% vs 78.5%. 마모 < 100 정의에서 낮 69.4% vs 74.0%. 11/12 정현우 측 이전 판정(79.4% vs 80.3%)과 같은 모양이다. 양성이 20% 안팎이라 정확도는 판단 기준으로 쓰지 않고 AUC·F1 을 본다.
- 30 미만 + 100 미만을 함께 마모로 치면 ★ AUC 낮 0.57 · 밤 0.71. 잠정 1.0 에서 30 미만 프레임의 33%(낮) 를 마모로 본다 (정상 프레임은 15%).
- 밤은 마모가 10개뿐이라 판단할 수 없다.

차선 찾기 (VL4 차선 polyline 과 비교, 이번에 처음 생긴 위치 정답, 이미지 3,281장 전부):

| | 자기 차로 선 라벨 있는 프레임 중 차선 찾음 | 채택 선분 길이 중 라벨 선 위(화면 폭 1.5% 이내) | 선분 절반 이상이 라벨 선 위인 프레임 |
|---|---|---|---|
| 낮 (2,126장) | 96.2% | 72% | 80% |
| 밤 (1,088장) | 97.5% | 73% | 78% |

- 표기 구분: **96~97% 는 '차선을 찾은 비율'**, 위치까지 맞은 비율은 **약 72~80%** (선분 길이 기준 72~73%, 프레임 기준 78~80%). 발표에서 둘을 섞지 않는다.
- 차선 위치는 대체로 맞다. 나머지 28% 는 라벨이 없는 선(정지선 근처 짧은 표시, 화면 끝 선 등)이거나 연석·가드레일일 수 있다 — 이 도구는 원인을 나누지 않는다.

**결론 (제안).** `PROVISIONAL_WEAR_SCORE`(1.0)·투표 규칙·`LaneWearMeter` 는 **바꾸지 않았다.** 이 데이터로 고르면 점수·기준값 모두 우연 수준이라, 바꿀 근거가 손 라벨(AUC 0.79)보다 약하다. 다만 손 라벨의 0.79 도 "보이는 마모"에 대한 것이고, **재귀반사 성능(야간 시인성)과 우리 점수는 별개임이 이번에 확인됐다** — 발표에서 "차선 마모"는 "낮 영상에서 보이는 페인트 벗겨짐"으로 말하고, 재귀반사도를 잰다고 하지 않는다. 판정 확정은 여전히 시연 영상 + 반복 관측(#6, 11/22)으로 한다.

한계: 한 도로(0401)·한 묶음(비온전 복합)·약 1 fps 토막, 측정한 선과 색을 모름, LTL-M 의 낮 측정이 밝기에 치우친 것으로 보임, 30 미만 해석은 추정.

GPS CSV 는 형식이 섞여 있다: A `시각,위도,경도,속도(knot),방위,…`(8칸), B `번호,시각,위도,경도,…`(9칸, 3·4번째 칸이 한국 좌표 범위라 구별), C 는 A 뒤에 빈 칸. 시각은 hhmmss.ss(날짜 없음, UTC). `0401_light_clear_smooth_10000005` 는 좌표 대신 다른 신호가 든 잘못된 행이라 `load_gps` 가 버린다 (좌표가 한국 범위 밖이면 None). LTL-M 이 있는 프레임은 모두 A 형식이었다(B 514개 + C 771개 = 1,285개는 LTL-M 없는 프레임 — 정현우 기록과 맞춤) — 앞의 숫자는 이 수정으로 바뀌지 않았다.

**라이선스 (AI Hub 이용정책).** 학습·연구 목적만, 승인 없는 제3자 제공·재배포 금지, 이용 시 "한국지능정보사회진흥원 사업결과" 출처 표기 필수. 그래서 레포에는 집계 숫자만 두고 이미지·프레임별 값(CSV·캐시)은 올리지 않는다. **발표 자료에 이 데이터 프레임을 보여도 되는지는 확인되지 않았다** — 정책에 명시 조항이 없으므로, 보여 주려면 AI Hub 에 문의하거나 우리 시연 영상 프레임을 쓴다.

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
| v3 모델 (균열 데이터 수정) | 11/16~17 학습, **11/18 판단** | 출력 형식은 v2 와 같음 — 쓰게 되면 파일만 바꾼다 (15장) |

## 11. AI 쪽이 받아야 하는 것 (계획 4장)

- **속도 측정 로그 형식**: 프레임·단계별 `{"frame", "t", "stage", "ran", "ms"}` JSONL — `bench_cpu` 실험이 이 로그를 읽는다. 형식이 바뀌면 알려주기. (`interval_sweep` 은 탐지기를 직접 돌려서 이 로그가 필요 없다.)
  주기 적용 전/후 로그를 하나씩 주면 단계별 실행 비율·ms·FPS·영상 1분당 계산 시간을 낸다: `uv run python -m daboyeo_ai.eval.bench_cpu log 전=before.jsonl 후=after.jsonl --out experiments/bench_cpu_log.csv`. 건너뛴 프레임도 `"ran": false` 로 한 줄씩 남겨 주기 (영상 길이·실행 비율 계산에 쓴다). `t` 가 없으면 `--video-fps 30` 을 준다.
- **프레임 샘플링 방식**: 분석 주기(N프레임마다 탐지 등)를 어떻게 적용하는지.
- **시연 영상 + 수동 라벨** (`experiments/gt_events/`): 실제 주행 영상 평가에 쓴다. 라벨 형식은 12장 (제안).

## 12. 수동 라벨 `experiments/gt_events/<video_id>.csv` (제안 — 정현우 확인 필요)

계획 0.10 의 수동 라벨 형식. `eval_detector`·`interval_sweep`(박겸, 4.1·4.3) 과 1.1 가짜 탐지기(정현우)가 **같은 파일**을
읽으므로 형식을 같이 정한다. `<video_id>` = 영상 파일 이름에서 확장자를 뺀 것 (`v1.mp4` → `v1.csv`).

```csv
defect_id,t_start_s,t_end_s,defect_type,memo
P001,12.0,14.5,pothole,오른쪽 차로
C003,30.2,,crack,t_end_s 를 비우면 t_start_s ± 1초
L001,41.0,55.0,lane_wear,구간 전체가 마모
G002,61.5,63.0,guardrail_damage,시설물은 '보임' 만 적는다 (파손 판정 안 함)
```

| 열 | 뜻 |
|---|---|
| `defect_id` | 결함 ID. **두 영상에 같은 결함이 보이면 같은 ID** (계획 0.10, 중복 정리 평가용). 비우면 `<video_id>#줄번호` |
| `t_start_s`, `t_end_s` | 영상 시작부터 초. 화면에 결함이 보이기 시작한 때 ~ 사라진 때. `mm:ss` 도 된다. `t_end_s` 를 비우면 한 시각으로 본다 |
| `defect_type` | `daboyeo_contracts.DefectType` 값: `pothole` `crack` `lane_wear` `guardrail_damage` `sign_damage` |
| `memo` | 자유 |

- 받는 쪽은 너그럽게 읽는다: 헤더 한국어(`ID`·`시작`·`끝`·`시각`·`유형`·`메모`), 시각 하나만 쓰는 `t_s` 열(→ `t_s ± 1초`),
  유형 한국어(`포트홀`·`균열`·`차선 마모`·`가드레일`·`표지판`)와 탐지 클래스 이름(`guardrail`·`sign`), `#` 주석 줄.
  그래도 **위 영어 형식으로 쓰기를 권한다** (가짜 탐지기도 읽기 쉽게).
- 가짜 탐지기(1.1)는 `t_start_s`(또는 `t_s`) 시각에 그 유형 탐지 1건을 내면 된다 → `observations` 행 = 라벨 수.
- 시설물: 파손 판정을 하지 않으므로(PRD v0.5) 평가는 "시설물 위치를 찾았나"로만 하고 결함 지표와 섞지 않는다.
  보이는 가드레일·표지판을 **전부** 적을 여유가 없으면 시설물 행은 빼도 된다 (적지 않으면 정밀도가 낮게 나올 뿐 결함 지표에는 영향 없음).
- 평가 규칙: 정답 창 `[t_start − 0.5초, t_end + 0.5초]` 안의 분석 프레임에 같은 유형 탐지가 하나라도 있으면 찾음.
  예측은 1초 이내로 이어지는 탐지를 한 이벤트로 묶고, 정답 창에 걸리면 맞음. 차선 마모는 `LaneWearTracker.confirmed` 프레임.

실행 (박겸, Mac):

```bash
uv run python -m daboyeo_ai.eval.interval_sweep data/videos/v1.mp4 --cache-dir data/eval_cache   # 모든 프레임 1번 + 간격 1/3/5/10/15
uv run python -m daboyeo_ai.eval.eval_detector data/videos/v1.mp4 --every 3 --cache-dir data/eval_cache   # conf 0.25~0.6 별 P/R/F1
```

클래스별 기준은 `--conf-json experiments/conf_thresholds.json` 을 붙인다 (`class_conf` 를 읽음). 캐시는 영상마다 한 개라,
`interval_sweep`(매 프레임) 을 **먼저** 돌리면 `eval_detector` 가 어떤 `--every` 로든 다시 쓴다 (conf 기준이 캐시보다 낮으면 다시 돈다).

결과는 `experiments/interval_sweep/sweep.csv`(+ `--with matplotlib` 로 실행하면 `sweep.png`), `experiments/eval_detector/summary.{csv,json}`.
캐시(`data/eval_cache/`)는 git 에 올리지 않는다.

## 13. 서버 검증기 V0 측정 (11/15, 결과: **쓰지 않음 — no-go**)

V0 = 서버가 받은 크롭 JPEG 에 **같은 `detector_v2`** 를 한 번 더 돌려 점수를 매기는 검증기 (정현우 제안 V0/V1/V2 중 V0).
코드: `daboyeo_ai.inference.verifier.Verifier(model_path, threads=1).verify(jpeg, defect_type) -> 0~1`
(같은 클래스·중심이 크롭 가운데 60% 안인 탐지의 최고 신뢰도, onnxruntime 만 사용). 크롭은 블랙박스 `crop_snapshot`
(`origin/be/port-existing` 3c62628 `edge/blackbox/sender.py`) 을 그대로 옮긴 `daboyeo_ai.eval.snapshot` 으로 만든다.

```bash
uv run python -m daboyeo_ai.eval.verify_eval --model /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \
    --parts-dir /Users/parkgyeom/daboyeo/data/aihub/parts --records /Users/parkgyeom/daboyeo/data/verify_v0/records.csv
```

**측정 (Set T** = `conf_thresholds` 와 같은 test 데이터·출처, 탐지 기준 = `class_conf_draft`, IoU 0.5 매칭, 탐지 7,558건).
비교 기준(baseline) = 검증기 없이 **탐지 신뢰도 기준만 올려서** 같은 수의 TP(또는 FP)를 남기는 것.

| 클래스 | TP / FP | TP 유지 ≥ 90% 에서 V0 FP 제거 | 같은 TP 유지에서 신뢰도 올리기 FP 제거 | FP 40% 제거 시 TP 유지: V0 / 신뢰도 |
|---|---|---|---|---|
| pothole | 336 / 142 | 19.7% (τ 0.01) | 22.5% | 78.6% / 85.4% |
| crack | 2,043 / 839 | 17.3% (τ 0.05) | 29.4% | 74.6% / 85.8% |
| guardrail | 890 / 381 | 0% (τ 0 — 거를 수 있는 τ 없음) | 0% | 77.5% / 89.6% |
| sign | 2,059 / 868 | 0% (τ 0) | 0.7% | 5.0% / 44.2% (FP 97% 제거 지점) |
| 전체 (클래스별 τ) | 5,328 / 2,230 | **7.8%** (TP 95.8% 유지) | **12.8%** | — |

낮은 신뢰도 구간 (초안 기준 ≤ conf < 0.6, 서버가 차량 2대 이상일 때만 보여주는 구간 — 검증기가 필요한 곳):
TP 2,344 / FP 1,964, 정밀도 0.544. TP 유지 ≥ 90% 에서 V0 FP 제거 **3.9%**, 신뢰도 올리기 7.0%. FP 40% 제거 시 TP 유지는
pothole 73.2% / 77.5%, crack 63.8% / 77.1%, guardrail 62.5% / 78.9% (V0 / 신뢰도). 정밀도는 V0 적용 후에도 0.544 그대로.

- **모든 클래스·두 구간에서 V0 가 '신뢰도만 올리기' 보다 못하다.** 같은 탐지기를 크롭에 다시 돌린 점수는 원래 신뢰도와 거의 같은
  정보이고, 크롭 확대 때문에 오히려 잡음이 더해진다.
- 표지판은 크롭의 **95%(TP)·97%(FP) 에서 다시 못 찾는다** (v = 0). KRID 표지판은 장면 속 작은 물체로 학습되어, 크롭을 640 으로
  키워 넣으면 탐지기가 표지판으로 보지 않는다. 가드레일도 TP 17% 가 v = 0. → '큰 탐지기를 크롭에' (V1) 도 같은 문제가 날 수 있다.
  크롭으로 학습한 분류기(V2) 가 아니면 크롭 검증은 기대하기 어렵다.
- 중심 규칙 대신 위치 무관(`v_any`) 으로 해도 결과는 거의 같다 (JSON `*_v_any`).
- 속도: 크롭 1장 **21.6 ms** 평균 (p95 23.9 ms, 1 스레드, Apple M4 Pro — 서버 CPU 에서는 다시 잴 것) — 속도 기준(≤ 200 ms) 은 통과.

**must-keep 확인** (AI Hub 로드마크 654 VS4 프레임, 레포에 올리지 않음 — 오버레이·크롭은 박겸 Mac `data/verify_d/`).
ID 4개는 모두 `VS4.비온전(복합).zip` 안의 **단일 프레임**(1920×1080) 이고, 앞뒤 ±6 프레임을 같이 돌렸다. 로드마크 라벨에는
포트홀·균열이 없어서 정답 박스 없이 오버레이를 눈으로 봤다 (`verify_mustkeep`).

| 프레임 | 목표 | 목표 프레임 | 앞뒤 13프레임 |
|---|---|---|---|
| `0401_light_rainy_smooth_03000136` | pothole | 0.257 (초안 0.27 미만 → **안 보냄**) | 03000135 에서만 0.284 로 탐지, 그 크롭 v = 0.11 |
| `0401_light_rainy_smooth_03000120` | crack | crack 탐지 없음 (conf 0.05 까지 내려도) | 13프레임 모두 crack 없음 |
| `0401_light_rainy_smooth_03000130` | crack | crack 탐지 없음. 같은 결함 자리에 **pothole 0.21** (초안 미만) | 13프레임 모두 crack 없음 |
| `0401_light_clear_smooth_07000023` | crack | 탐지 0건 (긴 세로 균열이 눈에 보이는데도) | 13프레임 모두 crack 없음 |

- 4개 중 목표 프레임에서 초안 기준으로 잡히는 것은 **0개**. 문제는 검증기 이전에 **탐지기 재현율** (실제 1080p 주행 영상의
  작은·먼 결함) 이다. V0 를 τ ≤ 0.1 로 두면 03000135 의 포트홀 크롭은 통과, τ 0.2 면 탈락.

**판단 (제안 기준: TP 유지 ≥ 90% 에서 FP 제거 ≥ 40%, must-keep 전부 통과, ≤ 200 ms/크롭)**: FP 제거 7.8% (낮은 구간 3.9%),
must-keep 0/4 (탐지기 단계에서 이미 놓침), 속도만 통과 → **V0 no-go**. 오탐을 줄이려면 신뢰도 기준을 올리는 쪽이 낫고
(`experiments/conf_thresholds.csv`), 남는 시간은 실제 주행 프레임에서의 탐지 재현율 쪽에 쓰는 것을 제안한다.
결과 파일: `experiments/verify_v0.json` (클래스별 요약·τ·v=0 비율), `experiments/verify_v0.csv` (τ 곡선).

## 14. 작고 먼 결함 재현율 — 추론 방식만 바꿔 보기 (11/15, 결과: **`road_tiles(2)` 를 선택 옵션으로 추가, 기본은 그대로**)

13장에서 must-keep 4개가 탐지기 단계에서 빠졌다. 모델은 그대로 두고 **프레임을 어떻게 잘라 넣는지만** 바꿔서 나아지는지 쟀다.
1920×1080 프레임을 640 으로 줄이면 배율이 0.33 이라 멀리 있는 포트홀이 몇 픽셀이 된다. 도로 영역을 잘라 조각으로 넣으면 배율이 올라간다.

**960·1280 전체 프레임은 못 쟀다.** `detector_v2.onnx` 의 입력이 `1×3×640×640` 고정이고 YOLO 출력 격자(8400개)가 그 크기로
굳어 있어서, 큰 입력을 넣을 수 없다. 재려면 Colab 에서 `.pt` 로 다시 내보내야 한다 (아래 셀). 대신 2×2 조각(`full_2x2`, 배율 0.61)
으로 1280 입력과 비슷한 배율을 흉내 냈다 — 한 번에 1280 으로 넣는 것과 같지는 않다 (조각 경계에서 큰 물체가 잘린다).

```python
# Colab (02_train_detector 의 Drive 마운트 셀 다음). .pt 를 /content 로 복사해서 내보낸다 —
# Drive 의 detector_v2.pt 에서 바로 export 하면 같은 폴더의 detector_v2.onnx 를 덮어쓴다.
import shutil, pathlib
from ultralytics import YOLO
MODELS = "/content/drive/MyDrive/daboyeo_data/models"
shutil.copy(f"{MODELS}/detector_v2.pt", "/content/detector_v2.pt")
for sz in (960, 1280):
    p = pathlib.Path(YOLO("/content/detector_v2.pt").export(format="onnx", imgsz=sz, opset=17, simplify=True, dynamic=False))
    shutil.copy(p, f"{MODELS}/detector_v2_{sz}.onnx")
    print(sz, p)
```

받은 파일은 `OnnxDetector("ai/models/detector_v2_1280.onnx")` 로 그대로 쓴다 (입력 크기는 파일에서 읽는다). 비용은 640 의 약
2.25배(960)·4배(1280) 로 예상 — 재지 않았다.

**어떻게 쟀나** (`uv run python -m daboyeo_ai.eval.recall_res ...`, 사용법은 파일 맨 위): 로드마크 654 VS4 프레임(1920×1080) 으로
① must-keep 4개 + 앞뒤 ±6 프레임 (약 1초 간격), ② 6개 조건(낮 맑음·맑음 정체·안개·비·비 정체·밤 맑음) 에서 50장씩 고른 300장.
정답 라벨이 없어서 ②는 클래스별 탐지 수와 'base 에 없던 새 탐지' 만 세고, 새 탐지는 박겸 Mac 에서 크롭을 눈으로 봤다.
탐지 기준은 모두 모델 카드 `class_conf_draft` (pothole 0.27 · crack 0.29 · guardrail 0.26 · sign 0.24). 집계 `experiments/recall_res.json`,
이미지(레포에 안 올림)는 `data/recall_res/` — `mustkeep/<변형>/contact_*.jpg`, `new_dets/<변형>.jpg`, `new_dets/road_new_crops.jpg`.

| 변형 | 모델 실행/프레임 | ms 1스레드 | ms 2스레드 | must-keep 목표 프레임 | 03000136 앞뒤 13장 중 탐지 | 300장 탐지/프레임 p · c · g · s | base 대비 새 탐지/프레임 p · c | base 에서 잃은 g · s |
|---|---|---|---|---|---|---|---|---|
| **base** (지금, 전체 640) | 1 | 20.8 | 16.6 | **0/4** (pothole 0.257, 균열 3개 없음) | 1 | 0 · 0 · 0.11 · 0.26 | — | — |
| roi (도로 45~100%, 가로 전체, 640) | 1 | 21.1 | 16.7 | 0/4 (0.19) | 1 | 0.007 · 0.003 · 0 · 0 | 0.007 · 0.003 | 0.11 · 0.26 (전부) |
| full_2x2 (1280 흉내) | 4 | 85.2 | 65.5 | 1/4 (pothole 0.54) | 3 | 0.033 · 0.013 · 0.043 · 0.197 | 0.033 · 0.013 | 0.097 · 0.197 |
| roi_t2 (도로 좌우 2조각) | 2 | 42.5 | 33.6 | 1/4 (0.45) | 4 | 0.023 · 0.007 · 0 · 0 | 0.023 · 0.007 | 0.11 · 0.26 (전부) |
| roi_t3 (도로 3조각) | 3 | 65.5 | 50.9 | 1/4 (0.44) | 2 | 0.06 · 0.013 · 0 · 0.003 | 0.06 · 0.013 | 0.11 · 0.257 |
| **base+roi_t2** (`road_tiles(2)`) | 3 | 65.5 | 49.2 | **1/4** (0.45) | **5** | 0.023 · 0.007 · 0.11 · 0.26 | 0.023 · 0.007 | **0 · 0** |
| base+roi_t3 (`road_tiles(3)`) | 4 | 85.3 | 66.2 | 1/4 (0.44) | 3 | 0.06 · 0.013 · 0.11 · 0.26 | 0.06 · 0.013 | 0 · 0 |

속도: Apple M4 Pro, onnxruntime CPU, 1920×1080 30장 평균 (자르기·디코드·NMS 포함). **실제 블랙박스 노트북에서 다시 잴 것.**

- **포트홀 목표(03000136)** 는 도로 조각을 넣으면 0.257 → **0.45** 로 초안 기준을 넘고, 같은 자리에서 잡힌다 (오버레이 확인).
  앞뒤 13장 중 탐지된 장도 1 → 5 로 늘어서, 서버가 여러 프레임·차량 관측을 묶기 쉬워진다.
- **균열 목표 3개는 어떤 변형으로도 못 잡는다** (03000120·07000023 은 crack 0.29 이상 0건, 07000023 은 3조각에서 0.16 이 최고).
  단, 03000130 의 결함은 조각을 넣으면 **pothole 0.495** 로 잡힌다 (클래스는 틀렸지만 신고 대상으로는 올라간다).
  → 균열은 추론 방식으로 해결되지 않는다. 학습 데이터(위에서 내려다본 사진 위주)와 주행 시점의 길고 가는 균열이 다른 문제로 보인다.
- **표지판·가드레일은 전체 프레임에서만 잡힌다.** 도로 조각만 쓰면(roi·roi_t2·roi_t3) 전부 잃고, 2×2 는 표지판을 키워 넣어
  절반 넘게 잃는다(13장의 '키운 크롭에서 표지판을 못 찾는다' 와 같은 현상). 그래서 쓰려면 **전체 + 조각** 이어야 한다.
- **새 탐지(오탐 비용, 눈으로 본 것 — 박겸 확인 필요)**: `base+roi_t2` 는 300장 중 7장에서 포트홀 7·균열 2, 모두 9건이 새로 생겼다
  (0.03건/프레임). 크롭으로 보면 실제 손상으로 보이는 것 7건 정도(실링한 균열 위 패임, 거북등 균열 옆, 03000130 결함, 먼 작은 패임,
  차선 옆 파임, 가는 균열), 노면 질감(소성변형 결)·보닛 근처 2건은 애매하다. 3조각은 새 탐지가 2배지만 **닳은 차선·노면 표시를 포트홀로 보는 오탐**이 대부분이다.
  2×2 는 하늘·나무·표지판을 포트홀/균열로 보는 오탐이 생긴다.
- **클래스별 기준(`class_conf_draft`)**: 조각은 배율이 0.61 이라 RDD2022·도로장애물 학습 사진의 물체 크기에 더 가깝다.
  새 탐지 신뢰도는 0.28~0.49 로 초안 기준 바로 위에 몰려 있다. 표본이 9건뿐이라 기준을 바꿀 근거는 없다 — **그대로 두고**,
  시연 영상에서 `--road-tiles 2` 로 다시 본다. RDD2022·179 test 로 정밀도 확인은 하지 않았다: 그 사진들은 노면을 가까이 찍어서
  '지평선 아래 도로 조각' 이 성립하지 않는다.

**제안 (정현우 확인 필요)**: 기본은 640 전체 한 번 그대로. 시연 영상에서 포트홀이 덜 잡히면
`OnnxDetector(..., windows=road_tiles(2))` 를 켠다 — 출력 형식(`Detection`, 0~1 bbox) 은 같고 시간만 약 3배
(M4 Pro 1스레드 65 ms, 2스레드 49 ms). 분석 주기를 두면(`interval_sweep`) 초당 2~3번 돌려도 1코어의 15~20% 정도로 예상한다.
3조각·2×2 는 쓰지 않는다 (오탐·표지판 손실). 균열은 데이터 쪽 문제로 따로 본다.

```python
from daboyeo_ai.inference import OnnxDetector, load_class_conf, road_tiles
det = OnnxDetector("ai/models/detector_v2.onnx", threads=1, conf=load_class_conf("ai/models/detector_v2.json"),
                   windows=road_tiles(cols=2))   # 전체 + 도로(높이 45~100%) 좌우 2조각, 10% 겹침
```

시연 영상으로 확인: `eval_detector`·`interval_sweep` 에 `--road-tiles 2` 를 주면 같은 방식으로 돈다 (캐시는 모델 이름에
`+road_tiles2` 가 붙어 따로 맞춘다 — 같은 `--cache-dir` 을 쓰면 서로 덮어쓰니 폴더를 나눌 것).
한계: 한 도로(0401)·1 fps 토막, 정답 라벨 없음, 새 탐지 판정은 한 사람이 눈으로 본 것, `top=0.45` 는 이 카메라의 지평선(약 50%)
기준이라 카메라 각도가 다른 시연 영상에서는 맞춰야 할 수 있다.

## 15. 균열 v3 — 도로장애물(179) 크랙 데이터 고치기 (11/15, 결과: **179 크랙 이미지 zip 이 라벨과 다른 사진 → 크랙 라벨을 같은 이름의 포트홀(TOA) 이미지에 붙여 `road_v3.zip` 을 새로 만듦. v3 학습 준비 완료, 11/18 판단**)

14장에서 must-keep 균열 3개(로드마크 654 VS4 `0401_light_rainy_smooth_03000120`·`..._03000130`·`0401_light_clear_smooth_07000023`)는
추론 방식을 바꿔도 안 잡혔다. v2 의 179 test crack mAP50 은 0.001 — 한국 주행 시점 균열 학습 신호가 사실상 없었다.

### 15-1. 진단 (숫자)

`road.zip`(v2) 의 179 크랙 이미지 2,289장 (C_Mainroad_B02 1,996 + C_Frontback_B01 293). 길이·크기는 학습 이미지(긴 변 640) px.

| | RDD2022 train crack | 179 크랙 (v2 규칙 = 선 하나에 박스 하나) |
|---|---|---|
| 이미지 | 16,313 | 2,289 |
| 박스/이미지 중앙값 · p90 · p99 · 최대 | 2 · 4 · 10 · 41 | 선 19 · 79 · 225 · 704 (박스로 남은 것 13 · 50 · 123 · 350) |
| 박스 가로 p10 / 중앙값 / p90 | 20 / 89 / 289 | 3 / 15 / 77 |
| 박스 세로 p10 / 중앙값 / p90 | 16 / 68 / 214 | 1 / 8 / 39 |
| 긴 변/짧은 변 중앙값 · p90 | 2.0 · 6.3 | 2.3 · 14.6 |
| 4 px 미만이라 버린 선 | — | 36% (세로 4 px 미만 30%) |

179 폴리라인: 점 수 중앙값 4 (p90 12), 선 길이 중앙값 24 px (p10 6 / p90 107) — 크랙 하나를 짧은 선 여러 개로 그렸다.

**더 큰 문제 — 크랙 이미지가 라벨과 다른 사진이다.**

- 박스를 그려 보니 선이 사람·차·하늘·횡단보도 위에 있고, 바로 옆의 뚜렷한 균열에는 라벨이 없는 이미지가 많았다.
  무작위 24장 (seed 7, 선 그대로): 보이는 균열을 따라가는 것 **8장 (33%)**. AI Hub 원본 zip(C_Mainroad_B04, 변환 코드 안 거침) 12장도 3~4장만 맞음.
- 같은 이름(`V3F_HY_..._05263`)이 포트홀(TOA) 이미지 zip 과 크랙 이미지 zip 에 둘 다 있는 프레임 215쌍: **전부 다른 사진**
  (평균 픽셀 차 44/255, 같은 것 0쌍). 크랙 선을 TOA 쪽 사진에 그리면 **균열 위에 정확히 맞는다** (6쌍 눈으로 확인).
  → 179 의 `2.Images_new_211012/2.CRACK/` 이미지가 이름과 다른 장면이다. 라벨(JSON)은 맞다. 포트홀(TOA)은 이미지·라벨이 맞다 (179 test pothole 0.573).
- 크랙 라벨 274,984개 중 **108,445개(39%)** 가 TOA 이미지 zip 에 같은 이름의 프레임이 있다 (라벨 zip 이름 비교, `pipeline.label_stems`).
  예: TOA `Mainroad_B02.zip`(Training, 11 GB) 하나에 C_Mainroad_B02 3,089 + C_Mainroad_B03 4,008 = 7,097 프레임.
- v2 의 179 test crack 0.001 은 박스가 잘게 쪼개진 탓 + **엉뚱한 사진** 탓이다. v2 학습의 179 크랙 2,289장은 사실상 잡음이었다.

### 15-2. 새 데이터 규칙 (선택 옵션 — 기본은 v2 그대로)

1. **크랙 라벨 → TOA 이미지** (`pipeline.process_crack_on_toa`, CLI `--crack-on-toa <TOA 이미지 zip 이름>`): TOA 이미지 zip 에서 크랙
   라벨이 있는 프레임만, 박스 = 크랙 + 같은 프레임의 TOA 포트홀 박스. 라벨의 이미지 크기가 실제 이미지와 다르면 버림.
   크랙 이미지 zip 은 쓰지 않는다 (`--road-crack-zips 0`). 짝 크랙 라벨 zip 은 이름이 겹치는 것을 자동으로 찾는다.
2. **선 묶어 박스 하나** (`yolo.CrackMerge` / `merge_polylines`, `--crack-rule v3`): 이미지의 크랙 선을 모아
   ① `merge_px`=16 px 두께로 그려 닿는 선끼리 한 묶음 ② 묶음마다 원래 점들의 외접 박스 ③ 선 길이 합 < 24 px 는 버림
   ④ 짧은 변을 12 px 까지 늘림 (가는 세로 균열도 박스) ⑤ 길이 합 순으로 이미지당 20개까지.
   값은 600장으로 merge 6·12·20·32 px × 최소 길이 0·24 px 를 비교해 RDD 와 가장 비슷한 쪽 (merge 12~20 → 박스/이미지 중앙값 3,
   가로 69~85, 세로 35~45; 32 는 화면 절반짜리 박스가 많아짐).
   확인: 포트홀(TOA) zip 3개를 같은 코드로 다시 쓰면 v2 와 개수가 정확히 같다 (예: J03 포트홀 1,641·버림 52·배경 68).

### 15-3. `road_v3.zip` 만들기 (Mac)

```bash
uv run python -m daboyeo_ai.data.prepare_local --out data/aihub --no-krid --krid-labels selected \
    --crack-rule v3 --tag v3 --road-crack-zips 0 --crack-on-toa Mainroad_B02.zip
```

`work_v3/`, `progress_v3.json`, `parts/road_v3.zip` 에 따로 만든다 (`road.zip`·`work/` 는 안 건드림). 포트홀 TOA 3개 zip 은 `work/` 에서
이미지를 다시 받지 않고 라벨만 다시 쓴다 (`pipeline.relabel_road_obstacle`). TOA `Mainroad_B02.zip` 은 새로 받는다 (11 GB, 약 14분).
**AI Hub 는 이어받기가 안 된다** (curl 33) — 받다가 끊으면 처음부터.

결과 (11/15, 빌드 1분 30초 + TOA zip 받기 15분): **`road_v3.zip` 303 MB, 이미지 5,078장** (train 4,064 / val 491 / test 523).

| | `road.zip` (v2) | `road_v3.zip` |
|---|---|---|
| 크랙 이미지 (사진이 맞는 것) | 2,058 (대부분 엉뚱한 사진) | **3,506** (TOA `Mainroad_B02.zip` 의 크랙 라벨 프레임 7,097 중 절반, `--crack-on-toa-stride 2`) |
| 크랙 박스 train / val / test | 34,210 / 4,020 / 4,927 | 10,773 / 1,343 / 1,387 |
| 크랙 박스/이미지 중앙값 · p90 · 최대 | 13 · 49 · 350 | **3 · 7 · 20** |
| 크랙 박스 가로 p10 / 중앙값 / p90 (640 px) | 7 / 21 / 99 | 23 / 74 / 291 |
| 크랙 박스 세로 p10 / 중앙값 / p90 | 5 / 14 / 54 | 12 / 39 / 200 |
| 포트홀 박스 train / val / test | 2,770 / 379 / 296 | 3,825 / 512 / 452 (크랙 프레임의 TOA 포트홀도 들어감) |

- 무작위 12장을 박스로 그려 확인: 12장 모두 박스가 실제 균열(거북등·세로·횡단보도 위 균열) 위, 포트홀 박스도 맞음.
  거북등이 차로 전체면 박스도 크다 (RDD D20 과 같은 성격).
- **v2 가 이 test 크랙을 얼마나 잡나** (ONNX, crack 0.29): 크랙이 있는 test 369장 중 크랙 탐지가 하나라도 있는 장 **4장 (1%)**,
  박스 재현율 IoU 0.3 기준 0.002. 한국 주행 시점 균열을 v2 는 사실상 못 본다 — v3 로 나아지는지를 여기서 본다.
- 같은 이름이 TOA 다른 묶음(`Mainroad_B02` 포트홀 part)에도 있는 480장은 크랙+포트홀 라벨 쪽을 쓴다 (zip 만들 때 먼저 넣음).

더 넣고 싶으면 (크랙 프레임/GB 순): `Mainroad_B03.zip` 11 GB → 6,491 (C_Mainroad_B04 5,870 + B03 621) · `1.Frontback_E01.zip` 10 GB → 5,852 (골목) ·
`1.Frontback_G01.zip` 11 GB → 5,533 · `5.Mainroad_G01.zip` 11 GB → 5,189. `--crack-on-toa` 에 이름을 더한다.

### 15-4. v3 학습 (Colab, `02_train_detector.ipynb` — 설정 셀이 v3)

`VERSION="v3"`, `PARTS=["krid","rdd2022","road_v3"]`, `BASE_MODEL=detector_v2.pt` (Drive `models/`), `EPOCHS=15`.
`road` 와 `road_v3` 를 같이 고르면 1단계에서 멈춘다 (같은 이미지가 있다). `DATASET` 은 버전마다 따로. 새 셀 **3-1 출처별 평가**:
test 를 RDD2022 / 179 / KRID 로 나눠 v3 와 v2 를 **같은 라벨로** 재고 `detector_v3.json` 의 `eval_by_source` 에 넣는다. `is_finished` 확인은 그대로.
예상 시간 (재지 않음 — train 약 3만 장, YOLO11n 640, batch 32, Colab 데이터 로더가 병목): T4 약 4분/epoch → 15 epoch 약 1시간
+ 풀기·평가·내보내기 15분, L4 약 40분, A100 약 30분. Colab 단위로 T4 약 2~3, L4 약 3~4, A100 약 7~8 (Pro 100 단위 중).

### 15-5. v3 평가 (Mac, v3 가 생기면)

```bash
uv run python -m daboyeo_ai.eval.compare_v3 \
    --old /Users/parkgyeom/daboyeo/ai/models/detector_v2.onnx \
    --new /Users/parkgyeom/daboyeo/ai/models/detector_v3.onnx \
    --new-card /Users/parkgyeom/daboyeo/ai/models/detector_v3.json \
    --mustkeep-dir /Users/parkgyeom/daboyeo/data/roadmark/verify_mustkeep \
    --sample-dir /Users/parkgyeom/daboyeo/data/recall_res/frames \
    --out-dir /Users/parkgyeom/daboyeo/data/compare_v3
```

must-keep 4개(균열 3 + 포트홀 1) ±6 프레임, 14장의 300장 표본 — 둘 다 `base`(전체 640) 와 `base+roi_t2`(`road_tiles(2)`).
기준은 두 모델 모두 v2 카드의 `class_conf_draft` (crack 0.29). 집계 `experiments/compare_v3.json`, 이미지는 `--out-dir`
(`mustkeep/<old|new>_<변형>/contact_*.jpg`, `new_dets/<변형>.jpg` — v3 에서 새로 생긴 탐지 모음, 눈으로 볼 것).
v2 끼리 돌려 확인함: 14장 값과 같다 (base 포트홀 0.26, 조각 0.45·13장 중 5장, 균열 0) — Mac M4 Pro 80초.

**Go 기준 (제안, `compare_v3` 상단 상수 — 11/18 박겸 확정):**
(1) must-keep 균열 3개 중 **2개 이상**을 `base` 에서 crack 0.29 이상, (2) RDD2022 test crack·pothole mAP50 이 v2 보다
**0.02 넘게 안 떨어짐** (v2: crack 0.584, pothole 0.418), (3) 300장 표본에서 v2 에 없던 pothole+crack 새 탐지 **≤ 0.10/프레임**
(v2 base 는 0), (4) guardrail+sign 잃은 것 ≤ 0.03/프레임. (1) 만 안 되고 나머지가 되면 179 test crack 이 오른 것만으로 v3 를 쓸지 따로 정한다.

### 15-6. v3 평가 결과 (11/16, 결과: 제안 = v2 유지 (no-go) → **시연 클립 결과로 v3 채택, 16장**)

`detector_v3.onnx` (v2 에서 15 epoch, `krid`+`rdd2022`+`road_v3`, Colab T4). ONNX 일치: 10장에서 Ultralytics 박스 65개 중 65개 맞음.
속도는 v2 와 같다 (M4 Pro, 1280×720 150장, 탐지 중앙값 1스레드 v2 20.4·20.6 / v3 20.5·20.7 ms, 2스레드 16.6 / 16.5~16.7 ms).
집계: `experiments/compare_v3.json` (15-5 명령 그대로), `experiments/compare_v3_review.json` (v3 자기 기준으로 다시 잼·눈 분류·속도),
`experiments/conf_thresholds_v3.{json,csv}`. 이미지는 레포 밖 `data/compare_v3/` (`new_crops/`, `lost_crops/`, `gained_gs/` 는 탐지 하나씩 자른 것).

**test 사진 (Colab, 같은 라벨로 v2 vs v3, mAP50)**

| 출처 | v2 | v3 |
|---|---|---|
| RDD2022 crack · pothole | 0.584 · 0.418 | 0.582 · 0.425 |
| 도로장애물(179) crack · pothole | 0.007 · 0.496 | **0.265** · 0.508 |
| KRID guardrail · sign | 0.612 · 0.659 | **0.716 · 0.703** |

참고 (Mac, `conf_thresholds` 와 같은 IoU 0.5 매칭): 179 크랙 test (`road_v3.zip`, 크랙 박스 1,387) 에서 crack 0.30 일 때
v3 P 0.62 / R 0.15, v2 R 0.002. 정밀도 0.7 을 넘기려면 v3 crack 0.41 (R 0.10).

**Go 기준 (15-5) — 실제 1080p 주행 프레임**

| 기준 | 필요 | `base` (v2 카드 기준) | `base` (v3 자기 기준) | `road_tiles(2)` (v2 카드 기준) | 판정 (`base`) |
|---|---|---|---|---|---|
| (1) must-keep 균열 3개 중 목표 프레임 명중 | ≥ 2 | **1** (`07000023` 0.63; `03000120` 0.17, `03000130` 0.06) | 1 | 1 (`03000120` 0.25, `03000130` 0.23 — 앞뒤 ±6 프레임에서는 3·4장 명중) | ✗ |
| (2) RDD2022 test mAP50 하락 | ≤ 0.02 | crack 0.002 · pothole −0.007 (오름) | — | — | ✓ |
| (3) 300장 표본 새 pothole+crack /프레임 | ≤ 0.10 | **0.22** (전부 crack, 84/300장) | 0.21 (정밀도 0.8 기준이면 0.12) | 0.59 | ✗ |
| (4) 잃은 guardrail+sign /프레임 | ≤ 0.03 | **0.155** (진짜만 세면 0.03 — 아래) | 0.14 | 0.155 | ✗ (경계) |

must-keep 포트홀 `03000136` 은 v3 가 0.32 로 명중 (v2 0.26 놓침). v3 자기 기준이면 must-keep 결과는 바뀌지 않는다 (균열 1개, 포트홀 명중).
정밀도 0.8 기준(crack 0.42)이면 새 crack 은 0.12/프레임으로 줄지만 포트홀 must-keep 도 놓친다.

**v3 에서 새로 생긴 crack 66개 눈 분류 (`base`, 300장, crack 0.29)**: 진짜 균열 **16** (불규칙한 세로·거북등 균열, 실링한 균열) ·
애매 **27** (곧은 세로 이음매(시공 줄눈)·패치 경계·흐린 선 — 사진으로 균열인지 줄눈인지 못 가림) · 분명한 헛것 **23**
(차선·노면 표시 9, 그림자·빛 반사 5, 새 포장/색 포장 경계 4, 연석·갓길 3, 젖은 노면 타이어 자국·얼룩 2). 새 pothole 은 0.
→ 분명한 헛것만 0.077/프레임, 애매한 것까지 0.17/프레임. 박스가 차로 하나만 할 정도로 크다 (거북등처럼 학습된 큰 박스).

**guardrail·sign (기준 (4))**: 잃은 46개 중 진짜 표지판 26·금속 가드레일 6 (나머지 콘크리트 중앙분리대 6·흐린 것·헛것),
새로 잡은 39개 중 진짜 표지판 17·금속 가드레일 6 (나머지 표지판 뒷면·문형 지주 8·중앙분리대 4·터널 벽 등 4).
v3 는 다른 표지판을 잡는 '교체'가 많고, 진짜만 세면 순손실 9개 = **0.03/프레임** (기준 경계). KRID test 가 오른 것만큼 실제 영상이 좋아지지는 않았다.
(참고: v3 학습의 KRID 는 v2 때보다 커서 (train sign 33,390 vs 15,571) KRID test 상승의 일부는 데이터 차이일 수 있다.)

**v3 클래스별 기준 (`conf_thresholds`, v2 와 같은 test·규칙, v3 카드 `class_conf_draft`)**

| 클래스 | 정밀도 ≥ 0.7 | 정밀도 ≥ 0.8 (`class_conf_p08`) | v2 (≥ 0.7 / ≥ 0.8) |
|---|---|---|---|
| pothole | **0.31** (P 0.71 / R 0.34) | 0.40 (0.80 / 0.28) | 0.27 / 0.39 |
| crack | **0.30** (0.70 / 0.48) | 0.42 (0.81 / 0.38) | 0.29 / 0.40 |
| guardrail | **0.21** (0.71 / 0.67) | 0.31 (0.81 / 0.62) | 0.26 / 0.42 |
| sign | **0.21** (0.70 / 0.68) | 0.34 (0.81 / 0.62) | 0.24 / 0.37 |

**제안 (11/18 박겸 확정): v2 유지.** 4개 기준 중 (1)·(3) 이 분명히 안 되고 (4) 는 경계. v3 는 한국 주행 시점 균열을 보기 시작했지만
(179 crack 0.007 → 0.265, 진짜 균열 16개 새로 잡음) 실제 영상에서 이음매·차선·그림자를 균열로 잡는 일이 많고, must-keep 균열은 3개 중 1개.
- 블랙박스 기본 모델·경로는 바꾸지 않았다 (`detector_v2.onnx` 그대로). 정현우 쪽 할 일 없음.
- 만약 v3 를 쓰기로 하면 정현우는 **모델 파일 이름만** `detector_v2.onnx` → `detector_v3.onnx` 로 바꾸고, 클래스별 기준을 쓰면
  `load_class_conf("ai/models/detector_v3.json")` 으로 읽는다 (입출력·클래스 순서 같음).
- 다음 후보 (AI 쪽, 필요하면): 시공 줄눈·차선·그림자 프레임을 배경(라벨 없음)으로 넣어 다시 미세조정, 또는 crack 만 서버 노출 기준을 0.42 이상으로.

## 16. 시연 클립 gn_a·gn_b 로 v2 vs v3 (11/16, 결과: **v3 + 균열 기준 0.42 — 11/16 박겸 확정**)

15-6 에서 "최종 선택은 시연 클립에서" 로 미룬 것. 정현우가 Drive `daboyeo_data/` 에 올린 클립
`gn_a_2050-2510.mp4` (강남대로 북행, 흐림·이슬비, 1920×1080 59.94fps, 262초) 와 `gn_b_3320-3620.mp4` (같은 구간, 맑음, 60fps, 182초) 를
0.5초 간격(블랙박스 `--detect-every 30` 과 같음) 같은 프레임 889장에 v2·v3 를 돌렸다. 기준: v2 = `experiments/conf_thresholds.json`,
v3 = `experiments/conf_thresholds_v3.json` (각자 카드 기준). 클립은 YouTube 표준 라이선스라 영상·크롭은 레포 밖 `data/demo_clip_eval/`.

```bash
uv run python -m daboyeo_ai.eval.demo_clip run --tiles \
    --clips data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4     # M4 Pro 약 80초
# data/demo_clip_eval/<clip>/<v3_only|v2_only|t2_gain>-<class>_N.jpg 를 보고 data/demo_clip_eval/review.json ({크롭 id: real|amb|fp}) 작성
uv run python -m daboyeo_ai.eval.demo_clip summarize                          # → experiments/demo_clip_compare.json
```

### 16-1. 분당 탐지 (기준 이상)

| 클립 | 모델 | pothole | crack | guardrail | sign |
|---|---|---|---|---|---|
| gn_a (4.4분) | v2 | 0 | 0 | 2.5 | 99 |
| | v3 | 0 | 0 | 17.6 | 209 |
| | v3 정밀도 0.8 기준 (`class_conf_p08`) | 0 | 0 | 7.1 | 124 |
| gn_b (3.0분) | v2 | 0 | 0.3 | 1.3 | 34 |
| | v3 | 0 | **14.5** | 9.2 | 86 |
| | v3 정밀도 0.8 기준 | 0 | 7.3 | 5.0 | 53 |

- gn_a (젖은 노면) 는 두 모델 모두 균열·포트홀 0. 차이는 gn_b 에서만 난다.
- 차가 신호에 멈춰 있으면 같은 헛것이 0.5초마다 반복돼 분당 수가 부푼다 (예: gn_a v3 sign, 같은 건물 유리 한 곳이 182~241초 동안 42번).
- v3 crack conf (gn_b, 44개): p10 0.31 · 중앙값 0.41 · p90 0.60.

### 16-2. 한쪽에만 있는 탐지 눈 분류 (크롭 단위, real / 애매 / 분명한 헛것)

| 무엇 | 개수 | real | 애매 | 헛것 | 메모 |
|---|---|---|---|---|---|
| v3 에만 있는 crack (gn_b, 전부) | 44 | **26** | 8 | 10 | real = 버스차로 파란 선 옆 긴 세로 균열(120~131초, 20장)·다른 균열 4곳. 헛것 = 교차로 블록 포장 이음매·횡단보도 |
| 　└ crack 0.42 이상만 | 22 | **19** | 2 | **1** | 0.42 = v3 정밀도 0.8 기준. 블록 포장 헛것(conf 0.31~0.39)이 모두 빠진다 |
| v2 에만 있는 crack | 1 | 0 | 0 | 1 | 블록 포장 |
| v2 에만 있는 sign (전부) | 49 | 9 | 7 | 33 | gn_a 헛것 대부분 = 파란 유리 건물·광고판·신호등 |
| v2 에만 있는 guardrail (전부) | 2 | 1 | 0 | 1 | |
| v3 에만 있는 sign (표본 72 / 688) | 72 | 7 | 17 | 48 | gn_a 표본 36장 real 0 (광고판·유리벽), gn_b 36장 real 7 (안내·주의 표지) |
| v3 에만 있는 guardrail (gn_a 표본 36 / 66) | 36 | 34 | 1 | 1 | 보도 펜스·중앙 가드레일 — v3 가 실제로 더 잡는다 |

장소로 묶으면 (2초 안 연속은 한 곳): v3 crack real 5곳 vs 헛것 5곳 (기준 0.30), 0.42 이면 real 4곳 vs 헛것 1곳.

### 16-3. 채택 기준 (제안) 판정

| 기준 | 값 | 판정 |
|---|---|---|
| 균열 real 증가 ≥ clear FP 증가 | +26 vs +9 (장소: +5 vs +4). 0.42 이면 +19 vs 0 | ✓ |
| 포트홀·가드레일·표지판 real 순손실 ≤ 0.03/프레임 | v2 에만 있는 real 10개 = 0.011/프레임 (v3 에만 있는 real 을 빼면 −0.035) | ✓ |

**확정 (11/16 박겸): v3 로 바꾸고, 균열만 기준을 0.42 로 올린다** (`{"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}`
= `ai/models/detector_v3.json` 의 `class_conf_draft`).
- v2 는 이 두 클립에서 진짜 균열을 하나도 못 잡는다 → 시연에서 균열을 보여 주려면 v3 가 필요하다.
- 15-6 의 걱정(이음매·차선을 균열로)은 여기서도 보인다 (블록 포장 이음매). 서버가 여러 차량 관측을 묶으므로 같은 헛것이 여러 대에서 반복되면 결함으로 확정된다 → 균열 0.42 로 막는다. 0.42 에서 남는 real 은 긴 세로 균열 포함 4곳.
- 포트홀은 두 클립 모두 두 모델 0. must-keep 포트홀(15-6)은 0.31 에서 v3 가 잡으므로 포트홀은 0.31 그대로.
- 표지판: v3 가 광고판·유리벽을 표지판으로 더 잡는다 (gn_a 99 → 209/분). 서버로 안 보내고 화면·로그에만 보이므로(11/16 확정) 판정에 넣지 않았다. 화면이 시끄러우면 sign 0.34 (gn_a 124/분).
- 한계: 정답 라벨 없이 크롭을 눈으로 본 것. 균열 real 26장 중 20장은 같은 균열 하나. 클립 2개·7.4분.

### 16-4. 자막·품질 게이트·도로 조각

- **아래 자막 띠 (gn_a 도로명·동 이름, 프레임 아래 약 10%)**: v2·v3 모두 아래 6~15% 띠 안 탐지 0 (gn_b 의 v3 crack 19개가 아래 15% 에 있지만 보닛 바로 앞 노면 균열 — gn_b 는 자막 없이 워터마크만). → **마스킹 필요 없음.**
- **위 교차로 이름 자막 (gn_a, 가운데 위 높이 4~18%, 가끔 뜸)**: v2 가 2번 sign 으로 잡음 (conf 0.40·0.47, 253·256초), v3 0번. 같은 자리에 머리 위 안내 표지판이 있어서 위를 가리면 진짜 표지판도 잃는다 → 마스킹하지 않는 것을 제안.
- **품질 게이트 (`edge/blackbox/quality.measure`, 읽기 전용)**: 두 클립 모두 889장 통과 100% (skip 0). 흐림·이슬비 gn_a 도 노면 영역 선명도·대비가 기준 이상.
- **`road_tiles(2)` (v2, 선택)**: gn_a 포트홀 0 → 0, gn_b 0 → 2.0/분 (6개: real 1 = 차로 가장자리 포장 깨짐, 애매 4, 헛것 1). 시간은 3배. 기본은 그대로 두는 것을 제안.

### 16-5. 정현우에게 전할 것 (제안)

- **[ ] 정현우**: Drive `daboyeo_data/models/detector_v3.onnx` 를 받아 `ai/models/` 에 두고, 블랙박스 모델 경로를 `detector_v3.onnx` 로,
  클래스별 기준은 `load_class_conf("ai/models/detector_v3.json")` (= 위 dict). 입출력·클래스 순서·속도 같음 (15-6).
- 자막 마스킹은 하지 않아도 된다 (아래 띠 탐지 0). 위 교차로 자막은 v3 에서 안 잡힘.
- 품질 게이트는 두 클립 모두 건너뛰는 프레임이 없다.

## 17. 시연 클립 gn_a·gn_b 로 차선 마모 판정 확인 (11/16, 결과: **이 두 클립에서는 시연에 쓸 수 없음 — 기준·투표는 바꿔도 해결 안 됨, 코드 기본값 그대로**)

> 오픈 이슈 #6 (11/22) 용. 처음으로 실제 시연 영상에서 `LaneWearMeter` + `LaneWearTracker` (기준 1.0, 5 중 3 투표) 를 돌렸다.
> 정답 라벨이 없어 Claude 가 오버레이를 눈으로 보고 real(보이는 페인트 손실) / amb / fp(헛것) 로 나눴다. 집계 `experiments/lane_wear_demo.json`.

```bash
uv run python -m daboyeo_ai.eval.lane_wear_demo measure --clips data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4
uv run python -m daboyeo_ai.eval.lane_wear_demo sheets      # data/lane_wear_demo/<clip>/sheets/ (레포 밖)
# data/lane_wear_demo/review.json 에 눈 분류를 쓰고
uv run python -m daboyeo_ai.eval.lane_wear_demo summarize --stopped gn_b_3320-3620:30-96,152-172
```

- **주기는 블랙박스와 같게**: `edge/blackbox/run.py` 의 `--lane-every 3` 은 탐지 주기와 따로인 **프레임** 주기다. 두 클립은 60fps 라 초당 20번,
  투표 창 5 프레임 = **0.25초** (60 km/h 면 약 4 m). 품질 판별에 떨어진 프레임은 tracker 에 안 넣는다 (두 클립 모두 통과 100%).
- 측정은 3 프레임마다 한 번 캐시하고, 주기 6·15·30 프레임(0.1·0.25·0.5초)은 캐시를 솎아 다시 돌렸다.
- 눈 분류 단위: 어떤 주기·기준·투표에서든 확정이 나온 **2초 칸** (gn_a 5칸, gn_b 76칸), 칸에서 점수가 가장 낮은 판정 프레임 1장을 본다.

### 17-1. 기본 설정 (3 프레임 · 기준 1.0 · 5 중 3)

| | gn_a (흐림·이슬비, 젖음, 262초) | gn_b (맑음, 182초) |
|---|---|---|
| 차선 찾은 비율 (품질 통과 프레임 중) | **2.8%** (5,242 중 146) | 96.7% |
| 판정 프레임 (warmup 뒤) | 128 | 3,500 |
| 점수 p25 / p50 / p75 | 0.14 / 0.57 / 0.94 | 0.87 / 1.50 / 2.23 |
| 한 프레임 마모 비율 (점수 < 1.0) | 81% | 30% |
| 기준 근처 (\|점수 − 1\| < 0.1) | 9% | 7.5% |
| **확정 (마모 관측)** | 1건 (130.5초) | **58건 = 19건/분** |
| 눈 분류 (확정 기준) | fp 1 | real 2 · amb 2 · **fp 54 (93%)** |

헛것 원인 (gn_b 76칸 중 fp 72칸):
- **정지 중 횡단보도** (31~95초, 152~172초 — 신호 대기 약 85초): 횡단보도 줄무늬의 비스듬한 끝선을 차선으로 잡는다. 같은 장면에서 계속 확정 (gn_b 확정 58건 중 25건).
- 주행 중: 차로 위 글자(노인보호·50·버스전용)·화살표·마름모, 옆 차·버스 차체 모서리, 그림자 경계. 대부분 **차선은 선명한데 찾은 선분이 짧아서** `paint_len` 이 작아 점수가 낮다 — "토막 난 차선"이 아니라 "토막만 찾은 것".
- gn_a: 젖은 노면에서 차선 대비가 `rel_delta` 밑이라 거의 못 찾고, 찾은 것은 건물 세로 모서리 (화면 가운데 위). 이 클립은 화각상 노면이 화면 아래 약 25% 뿐이고 그 위는 건물이다.

real · amb: **gn_b 135~140초 (신논현역 앞 공사 복공판 구간)** — 철판 위 점선·버스차로 파란 선이 실제로 흐리다 (real 2칸 137·139초, amb 135초). 그 밖에 177초 교차로 안 유도선(amb).

### 17-2. 같은 구간 날씨 차이 (gn_a 젖음 vs gn_b 맑음)

같은 강남대로 북행이지만 **판정이 갈리는 정도가 아니라 gn_a 에서는 차선을 거의 못 찾는다** (2.8% vs 96.7%). 젖은 노면은 반사로 노면 자체가 밝아져 차선이 "양옆보다 8% 이상 밝은 줄무늬" 조건을 못 넘는다. 그래서 gn_a 에서 헛것이 많은 게 아니라 판정 자체가 거의 없다 (확정 1건, 건물 모서리).
→ 대응 제안: 젖은 노면은 **잴 수 없음으로 두는 게 맞고**, 지금 코드가 이미 그렇게 동작한다 (못 찾으면 보내지 않음). 품질 게이트에 젖은 노면 조건을 따로 넣을 필요는 이 결과로는 없다. 대신 "차선 찾은 비율이 낮은 주행은 기준(baseline)을 믿지 않는다"는 보호가 있으면 좋다 — gn_a 는 262초 동안 차선 프레임이 146개뿐이고 그중 상당수가 건물 모서리라, 그게 기준을 만든다 (warmup 20 개만 넘으면 판정 시작).

### 17-3. 기준 · 투표 · 주기를 바꿔 보기 (gn_b, fp 비율 = fp ÷ 분류된 확정)

| 주기 (프레임) | 기준 \ 투표 | 2/5 | 3/5 | 4/5 | 5/5 |
|---|---|---|---|---|---|
| 3 (0.05초, 지금) | 0.8 | 81건 · 94% | 53 · 94% | 39 · 92% | 29 · 93% |
| | **1.0** | 79 · 92% | **58 · 93%** | 43 · 93% | 31 · 94% |
| | 1.3 | 66 · 89% | 58 · 91% | 48 · 92% | 38 · 92% |
| 6 (0.1초) | 1.0 | 47 · 91% | 35 · 91% | 26 · 92% | 22 · 91% |
| 15 (0.25초) | 1.0 | 19 · 89% | 18 · 89% | 13 · 85% | 6 · 100% |
| 30 (0.5초) | 1.0 | 11 · 82% | 8 · 100% | 5 · 100% | 5 · 100% |

(전체 표 = 기준 0.8~1.3 × 투표 2/5~5/5 × 주기 3·6·15·30 은 json 의 `sweep`. gn_a 는 모든 설정에서 확정 0~1건, 전부 fp.)
- **어떤 조합도 헛것 비율이 80% 밑으로 안 내려간다.** 기준·투표를 엄격하게 하면 건수만 줄고 real 도 같이 사라진다 (real 은 많아야 4건).
- 주기를 늘리면 건수가 크게 준다 — 20Hz 에서 연속 5 프레임은 거의 같은 장면이라 투표가 독립 관측이 아니다. 하지만 비율은 그대로.
- **정지 구간을 빼도** (31~96초·152~172초, `moving_only`): 3 프레임·1.0·3/5 에서 33건 · fp 88%. 주기 15 면 12건 · fp 83%.

### 17-4. 결론 (제안 — 박겸 확인 필요)

- **기준 1.0 · 투표 3/5 · 코드 기본값은 그대로 둔다.** 바꿔서 나아지는 조합이 없다. 문제는 기준값이 아니라 차선 찾기(무엇을 차선으로 보나)와 점수의 `paint_len` (찾은 길이)이다.
- **이 두 클립으로 시연할 때 차선 마모는 서버로 보내지 않는 것을 제안한다** (블랙박스 `--lane-every 0`). 켜 두면 gn_b 한 대가 분당 약 19건의 마모 관측을 보내고 대부분 헛것이다. 가상 블랙박스 여러 대가 **같은 영상**의 변형이라 서버의 "2대 이상 · 3건 이상" 규칙(8장 3단계)도 같은 헛것을 그대로 확정한다 — 서로 독립인 관측이 아니다.
- 보여 주고 싶다면 **gn_b 135~140초 (신논현역 복공판, 영상 기준)** 한 구간만, 화면에서 "실험 기능"으로. 클립 전체를 돌린 기본 설정(3 프레임·1.0·3/5)에서 이 구간 확정은 3건 (135.8 amb · 137.1 real · 139.1 real). 다만 같은 실행에서 다른 곳 헛것도 나오므로, 보여 줄 때는 이 구간만 잘라서 다시 확인해야 한다 (warmup 이 달라진다). 발표 설명은 그대로 "낮 영상에서 보이는 페인트 손실" (8-1장).
- 11/22 마일스톤 전에 할 수 있는 AI 쪽 개선 후보 (아직 안 함, 박겸 결정): (1) 점수에서 `paint_len` 을 빼거나 "같은 주행의 보통 paint_len" 으로 나누기 — 지금은 선명한 차선도 토막만 찾으면 마모, (2) 정지 판정(프레임 차이가 작으면 판정·투표 안 함) — 시각만으로 가능, (3) 횡단보도 줄무늬(나란한 짧은 선분 여러 개) 거르기. 각각 이 도구로 다시 재면 된다.
- 한계: 정답 라벨 없이 눈으로 본 2초 칸 단위, 칸에서 가장 낮은 점수 프레임 1장만 봤다. 클립 2개 · 7.4분 · 한 도로.

### 17-5. 정현우에게 전할 것 (제안)

- **[ ] 정현우**: 시연 실행에서 차선 마모는 끄는 것을 제안 (`--lane-every 0`). 켤 경우 결과는 위 표처럼 대부분 헛것.
- 블랙박스 쪽에서 할 수 있는 보호 (정현우 담당, 결정 대기): GPS 속도가 낮을 때(예: 5 km/h 미만, 신호 대기) 차선 판정을 건너뛰기 — gn_b 확정 58건 중 25건이 정지 중. 주기를 `--lane-every 15` (60fps 에서 0.25초) 로 늘리면 건수가 1/3 로 준다 (비율은 같음).
- 서버 차선 마모 확정 규칙(8장 3단계, 2대 · 3건)은 **같은 영상 변형 차량끼리는 걸러 주지 못한다.** 시연에서 이 규칙을 "여러 차량이 확인했다"로 설명하지 않는다.

### 17-6. 개선 시도 (11/16, 결과: **통과 기준 못 넘음 → 코드 기본값 그대로, 시연에서는 끔**)

11/16 박겸 결정으로 17-4 의 후보 3개 + 차선 찾은 비율 보호를 시도했다. 통과 기준: gn_b · 주기 3 프레임에서 **확정 중 헛것 ≤ 50%**, 신논현 복공판(135~140초) real 확정 ≥ 1건, gn_a 확정 0~소수.
변형은 `ai/daboyeo_ai/eval/lane_wear_variants.py` (평가용, `lane_wear.py` 기본 동작은 그대로), 비교는 같은 도구에 `measure --variants` → `summarize` (json 의 `variants_17_6`). 눈 분류는 17-1 의 2초 칸 라벨을 그대로 쓰고, 새로 확정이 생긴 칸은 `unlabeled` 로 따로 셌다 (전 설정에서 0~10건).

| 후보 | 방법 |
|---|---|
| 정지 판정 `motion` | 노면 영역(160×90 회색의 가운데 아래)을 약 0.5초 전 프레임과 비교한 절대 차이의 **중앙값** < 3 이면 판정·투표에 안 넣음. gn_b 신호 대기(35~90초·150~175초)는 0~2, 주행 중 4~23 으로 잘 갈린다 — 영상만으로 동작 |
| 횡단보도 `crosswalk_k4/k6` | 행마다 선분이 지나는 x 를 모아, 같은 선 조각은 하나로 묶고, 간격이 차선 폭 상한 d 의 4(6)배보다 좁게 이어진 서로 다른 선이 3(4)개 이상이면 그 선분을 뺌 |
| 점수 `p25_only` · `rel_len` · `rel_len_cap` · `sqrt_rel_len` | paint_len 을 빼거나, 같은 주행의 보통 paint_len(최근 100 프레임 중앙값)으로 나눔 (그대로 / 1 로 자름 / 제곱근) |
| 보호 `found_guard` | 최근 측정 100 프레임 중 차선 찾은 비율 < 30% 면 판정 안 함 |

gn_b 결과 (주기 3 프레임, 5 중 3, fp% = fp ÷ 분류된 확정):

| 후보 | 기준 | 확정 | real · amb · fp (새 칸) | fp% | 복공판 real | gn_a 확정 |
|---|---|---|---|---|---|---|
| 지금 (base) | 1.0 | 58 | 2 · 2 · 54 (0) | 93% | 2 | 1 |
| motion | 1.0 | 40 | 2 · 2 · 36 (0) | 90% | 2 | 1 |
| crosswalk_k6 | 1.0 | 60 | 3 · 3 · 53 (1) | 90% | 3 | 1 |
| p25_only | 0.6 | 39 | 2 · 2 · 33 (2) | 89% | 2 | 1 |
| sqrt_rel_len | 0.5 | 40 | 2 · 3 · 34 (1) | 87% | 2 | 2 |
| motion + crosswalk_k6 | 1.0 | 45 | 3 · 3 · 39 (0) | 87% | 3 | 1 |
| **motion + crosswalk_k6 + sqrt_rel_len + found_guard (가장 나음)** | 0.5 | 38 | 3 · 3 · 31 (1) | **84%** | 3 | **0** |

(각 후보 기준 0.4~1.0 전체는 json. 위는 후보마다 fp% 가 가장 낮은 기준.)

- **어느 조합도 fp 80% 밑으로 못 내려간다** (기준 50% 미달). 남은 헛것을 눈으로 다시 보면 (가장 나은 조합의 확정 38건 sheet): 주행 중 노면 글자(신논현역·노인보호)·화살표·마름모, 옆 차·버스 차체 모서리, 그림자 경계 — 세 후보 어느 것도 겨냥하지 않는 종류다. 140~145초(복공판 끝, 버스 옆) 4건은 다시 보면 흐린 점선을 잡은 것도 있어 amb 로 쳐도 fp 27/38 = 71%.
- 정지 판정은 효과가 있다 (58 → 40건, 정지 중 확정 대부분 사라짐). 하지만 남는 건 주행 중 헛것.
- **횡단보도 거르기는 단독으로는 오히려 확정이 늘어난다** (58 → 60~62건): 줄무늬 선분을 빼면 찾은 길이(paint_len)가 더 짧아져 점수가 더 낮아진다. paint_len 을 점수에서 줄여야 같이 의미가 있다.
- 점수 식은 paint_len 을 빼든 나누든 fp 87~90% 범위에서 비슷하다 — 선분을 잘못 고르는 문제가 점수 식보다 크다.
- 주기를 15 프레임으로 늘리고 투표를 15/20 처럼 아주 엄격하게 하면 확정 2건(real 1 · fp 1, 50%) 이 되는 조합이 있었지만, 주기 3 기준이 아니고 2건으로는 판단할 수 없어 채택 근거로 쓰지 않았다.
- found_guard 는 gn_a 의 건물 모서리 확정을 막는다 (gn_a 1 → 0). gn_b 에는 영향 없음 (차선 찾은 비율 97%).
- 손 라벨(PR #10/#11 AUC 0.79) 영향은 재지 않았다 — 채택하지 않으므로.

**결론: 통과 못 함 → `lane_wear.py` 기본값·판정 로직은 그대로.** `LaneWearMeter.measure_debug` 를 `_segments`(선분 찾기) + `_finish`(대비·페인트 양) 로 나눈 것만 들어갔다 — 평가 도구가 선분을 더 거른 뒤 측정하려고 나눈 것이고, 두 클립 8,882 프레임 전부와 기존 테스트에서 결과가 바이트 단위로 같다.
- **시연: 차선 마모는 끈다 (`--lane-every 0`).** 보여 주려면 gn_b 135~140초(신논현역 복공판) 한 구간만 "실험 기능"으로, 발표 설명은 "낮 영상에서 보이는 페인트 손실" (8-1장).
- 다음에 할 일이 있다면 (마감 뒤 또는 시간이 남으면): 노면 글자·화살표(짧고 넓은 흰 덩어리)와 차체 모서리(차량 탐지 박스 안 선분)를 거르는 것 — 후자는 탐지 모델 박스를 쓸 수 있다. 정지 판정(`MotionProbe`)은 그대로 옮겨 쓸 수 있다.

**정현우에게 전할 것 (17-5 갱신)**: 시연에서 `--lane-every 0`. 블랙박스 쪽 GPS 속도 정지 판정 제안(17-5)은 영상만으로도 같은 효과(정지 중 확정 대부분 제거)를 확인했지만, 남는 주행 중 헛것 때문에 켜도 쓸 수 없는 상태는 같다 — 지금은 구현하지 않아도 된다.

## 18. 시연 클립 gn_a·gn_b 정답 라벨과 탐지 v3 이벤트 단위 평가 (11/16, 결과: **v3 균열 재현율 0.30 · 정밀도 0.67 (확실한 정답 기준), v2 는 균열 0건**)

> 16·17장은 정답 없이 탐지 크롭을 눈으로 본 것이라 "모델이 놓친 결함"을 셀 수 없었다. 발표용 숫자를 위해 두 클립에 12장 형식 정답을 만들고
> `eval_detector` 로 이벤트 단위 정밀도·재현율을 냈다. 정답 CSV: `experiments/gt_events/gn_a_2050-2510.csv`, `gn_b_3320-3620.csv`
> (12장·`eval_detector --gt-dir` 기본 경로). 시간·유형·메모만 있어 영상 재배포가 아니다. 프레임·크롭은 레포 밖 `data/demo_gt/`.

### 18-1. 라벨 규칙

- **만든 방법**: Claude 가 모델과 상관없이 두 클립 전 구간을 0.5초 간격 노면 띠(아래 절반)·1초 간격 전체 화면 contact sheet 로 보고, 애매한 곳은 원본 해상도로 다시 봤다.
  16·17장에서 real 로 본 곳(긴 세로 균열 120~131초, 복공판 135~140초 등)은 출발점으로만 썼다.
- **한 행 = 화면에 보이는 연속 구간 하나.** 깨끗한 노면이 1초 이상 끼면 나눈다. 시각은 클립 시작 기준 초, 결함이 판독되기 시작한 때 ~ 화면 아래로 사라진 때.
- **유형**: `pothole` `crack` `lane_wear` (계약 값) + 시설물 `guardrail` `sign` (`video_events` 가 `guardrail_damage`·`sign_damage` 로 읽어 **위치 찾기**로만 잰다 — 파손 판정 아님, PRD v0.5).
- **애매**: `memo` 를 `amb` 로 시작. 평가를 두 번 한다 — strict (amb 행을 뺀 정답, amb 에 걸린 예측은 헛것으로 셈) / lenient (amb 도 정답).
- **가드레일**: 중앙 가드레일과 차도 옆 보행자 펜스(가로대가 있는 것). 화면에 충분히 길게(폭 1/4 이상) 보이는 연속 구간. 공사용 플라스틱 방호벽은 amb.
- **표지판**: 교통 표지(규제·주의·지시·안내·도로명, 머리 위 차로 안내 전광판). **원본 기준 짧은 변 약 40 px 이상으로 커진 때부터** 화면을 벗어날 때까지. 같은 때 보이는 표지는 한 행.
  가게 간판·광고판·신호등·영상 자막(교차로 이름)은 넣지 않는다. 작거나 판독이 애매하면 amb.
- 신호 대기(정지) 중 보이는 시설물은 대기 구간 전체가 한 행 (gn_a 0~60초 정류장 펜스, gn_b 31~96초 신호등 기둥 표지).
- **두 클립 사이 같은 결함 ID 는 맞추지 않았다** (같은 강남대로지만 날씨·차로·촬영일이 달라 같은 지점인지 확인 불가). GPS 로 맞춰 본 결과는 20장. ID 는 `A_`·`B_` + 유형 글자 + 번호.

### 18-2. 정답 수

| 클립 | 균열 | 포트홀 | 차선 마모 | 가드레일 | 표지판 |
|---|---|---|---|---|---|
| gn_a (흐림·젖음, 262초) | 1 (+amb 1) | 0 (+amb 2) | 0 | 3 (+amb 1) | 2 (+amb 3) |
| gn_b (맑음, 182초) | 9 (+amb 2) | 1 | 1 (+amb 1) | 2 (+amb 1) | 7 (+amb 1) |

- gn_a 는 262초 중 약 130초가 신호 대기이고 주행 중에도 앞차(SUV)가 노면 대부분을 가린다. 젖은 노면이라 패치·물 고임과 패임을 가르기 어려워 포트홀·균열 대부분이 amb.
- gn_b 균열: 버스차로 옆 긴 세로 균열(B_C10) 외에 차로 가운데 균열 묶음(5~15초 B_C01·B_C02), 화살표 둘레(B_C03), 횡단보도 흰 줄 위 균열 2곳, 교차로 안(180초) 등.

### 18-3. 결과 (0.5초 간격 = 블랙박스 `--detect-every 30`, 정답 창 ±0.5초, 1초 안 hit 는 한 이벤트, 두 클립 7.4분 합)

v3 기준 `{pothole 0.31, crack 0.42, guardrail 0.21, sign 0.21}` (16장 확정), v2 기준 `experiments/conf_thresholds.json`.

| 유형 | 모델 | 정답 (strict) | 찾음 | 예측 이벤트 | 맞음 | 정밀도 | 재현율 | F1 | 분당 | lenient P / R |
|---|---|---|---|---|---|---|---|---|---|---|
| **crack** | **v3** | 10 | 3 | 6 | 4 | **0.67** | **0.30** | 0.41 | 0.81 | 0.67 / 0.23 |
| | v2 | 10 | 0 | 0 | 0 | — | 0.00 | — | 0 | — / 0.00 |
| pothole | v3 | 1 | 0 | 0 | 0 | — | 0.00 | — | 0 | — / 0.00 (정답 3) |
| | v2 | 1 | 0 | 1 | 0 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 / 0.00 |
| guardrail (위치) | v3 | 5 | 5 | 49 | 23 | 0.47 | 1.00 | 0.64 | 6.6 | 0.47 / 0.86 |
| | v2 | 5 | 3 | 10 | 9 | 0.90 | 0.60 | 0.72 | 1.4 | 0.90 / 0.43 |
| sign (위치) | v3 | 9 | 9 | 34 | 13 | 0.38 | 1.00 | 0.55 | 4.6 | 0.44 / 1.00 |
| | v2 | 9 | 9 | 46 | 11 | 0.24 | 1.00 | 0.39 | 6.2 | 0.35 / 1.00 |
| lane_wear | (tracker) | 1 | 1 | 8 | 1 | 0.13 | 1.00 | 0.22 | 1.1 | 0.25 / 1.00 |

숫자 전체(클립별·missed_ids)는 `experiments/demo_gt_eval.json`.

- **균열 (v3)**: 찾은 것 = B_C02 (11초), B_C08 (114~115초), B_C10 (긴 세로 균열). 놓친 것 = B_C01·B_C03·B_C06 (차로 가운데·화살표 둘레 균열), B_C04·B_C05 (횡단보도 흰 줄 위), B_C11 (교차로 안), gn_a A_C02. 헛것 2 = 4.0초 (16장 amb, 정답 B_C01 보다 1초 앞) · 140.0초 (복공판 끝).
  - 0.42 아래로 내리면 찾는 균열은 늘지만 16장의 블록 포장 헛것(0.31~0.39)이 같이 돌아온다. 기준은 그대로 둔다.
  - 이벤트 단위 맞추기는 **시각만** 본다. 115.5초 hit 는 16장 크롭에서 왼쪽 헛것이었는데 같은 때 오른쪽 균열 B_C08 창 안이라 맞음으로 셌다 (정밀도가 조금 후하게 나올 수 있다).
- **포트홀**: 두 클립에 확실한 포트홀은 B_P01 (99초 차로 가장자리 포장 깨짐) 하나뿐이고 두 모델 모두 못 잡는다. 이 클립으로 포트홀 탐지를 보여 줄 수는 없다.
- **gn_a**: 두 모델 모두 균열·포트홀 탐지 0. 확실한 결함 정답도 1개(횡단보도 흰 줄 균열)라 젖은 노면 재현율은 이 데이터로 말하기 어렵다.
- **시설물**: 재현율은 둘 다 높고, v3 는 가드레일을 더 찾는 대신(5/5) 헛것도 많다 (정류장 유리벽·건물 아래 띠 등). 표지판 헛것은 대부분 광고판·간판 (16장과 같음). 서버로 안 보내므로 판정에는 쓰지 않는다.
- **차선 마모**: 0.5초 간격이라 블랙박스 기본(`--lane-every 3`)과 다르다. 17장 결론(시연에서 끄기)과 같은 방향 — 8건 중 맞음 1 (복공판 B_L01).
- 한계: 정답은 Claude 1인이 본 것 (교차 확인 없음, 박겸 훑어보기만). 클립 2개 · 7.4분 · 한 도로. 균열 정답 10개라 재현율 0.30 의 불확실성이 크다 (3/10).

발표용 한 줄 (제안): "시연 영상(맑은 날 3분)에서 v3 는 눈으로 확인한 도로 균열 10곳 중 3곳을 찾았고, 낸 균열 경보 6건 중 4건이 실제 균열이었다 (v2 는 0곳)."

```bash
# 재현 (박겸, Mac). amb 를 뺀 정답은 awk 로 만든다
mkdir -p data/demo_gt/gt_strict && for f in experiments/gt_events/gn_*.csv; do awk -F, '$5 !~ /^amb/' "$f" > data/demo_gt/gt_strict/$(basename $f); done
echo '{"class_conf": {"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}}' > data/demo_gt/v3_conf.json
uv run python -m daboyeo_ai.eval.eval_detector data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4 \
    --model ai/models/detector_v3.onnx --every 30 --conf-json data/demo_gt/v3_conf.json \
    --gt-dir data/demo_gt/gt_strict --cache-dir data/eval_cache_v3 --out data/demo_gt/eval/v3_strict   # --gt-dir 를 빼면 lenient
```

모델마다 캐시 폴더를 따로 둔다 (`eval_cache_v3`·`eval_cache_v2`). 같은 폴더를 쓰면 모델을 바꿀 때마다 다시 돈다.

### 18-4. 정현우에게 전할 것 (제안)

- **[ ] 정현우**: 1.1 가짜 탐지기·3.7 replay 모드가 이 CSV 를 그대로 읽을 수 있다 (`experiments/gt_events/<영상 이름>.csv`, 12장 형식 그대로).
  확인 부탁: (1) 헤더 `defect_id,t_start_s,t_end_s,defect_type,memo` 와 `#` 주석 줄, (2) 시설물 유형을 `guardrail`·`sign` (탐지 클래스 이름)으로 썼다 — 12장은 둘 다 받는다,
  (3) `memo` 가 `amb` 로 시작하는 행을 가짜 탐지기에서 낼지 (제안: 내지 않기 — 확실한 정답만 서버로).
- 가짜 탐지기로 시연할 때 서버로 갈 결함 관측: gn_b 균열 9 · 포트홀 1 · (차선 마모 1, 17장 따라 끄면 0), gn_a 균열 1. 시설물은 서버로 안 보냄 (11/16 확정).
- 시각은 **클립 시작 기준 초**다. 원본 영상 시각 = 클립 시각 + **클립 실제 시작 초** (gn_a +1247.646, gn_b +1998.0, `ab_overlap.CLIP_START_S`).
  이름의 2050·3320 은 분:초(20:50·33:20)이고, 재인코딩 없이 잘라 앞 키프레임에서 시작해 그보다 2.354·2.0 초 앞이다 (`docs/plan.md` 0.8, 11/17 정정 — 이전 "+2050·+3320" 은 틀림).
  예: A_C01 61~63.5초 = 원본 1308.646~1311.146초, B_C10 120~131초 = 원본 2118.0~2129.0초 (모든 결함은 `experiments/ab_overlap.json` 의 `t0_src`·`t1_src`).

## 19. 발표용 RDD2022 포트홀·균열 예시 이미지 (11/16, 결과: **슬라이드 추천 6장 (잘 잡음 4 · 놓침 1 · 헛것 1), 이미지는 `data/presentation/rdd_pothole/` (커밋 안 함)**)

시연 클립에는 모델이 잡는 포트홀이 없어서, AI 탐지 능력은 학습에 쓰지 않은 RDD2022 **test** 이미지로 보여 준다 (포트홀 서버 흐름은 정현우의 가짜 탐지기로 모의).

재현 (추론 결과는 `<out>/preds.json` 에 캐시, Mac CPU 약 6분):

```bash
uv run python -m daboyeo_ai.eval.rdd_figs --zip data/aihub/parts/rdd2022.zip --model ai/models/detector_v3.onnx --out data/presentation/rdd_pothole
# 정해진 이미지만: --pick rdd_Czech_001437,rdd_Japan_007378,rdd_Japan_006427,rdd_India_000781,rdd_Czech_000270,rdd_India_009182 --out <다른 폴더>
```

그림: 얇은 초록 = 정답, 굵은 빨강 = 모델 포트홀, 굵은 노랑 = 모델 균열 (카드 `class_conf_draft`: 포트홀 0.31, 균열 0.42). 가드레일·표지판은 RDD 에 라벨이 없어 그리지 않는다.

숫자 (RDD test 3,526장, IoU 0.5, `experiments/rdd_figs_v3.json`):

| | mAP50 (카드) | 기준 | 정밀도 | 재현율 |
|---|---|---|---|---|
| 포트홀 | 0.425 | 0.31 | 0.693 (196/283) | 0.307 (196/638) |
| 균열 | 0.582 | 0.42 | 0.804 | 0.378 |

- 국가별 포트홀 재현율 (기준 0.31): Japan 0.37 · India 0.28 · Czech 0.17 · Norway 0.03 · US 0.07 · China_MotorBike 0.93 (14개). 블랙박스와 화각이 가장 비슷한 Czech(앞유리 안쪽) 에서도 0.17 — 멀고 작은 포트홀은 거의 놓친다 (14장과 같은 한계).
- 헛것 중 일부는 정답 라벨이 빠진 진짜 포트홀이다 (예: Japan_010072·005044). 슬라이드의 헛것 예(India_009182, 흙길 바퀴 자국 → 포트홀 0.53)는 진짜 헛것으로 골랐다.

**라이선스**: figshare 공식 기록(doi:10.6084/m9.figshare.21431547)은 **CC BY 4.0**, GitHub `sekilab/RoadDamageDetector` README 는 "이미지는 **CC BY-SA 4.0**" — 서로 다르다. 더 엄격한 CC BY-SA 4.0 조건(출처·라이선스 표기, 박스를 그려 변경했음을 표시, 이 그림을 다시 배포하면 같은 라이선스)으로 표기하면 둘 다 만족한다. 발표에 보여 주는 것 자체는 문제없음. **United_States 이미지는 Google 스트리트뷰 워터마크가 있어 쓰지 않는다.** 출처 문구:

> 이미지: RDD2022 (Arya et al., 2022, figshare doi:10.6084/m9.figshare.21431547; Geoscience Data Journal 2024), CC BY-SA 4.0 · 탐지 박스는 다보여 팀이 추가

## 20. 다른 날 두 주행(gn_a·gn_b)이 같은 결함을 확인하는가 — GPS 로 맞춰 보기 (11/17, 결과: **시연에서 "다른 날 두 차량이 같은 결함 확인"은 보여 줄 수 없음 — v3 가 gn_a(젖은 노면)에서 균열·포트홀을 하나도 못 잡아 서버 20 m 묶기에서 2대 결함 0**)

> 17장: 가상 블랙박스 여러 대는 같은 영상의 변형이라 서버의 "2대 이상"이 독립 관측이 아니다. 그래서 약 8개월 차이로 같은 강남대로 북행을 찍은
> gn_b(2022-01 맑음)와 gn_a(2022 여름 흐림·젖음)를 GPS 로 맞춰, 같은 결함이 두 클립에서 같은 위치로 잡히는지 봤다. 도구 `daboyeo_ai.eval.ab_overlap`, 숫자 `experiments/ab_overlap.json`.

### 20-1. GPS (정현우 `tools/gps`, 실행만)

- `tools/gps/anchors/gn_a.csv`·`gn_b.csv` (11/16 확정 기준점 12개씩, 원본 영상 시각)를 `ab_overlap` 이 `reconstruct` 로 읽기 전용 호출해 클립 경로를 만든다.
  클립 시작 초는 `ab_overlap.CLIP_START_S` 한 곳: **gn_a 1247.646 · gn_b 1998.0** (`fetch_log.jsonl` 실제 값, `docs/plan.md` 0.8).
  아래 명령으로 만든 CSV 와 같다 (`run --gps-csv` 로 CSV 를 대신 읽을 수 있다).
  ```bash
  uv run python -m tools.gps.reconstruct gn_a --clip-start 1247.646 --out data/videos/gn_a_2050-2510.gps.csv
  uv run python -m tools.gps.reconstruct gn_b --clip-start 1998.0   --out data/videos/gn_b_3320-3620.gps.csv
  uv run python -m tools.gps.holdout_error gn_a gn_b
  ```
  > **11/17 정정**: 처음에는 1250.25·2000 (가정)으로 계산했다. 실제 시작이 2.6·2.0 초 앞이라 같은 클립 시각의 위치가 평균 5~7 m, 최대 24.5 m 달라졌다.
  > 아래 숫자는 모두 실제 값으로 다시 낸 것이고, 바뀐 값은 `(이전 …)` 으로 적었다. 결론(2대 결함 0, GPS 오차는 원인 아님)은 그대로다.
- 덮는 구간: gn_a 클립 2.4~259.4초 (이전 0.2~256.8, 강남역 중앙 정류장 → 교보타워사거리 북쪽 횡단보도), gn_b 0.3~177.0초 (이전 ~175.0) (강남역사거리 → 교보타워사거리 남쪽 횡단보도, 35:30~35:31.5 편집 컷으로 약 185 m 빠짐).
  **두 클립이 겹치는 길은 강남역 정류장 ~ Nike 앞 약 150 m + 교보타워사거리 앞뿐**이다. gn_b 의 앞 20초(B_C01~C03, v3 탐지 5건)는 gn_a 시작 지점보다 남쪽이라 gn_a 에 없다.
- hold-out 오차 (기준점 하나 빼고, 보수적): gn_a 평균 30.3 m · 중앙값 22.5 m (멈춤·컷 경계 뺀 5개 평균 25.7 m, 최대 82 m 는 22:33.5 출발), gn_b 평균 38.7 m · 중앙값 13.6 m (경계 뺀 4개 평균 **7.6 m**, 최대 166 m 는 편집 컷).
  모든 기준점을 쓴 실제 재구성은 이보다 작다. 랜드마크로 본 실제 정합: 두 클립 모두 1차로(왼쪽 가드레일·파란 버스차로 선)를 달린 구간에서 노면 글자 "50" 이 gn_a 71.5초 ↔ gn_b 약 114초 (영상으로 본 시각).
  그 두 시각의 GPS 위치 거리는 **9.8 m** (이전 6.5 m), gn_b 114초 위치를 gn_a 가 GPS 로 지나는 시각은 72.6초 (이전 70.8초, 눈으로 본 71.5초와 1.1초 차) — **약 10 m 안에서 맞는다**.
  가드레일 정답 구간도 같은 곳에 겹친다 (B_G03 114.5~131.5초 위치를 gn_a 가 72.8~80.6초에 지남 ↔ A_G02 61.5~77.5초).

### 20-2. 정답 결함 짝 (18장 정답, 구간 가운데 시각의 GPS. 블랙박스처럼 차 위치 = 결함 위치로 둠)

결함은 실제로 차 앞 수~수십 m 에 있지만 두 클립 모두 같은 규칙이고 속도도 비슷해 대부분 상쇄된다 (`--ahead-m` 로 바꿔 볼 수 있음). 같은 유형만 짝짓는다.

| 짝 | 거리 | 반경 20/30/50 m | 눈으로 본 판정 (`data/ab_overlap/`) |
|---|---|---|---|
| A_C01 (amb, 61~63.5초) ↔ **B_C06** (98~101초) | **3.9 m** (이전 3.6) | 20 ✓ | 같은 자리 (102길 앞 횡단보도 직후, 마름모 표시 앞)는 맞다. B 는 차로 가운데 세로 균열이 또렷하지만, A 는 젖은 노면이 바로 앞 SUV·옆 세단에 가려 **같은 균열인지 확인 불가** |
| A_C01 ↔ B_C05 (횡단보도 흰 줄 균열) | 10.1 m (이전 18.3) | 20 ✓ | 다른 결함 (A 는 횡단보도를 지날 때 노면이 앞차에 거의 다 가려 흰 줄 균열이 안 보임) |
| A_C01 ↔ B_C04 (교차로 출발 횡단보도 흰 줄 균열) | 16.3 m (이전 28.2) | 20 ✓ (이전 30) | 다른 결함 (B_C05 와 같은 이유, `pair_A_C01_B_C04.jpg`) |
| A_C02 (교보 북쪽 횡단보도) ↔ B_C11 (교보 교차로 안) | 28.1 m (이전 38.0) | 30 (이전 50) | 다른 결함. gn_b 는 교차로에서 좌회전해 북쪽 횡단보도를 지나지 않음. B_C11 은 GPS 끝을 3~5초 넘어 늘인 위치 (A_C02 는 이제 GPS 안) |
| A_P01 (amb) ↔ B_P01 | 30.6 m (이전 38.2) | 50 | 다른 결함. 거리도 멀고 A_P01 은 물 고인 패치 |

- 20 m 안 같은 유형 짝 = 3개 (이전 2개), 모두 A_C01 (amb) 쪽이고 이고 눈으로 같은 결함이라고 확인되는 것은 **0개**.
- **gn_b 에서 v3 가 확실히 잡는 B_C10 (버스차로 선 옆 긴 세로 균열, 120~131초)** 은 gn_a 75~80초 (이전 72~77초)에 같은 1차로로 지나는 곳이다. gn_a 프레임에서는 젖은 노면 반사와 낮은 해상도로 균열이 보이지 않는다 —
  **보수돼 사라졌는지, 젖어서 안 보이는지 가를 수 없다** (촬영 순서는 gn_b 2022-01-16 → gn_a 2022-09-10 게시·여름, `tools/video/sources.csv`. 그래서 "gn_b 에 있고 gn_a 에 없음"이 보수 후보가 되는 방향이다).
- 같은 이유로 "8개월 사이 보수돼 사라진 결함"을 주장할 수 있는 곳은 없다. gn_a 는 주행 중에도 앞차가 10 m 안에 붙어 노면 대부분을 가린다 (18-2).

### 20-3. v3 탐지를 서버처럼 묶기 (`class_conf_draft`, 0.5초 간격 캐시 = `--detect-every 30`)

탐지 → `edge/blackbox/track_merge.TrackMerger` 로 트랙 합치기(읽기 전용 호출) → 트랙마다 최고 신뢰도 프레임 시각의 GPS 로 관측 1건 → `server/dedup.ingest` 규칙(같은 유형, 반경 안 가장 가까운 결함, 위치는 관측 평균)으로 촬영 순서대로 B(2022-01) 다음 A(2022 여름) 를 묶음.

| | gn_a 관측 | gn_b 관측 | 결함 (20 / 30 / 50 m) | 2대가 같이 본 결함 |
|---|---|---|---|---|
| 균열 | **0** | 9 (3~11초 5건, 114~129초 4건) | 6 / 4 / 3 | **0 / 0 / 0** |
| 포트홀 | 0 | 0 | 0 | 0 |

- gn_a 에서 v3 균열 신뢰도는 클립 전체 최고 0.127 (기준 0.42), 포트홀은 0.05 이상이 하나도 없다. 합치기를 끄고(`--no-merge`) 탐지마다 보내도 같다.
- 정답 위치로도 두 클립이 확실히 같이 보는 결함이 없으므로, 모델이나 기준을 바꿔도 이 두 클립으로 "독립 2대 확인"을 만들기는 어렵다.

### 20-4. 결론과 발표 (제안)

- **시연에서 "다른 날 지나간 두 차량이 같은 결함을 확인"을 실제 데이터로 보여 줄 수 없다.** 이유 (영향 큰 순): ① 젖은 노면에서 v3 가 균열·포트홀을 못 잡음 (gn_a 0건),
  ② gn_a 는 앞차가 노면을 가리고 신호 대기가 길어 정답으로도 확실한 결함이 1개뿐, ③ 두 클립이 겹치는 길이 짧음 (gn_b 앞 20초·교보 북쪽은 한쪽에만 있음), ④ 차로가 다른 구간이 있음 (A_C01/B_C06).
  GPS 오차는 원인이 아니다 (겹치는 구간 랜드마크 기준 약 10 m (9.8 m), 20 m 반경 안).
- 서버의 "2대 이상"은 지금처럼 **같은 영상 변형 여러 대 = 모의**라고 밝히고 (17장 표현), 대신 "두 날 영상을 GPS 로 맞추면 같은 지점이 약 10 m 안에서 겹친다 (노면 글자·가드레일 기준)"를 근거로 쓸 수 있다.
  (11/17 정정: 처음 쓴 "5~10 m" 는 가정한 클립 시작 초로 낸 값(6.5 m)이다. 실제 시작 초로는 9.8 m 라 **"약 10 m 안"으로 고쳐 쓴다**. 반경 20 m 안이라는 결론은 같다.)
- 꼭 실제 두 주행을 보여 주려면: 맑은 날 같은 구간 영상을 하나 더 구해 gn_b 의 B_C10 (37.50214, 127.02577 근처 — 이전 37.50221, 127.02574 에서 약 8 m, 강남대로 북행 1차로 버스차로 선 옆) 을 v3 가 잡는지 본다.

```bash
uv run python -m daboyeo_ai.eval.ab_overlap sheets             # 짝 후보 A(위)·B(아래) 프레임 → data/ab_overlap/ (커밋 안 함)
uv run python -m daboyeo_ai.eval.ab_overlap run --json experiments/ab_overlap.json   # 탐지 캐시: demo_clip run 의 data/demo_clip_eval/<gn_a|gn_b>/raw.json
```

## 21. 더 큰 탐지 모델 v4 (YOLO11s) — v3 와 같은 데이터·같은 평가로 교체 판단 (11/18, 결과: **v4 는 기준 3개 중 1개만 통과 → v3 유지 (11/18 박겸 확정). 공개 test 는 모두 오르지만 시연 클립 균열 헛것이 2 → 5건**)

v3 (YOLO11n) 는 RDD2022 test 에서 시연 기준으로 포트홀 P 0.69 / R 0.31, 균열 P 0.80 / R 0.38 (19장), 시연 클립 균열 10곳 중 3곳 (18장).
재현율을 올리려고 같은 데이터(`krid` + `rdd2022` + `road_v3`)로 **더 큰 모델 YOLO11s** 를 학습해 비교한다. **시연 기본 모델은 계속 v3** 이고, 아래 기준을 모두 넘을 때만 바꾼다.

### 21-1. 학습 (Colab, `02_train_detector.ipynb` — 설정 셀이 v4)

`VERSION="v4"`, `PARTS` 는 v3 와 같음, `BASE_MODEL="yolo11s.pt"` (COCO 사전학습 — n급 가중치는 s급에 못 옮겨서 처음부터), `EPOCHS=40`, `PATIENCE=15`, 640, batch 32.
3-1 출처별 평가는 `COMPARE_WITH=detector_v3.pt` 와 같은 라벨로 잰다. 데이터 로더는 런타임 CPU 수만큼 (`workers=min(8, CPU 수)`, T4 는 2).
여유가 있으면 `VERSION="v4m"`, `BASE_MODEL="yolo11m.pt"`, `BATCH=16` 으로 한 번 더 (아래 비교를 같은 방법으로).

예상 시간 (재지 않음 — YOLO11s 는 n 의 약 3배 계산, train 약 3.6만 장): **L4 약 2~3시간, A100 약 1.5~2시간, T4 약 5~6시간** + 풀기·평가·내보내기 20분.
세션이 끊겨도 Drive `runs/v4/weights/last.pt` 에서 이어진다 (위에서부터 다시 '모두 실행'). `is_finished` 확인은 그대로.
04 노트북(차선 분할)과 저장 폴더가 겹치지 않는다 (`runs/v4`·`/content/dataset_v4` vs `runs/lane_seg_v1`·`/content/lane_seg`).

### 21-2. 비교 (Mac, v4 가 생기면 — 전부 v3 와 같은 도구·같은 규칙)

Drive `daboyeo_data/models/detector_v4.onnx`·`detector_v4.json` → `/Users/parkgyeom/daboyeo/ai/models/`. v3 결과 파일은 덮어쓰지 않는다 (v4 는 이름에 `_v4`).

```bash
cd /Users/parkgyeom/daboyeo
# 1) 클래스별 기준 — v3 와 같은 test·규칙 (pothole·guardrail·sign 정밀도 ≥ 0.7, crack 정밀도 ≥ 0.8)
uv run python -m daboyeo_ai.eval.conf_thresholds --model ai/models/detector_v4.onnx --parts-dir data/aihub/parts --out-dir data/model_gate/conf_v4
cp data/model_gate/conf_v4/conf_thresholds.json experiments/conf_thresholds_v4.json && cp data/model_gate/conf_v4/conf_thresholds.csv experiments/conf_thresholds_v4.csv
uv run python -m daboyeo_ai.eval.model_gate conf experiments/conf_thresholds_v4.json --out experiments/class_conf_v4.json
# 2) RDD2022 test 포트홀·균열 (탐지 캐시 preds.json — v3 는 data/presentation/rdd_pothole/preds.json)
uv run python -m daboyeo_ai.eval.rdd_figs --zip data/aihub/parts/rdd2022.zip --model ai/models/detector_v4.onnx \
    --card experiments/class_conf_v4.json --out data/model_gate/rdd_v4
# 3) 시연 클립 gn_a·gn_b 이벤트 단위 (18장과 같은 strict 정답, 0.5초 간격). v3 도 같은 명령으로 (캐시가 있어 금방)
mkdir -p data/demo_gt/gt_strict && for f in experiments/gt_events/gn_*.csv; do awk -F, '$5 !~ /^amb/' "$f" > data/demo_gt/gt_strict/$(basename $f); done
uv run python -m daboyeo_ai.eval.model_gate conf experiments/conf_thresholds_v3.json --out data/model_gate/class_conf_v3.json
for v in v3 v4; do
  uv run python -m daboyeo_ai.eval.eval_detector data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4 \
      --model ai/models/detector_$v.onnx --every 30 --conf-json $([ $v = v3 ] && echo data/model_gate/class_conf_v3.json || echo experiments/class_conf_v4.json) \
      --gt-dir data/demo_gt/gt_strict --cache-dir data/eval_cache_$v --out data/model_gate/demo_$v
done
# 3-1) 사진 단위 지표 (22장) — 박스 단위와 한 파일에
uv run python -m daboyeo_ai.eval.model_gate image --preds data/model_gate/rdd_v4/preds.json --conf experiments/class_conf_v4.json --out experiments/image_level_v4.json
# 4) CPU (M4 Pro, 시연 클립 60 fps — bench_cpu_v3 와 같은 설정)
uv run python -m daboyeo_ai.eval.bench_cpu run --model ai/models/detector_v4.onnx --src data/videos/gn_b_3320-3620.mp4 \
    --res 640x360 1280x720 --threads 1 2 --repeat 3 --video-fps 60 --intervals 1 3 5 10 15 30 --out experiments/bench_cpu_v4.csv
# 5) 표 + 판정 → experiments/model_gate_v4.json
uv run python -m daboyeo_ai.eval.model_gate judge --old v3 --new v4 \
    --rdd v3=data/presentation/rdd_pothole/preds.json --rdd v4=data/model_gate/rdd_v4/preds.json \
    --demo v3=data/model_gate/demo_v3/summary.json --demo v4=data/model_gate/demo_v4/summary.json \
    --out experiments/model_gate_v4.json
```

`model_gate judge` 를 v3 끼리 돌려 확인함 (11/18): RDD 포트홀 conf 0.31 P 0.693 R 0.307, 균열 0.42 P 0.804 R 0.378 (19장과 같음),
시연 클립 균열 정답 10 · 찾음 3 · 경보 6 (진짜 4, 헛것 2) (18장과 같음), 1스레드 21.7 ms · 60 fps 10프레임마다 계산/영상 시간 0.13.

### 21-3. 교체 기준 (11/21 판단, `model_gate` 상단 상수 — 셋 다 넘어야 교체)

1. **RDD2022 test 재현율**: 포트홀·균열 각각 v3 보다 **+5%p 이상**, **같은 정밀도에서** (v4 기준을 0.01 씩 올려 RDD 정밀도가 v3 (포트홀 0.693 · 균열 0.804) 이상이 되는 가장 낮은 기준).
   v4 자기 규칙 기준(시연에 쓸 기준)의 P/R 도 같이 적는다.
2. **시연 클립 균열** (v4 자기 규칙 기준): 찾은 정답 ≥ v3 (3곳) 이고 헛것 이벤트가 v3 (2건) 보다 많지 않음. 포트홀·가드레일·표지판은 표에만 (판정 제외 — 포트홀 정답 1개, 시설물은 서버로 안 보냄).
3. **CPU**: 1스레드, 60 fps 영상을 10프레임마다 분석할 때 계산 시간 ÷ 영상 시간 < 1 (640x360·1280x720 모두).

### 21-4. 결과 (11/18, Mac M4 Pro — 21-2 명령 그대로)

`detector_v4.onnx` = YOLO11s, `yolo11s.pt` 에서 40 epoch (Colab L4), v3 와 같은 parts. ONNX 일치 10장 67/67.
집계 `experiments/model_gate_v4.json`, 기준 `experiments/conf_thresholds_v4.{json,csv}`·`experiments/class_conf_v4.json`, 사진 단위 `experiments/image_level_v4.json`, 속도 `experiments/bench_cpu_v4.{csv,json}`.
이미지·캐시는 레포 밖 `data/model_gate/` (`rdd_v4/` 그림, `demo_v3/`·`demo_v4/` 요약), `data/eval_cache_v4/`.

| 항목 | v3 (YOLO11n) | v4 (YOLO11s) |
|---|---|---|
| 공개 test mAP50 (pothole · crack · guardrail · sign · 전체) | 0.460 · 0.508 · 0.716 · 0.703 · 0.597 | **0.517 · 0.546 · 0.805 · 0.814 · 0.671** |
| RDD2022 만 mAP50 (pothole · crack) | 0.425 · 0.582 | 0.474 · 0.618 |
| 클래스별 기준 (v3 규칙: 정밀도 ≥ 0.7, 균열 ≥ 0.8) | 0.31 · 0.42 · 0.21 · 0.21 | 0.28 · 0.40 · 0.15 · 0.16 |
| RDD test 포트홀 P / R — 자기 기준 | 0.693 / 0.307 (638개 중 196) | 0.687 / 0.389 (638개 중 248) |
| 　　　　　　　　 — **같은 정밀도** (v4 0.29) | | 0.703 / **0.378** → **+7.1%p ✓** |
| RDD test 균열 P / R — 자기 기준 | 0.804 / 0.378 (4,239개 중 1,603) | 0.801 / 0.422 (4,239개 중 1,790) |
| 　　　　　　　 — **같은 정밀도** (v4 0.42) | | 0.812 / **0.403** → **+2.5%p ✗** |
| RDD test 사진 단위 포트홀 P / R (참고, 22장) | 0.905 / 0.546 | 0.857 / 0.618 |
| RDD test 사진 단위 균열 P / R (참고, 22장) | 0.961 / 0.660 | 0.961 / 0.699 |
| 시연 클립 균열 (strict 정답 10) 찾음 / 경보 (진짜 · 헛것) | 3 / 6 (4 · 2) | 4 / 10 (5 · **5**) ✗ |
| 시연 클립 포트홀 (정답 1) | 0 | 0 |
| 시연 클립 가드레일 · 표지판 위치 (판정 제외) 찾음 · 경보 진짜/헛것 | 5/5 · 23/26, 9/9 · 13/21 | 5/5 · 28/20, 9/9 · 8/32 |
| CPU 1스레드 ms (1280×720) · 60 fps 10프레임마다 계산/영상 | 21.7 · 0.13 | 43.5 · **0.26 ✓** |
| CPU 2스레드 ms | 16.5 | 30.3 |
| **판정** | — | **기준 1 ✗ (균열) · 기준 2 ✗ · 기준 3 ✓ → v3 유지** |

- **기준 1 (RDD 재현율 +5%p)**: 포트홀은 통과 (+7.1%p), 균열은 +2.5%p 로 미달. 자기 기준(0.40)이면 +4.4%p 지만 정밀도가 v3 보다 조금 낮다.
- **기준 2 (시연 클립 균열)**: 찾은 곳은 3 → 4 (B_C02 를 잃고 B_C06·B_C11 을 새로 찾음, B_C08·B_C10 은 둘 다) 이지만 헛것 경보가 2 → 5건. 기준을 올려도 안 된다:
  v4 crack 0.42 = 4곳·경보 10 (헛것 5), 0.45 = 3곳·경보 7 (헛것 3 — 그래도 v3 2건보다 많음), 0.50 = 2곳·6 (헛것 3), 0.55 = 2곳·5 (헛것 2, 찾은 곳이 v3 보다 적음). gn_a(젖은 노면)는 v4 도 균열 0.
- **기준 3 (CPU)**: 1스레드 43 ms 로 v3 의 2배지만 60 fps 영상 10프레임마다면 계산 시간이 영상 시간의 26% — 통과. 30 fps (`--detect-every 10`) 이면 13%.
- 표지판: v4 기준 0.16 에서 광고판·간판을 더 잡아 헛것 21 → 32 (서버로 안 보내므로 판정 제외, 16장과 같은 경향).
- 한계: 시연 클립 균열 정답 10개라 3 vs 4곳·헛것 2 vs 5건 차이의 불확실성이 크다. 다만 기준을 미리 정해 두었고, 헛것 증가는 기준을 바꿔도 남는다.

**결론 (11/18 박겸 확정: v3 유지): 시연은 v3 그대로. 정현우 쪽 할 일 없음** (모델 경로·기준·replay 기록 그대로).
v4 는 공개 데이터에서 모든 클래스가 오르므로 발표 "다음 단계" 근거로 쓴다 — "모델을 키우면 공개 사진 정확도가 오르지만(mAP50 0.60 → 0.67), 시연 영상의 헛경보를 줄이려면 한국 도로 영상 라벨(블록 포장 이음매·차선 등 배경)이 더 필요하다."

발표용 한 줄 (제안): "더 큰 모델(v4)은 학습에 안 쓴 공개 사진에서 포트홀 638개 중 248개(39%)를 찾아 v3(196개, 31%)보다 많았지만, 시연 영상에서는 균열 10곳 중 4곳을 찾는 대신 헛경보가 2건에서 5건으로 늘어 시연 모델은 v3 를 유지했다."

## 22. RDD2022 test 사진 단위 지표 — 박스 단위와 나란히 (11/18, 결과: **v3 포트홀 사진 재현율 0.55 · 정밀도 0.91, 균열 0.66 · 0.96. 위치 조건을 붙여도 1~3%p 만 내려가 후한 값이 아님. v4 는 오면 같은 명령으로**)

박스 단위(IoU 0.5) 재현율 0.31 은 "결함 박스 하나하나를 정확한 위치로 찾았나" 라서, 사진에 결함이 있다는 것을 알아챘는지와는 다르다.
블랙박스는 결함이 있는 장면에서 경보가 뜨는지가 중요하므로 사진 단위 값을 같이 낸다. **교체 판정(21-3)에는 쓰지 않는다** (참고·발표용).

**정의 (발표에 그대로 쓸 수 있는 문장)** — RDD2022 test, 학습에 쓰지 않은 사진, 클래스별 시연 기준 이상 경보만 셈:
- **박스 단위 (IoU 0.5)**: 경보 박스가 같은 종류 정답 박스와 50% 이상 겹치면 맞음 (정답 하나에 경보 하나). 정밀도 = 맞은 경보 ÷ 전체 경보, 재현율 = 찾은 정답 박스 ÷ 전체 정답 박스.
- **사진 단위 재현율**: 그 종류 결함이 있는 사진 중, 같은 종류 경보가 하나라도 뜬 사진의 비율 (위치 무관).
- **사진 단위 정밀도**: 그 종류 경보가 뜬 사진 중, 실제로 그 종류 결함이 있는 사진의 비율 (위치 무관).
- **겹침 조건 (참고)**: 경보 박스가 같은 종류 정답 박스와 조금이라도 겹칠 때만 '알아챔'으로 센 사진 단위 값 — 위치 무관 값이 지나치게 후하지 않은지 확인용.

```bash
uv run python -m daboyeo_ai.eval.model_gate image --preds data/presentation/rdd_pothole/preds.json \
    --conf ai/models/detector_v3.json --out experiments/image_level_v3.json     # v4 는 21-2 의 3-1)
```

**v3 (기준 pothole 0.31, crack 0.42)** — 전체 = RDD test 3,526장 (박스 단위 값이 19장과 같은 집합). US 제외 = 19장 슬라이드 규칙(스트리트뷰 워터마크)대로 United_States 462장을 뺀 3,064장.

| 클래스 | 사진 | 박스 P / R (IoU 0.5) | 사진 단위 P / R | 겹침 조건 P / R | 정답 사진 · 경보 사진 |
|---|---|---|---|---|---|
| 포트홀 | 전체 3,526 | 0.693 / 0.307 | **0.905 / 0.546** | 0.876 / 0.529 | 348 · 210 |
| | US 제외 3,064 | 0.694 / 0.312 | 0.909 / 0.562 | 0.880 / 0.545 | 336 · 208 |
| 균열 | 전체 3,526 | 0.804 / 0.378 | **0.961 / 0.660** | 0.944 / 0.648 | 1,995 · 1,369 |
| | US 제외 3,064 | 0.815 / 0.331 | 0.947 / 0.603 | 0.928 / 0.591 | 1,535 · 977 |

**v4 (기준 pothole 0.28, crack 0.40 — 21-4)**, `experiments/image_level_v4.json`:

| 클래스 | 사진 | 박스 P / R (IoU 0.5) | 사진 단위 P / R | 겹침 조건 P / R | 정답 사진 · 경보 사진 |
|---|---|---|---|---|---|
| 포트홀 | 전체 3,526 | 0.687 / 0.389 | **0.857 / 0.618** | 0.837 / 0.603 | 348 · 251 |
| | US 제외 3,064 | 0.685 / 0.394 | 0.855 / 0.634 | 0.835 / 0.619 | 336 · 249 |
| 균열 | 전체 3,526 | 0.801 / 0.422 | **0.961 / 0.699** | 0.947 / 0.688 | 1,995 · 1,450 |
| | US 제외 3,064 | 0.809 / 0.370 | 0.947 / 0.644 | 0.931 / 0.633 | 1,535 · 1,044 |

v4 는 포트홀 사진 재현율이 0.55 → 0.62 로 오르지만 사진 정밀도가 0.91 → 0.86 으로 내려간다 (헛것 사진 약 10% → 14%).

- 사진 단위 재현율이 박스 단위보다 훨씬 높다 (포트홀 0.31 → 0.55): 한 사진에 포트홀이 여러 개면 하나만 잡아도 '알아챔'이고, 박스 위치가 IoU 0.5 에 못 미쳐도 셈.
- **겹침 조건을 붙여도 1~3%p 만 내려간다** → 위치 무관 값의 대부분은 실제 결함 위에 뜬 경보다. 다만 균열 정밀도 0.96 은 RDD 균열 사진이 많아서(정답 사진 1,995 / 3,526) 아무 데나 뜬 경보도 맞을 확률이 높다는 점은 밝혀 둔다.
- 사진 단위 정밀도가 높은 것은 경보가 뜬 사진 대부분에 그 결함이 정말 있다는 뜻 — 헛것 사진 비율: 포트홀 약 10%, 균열 약 4%.
- 한계: 해외 도로 사진이다. 시연 영상(한국, 블랙박스) 값은 18장의 이벤트 단위 (균열 10곳 중 3곳) 를 쓴다.

발표용 한 줄 (제안): "학습에 안 쓴 공개 도로 사진에서, 포트홀이 있는 사진의 55%·균열이 있는 사진의 66% 에서 경보가 떴고, 경보가 뜬 사진의 91%·96% 에 실제 결함이 있었다 (박스 위치까지 맞힌 비율은 31%·38%)."

## 23. 차선 영역 분할로 차선 마모 헛것 줄이기 (11/18, 결과: **통과 기준 못 넘음 → 시연은 그대로 끔 (`--lane-every 0`). 마스크만으로 정지 중 횡단보도 확정 26 → 4건 (발표 근거)**)

> 17·17-6장 후속. 판정 방법(차선과 바로 옆 노면의 밝기 대비)은 그대로 두고, **잴 곳을 분할 모델이 찾은 차선 영역으로만 한정**해 본 실험.
> 통과 기준 (11/22 판단): gn_b 확정 중 헛것 ≤ 50% (17-6 최선 84%), 신호 대기 중 횡단보도 확정 0건, gn_a 차선 찾은 비율이 2.8% 보다 뚜렷이 높음.

### 23-1. 분할 모델 `lane_seg_v1` (YOLO11n-seg 640, 2클래스 lane · marking)

- 데이터: AI Hub 087 도로 로드마크 VS4 사진 + VL4 라벨 3,281장 (이미 Mac 에 있음). 차선(cat 381~391, 선 라벨)은 원근 폭(화면 아래 0.008·폭)을 준 다각형, 나머지 노면 표시(화살표·글자·횡단보도 등)는 marking.
  낮 1,400 · 밤 1,105 · 비 611 · 안개 165장. 주행 + 50 프레임 묶음으로 train 2,882 / val 399. 고속도로가 많아 횡단보도 11개 · 정지선 24개뿐.
  변환 `uv run python -m daboyeo_ai.data.lane_seg` → `lane_seg.zip` (Drive `parts/`), 학습 `ai/notebooks/04_train_lane_seg.ipynb` (Colab T4, 80 epoch).
- val 마스크 지표: **lane P 0.903 · R 0.787 · mAP50 0.837**, marking 0.875 · 0.693 · 0.797. 모델 카드 `ai/models/lane_seg_v1.json` (ONNX 는 Drive `models/`).
- ONNX ↔ Ultralytics 마스크 IoU 평균 0.864 (10장, 노트북의 0.9 확인선 미달): 차선 마스크가 몇 화소 폭이라 경계 1~2 화소 차이(`LaneSegmenter` 는 같은 클래스를 프로토타입 크기에서 먼저 합침)가 IoU 를 크게 깎는다. 측정은 마스크를 4 화소 넓혀 쓰므로 영향 없음 — gn 오버레이에서 차선 위치가 맞는 것을 눈으로 확인.
- **CPU (Mac, 1280×720 영상 1 프레임)**: 분할 30 ms (1 스레드) · 24 ms (2 스레드). 측정 자체는 마스크가 있으면 오히려 빨라짐 (5~7 → 3~4 ms). 블랙박스 주기 3 프레임(60fps 초당 20번)이면 분할만 초당 0.6 코어 → 탐지와 같이 돌리기엔 무겁다. 마스크를 15 프레임마다만 새로 구하면(`seg_m15`) 초당 0.12 코어.
- 쓰는 법 (실험, 기본 꺼짐 — PR #48): `LaneWearMeter().measure(frame, lane_mask=LaneSegmenter("…/lane_seg_v1.onnx").lane_mask(frame))`. 마스크를 안 주면 결과가 이전과 바이트 단위로 같다 (gn 1,000 프레임 확인).

### 23-2. 재평가 (17장과 같은 방법)

```bash
uv run python -m daboyeo_ai.eval.lane_wear_seg measure --model ai/models/lane_seg_v1.onnx \
    --clips data/videos/gn_a_2050-2510.mp4 data/videos/gn_b_3320-3620.mp4      # 약 7분
uv run python -m daboyeo_ai.eval.lane_wear_seg sheets     # data/lane_seg_eval/<clip>/sheets/ (레포 밖)
# data/lane_seg_eval/review.json 에 눈 분류를 쓰고
uv run python -m daboyeo_ai.eval.lane_wear_seg summarize --stopped gn_b_3320-3620:30-96,152-172
```

- 주기 3 프레임 · 품질 통과 프레임만 · `LaneWearTracker` 5 중 3 (17장과 같음). 정지 판정 `motion`·찾은 비율 보호 `guard`·점수 식은 17-6 의 것 그대로.
- 눈 분류: 마스크 설정 중 하나라도 확정이 나온 2초 칸 (gn_b 65칸, gn_a 3칸) 마다 마스크·선분 오버레이 1장을 새로 보고 real / amb / fp. (17장 칸 라벨은 마스크 없는 선분을 본 것이라 쓰지 않음. 그래서 아래 '마스크 없음' 행은 새 라벨로 다시 센 값이라 17장 숫자와 조금 다르다.)
- 집계 `experiments/lane_wear_seg.json`.

gn_b (맑음, 182초) — fp% = fp ÷ 분류된 확정, 정지 중 = 신호 대기 구간(30~96초 · 152~172초) 안 확정:

| 설정 | 기준 | 확정 | real · amb · fp | fp% | 정지 중 |
|---|---|---|---|---|---|
| 지금 (마스크 없음, 17장) | 1.0 | 58 | 2 · 5 · 33 (+18 미분류) | 82% (17장 93%) | **26** |
| 마스크만 `seg` | 1.0 | 44 | 2 · 3 · 39 | 89% | **4** |
| 마스크 + 정지 판정 | 1.0 | 44 | 2 · 3 · 39 | 89% | 4 |
| 마스크(5% 대비) `seg_rd05` | 1.0 | 39 | 2 · 3 · 34 | 87% | 2 |
| 점수 p25_only + 정지 + 보호, 마스크 없음 | 0.4 | 24 | 1 · 6 · 17 | 71% | 1 |
| **마스크 + p25_only + 정지 + 보호** | **0.4** | 17 | 2 · 7 · 8 | **47%** | 2 |
| 〃 | 0.5 | 15 | 2 · 5 · 8 | 53% | 2 |
| **마스크 + sqrt_rel_len + 정지 + 보호** | **0.4** | 18 | 2 · 8 · 8 | **44%** | 2 |
| 〃 | 0.5 | 24 | 3 · 6 · 15 | 63% | 2 |
| 마스크 15 프레임마다 + p25_only + 정지 + 보호 | 0.4 | 15 | 2 · 4 · 9 | 60% | 2 |

gn_a (흐림·젖음, 262초): 차선 찾은 비율 **2.8% → 1.0%** (`seg_rd05` 1.3%). 확정 0~2건, 전부 fp (보호를 켜면 0).

### 23-3. 판정

| 통과 기준 | 전 (17-6 최선) | 후 (마스크) | 판정 |
|---|---|---|---|
| gn_b 확정 중 헛것 ≤ 50% | 84% | 44~47% (기준 0.4 에서만, 확정 17~18건) · 0.5 면 53~63% | **경계 — 안정적 통과 아님** |
| 신호 대기 중 횡단보도 확정 0건 | 26건 (지금) · 정지 판정 시 8건 | 마스크만 4건 · 최선 조합 2건 | **미달** |
| gn_a 차선 찾은 비율 > 2.8% 뚜렷이 | 2.8% | 1.0~1.3% | **미달 (오히려 낮아짐)** |

**결론: 통과 못 함 → `lane_wear.py` 기본값 그대로, 시연에서 차선 마모는 끈다 (`--lane-every 0`, 17-6 과 같음). 정현우에게 연결 요청 없음.**

- **마스크는 "엉뚱한 선분"을 실제로 걷어 낸다.** 오버레이를 보면 남은 선분이 거의 다 진짜 차선 위에 있다 (노면 글자·화살표·차체 모서리 선분이 사라짐). 그래서 마스크가 있을 때만 점수 식(paint_len 빼기)이 효과를 낸다 — 같은 점수 식이 마스크 없이 71% · 마스크와 함께 44~47%.
- 그런데도 남는 헛것은 **선명한 차선을 마모로 본 것**이다. 마스크 안에서도 차선의 일부 토막만 선분으로 잡혀 대비 하위 25% 가 낮아진다. 판정을 바꾸지 않고 측정 영역만 좁히는 방법으로는 여기까지다.
- 기준 0.4 에서 50% 밑으로 내려간 것은 이 클립에 맞춰 고른 값이고 (0.5 면 다시 53~63%), 분류된 확정이 17~18건 · real 2건뿐이라 근거로 약하다.
- 정지 중 횡단보도: 마스크만으로 26 → 4건 (−85%). 남은 2~4건은 신호 대기 시작·출발 직후(31.5초 · 94.3초)에 횡단보도 줄무늬 끝선이 차선 방향과 겹쳐 마스크 안에 들어온 것 — 학습 데이터에 횡단보도가 11개뿐이라 모델이 횡단보도를 지나는 차선을 이어 그린다.
- gn_a: 분할 모델은 젖은 노면의 차선도 찾지만 (23-1 오버레이), 이 화각은 노면이 화면 아래 약 25% 뿐이고 앞차가 가려 마스크 면적이 화면의 0.1~0.5% 다. 남은 차선 토막이 측정 영역(높이 92% 까지)·길이 조건(160 화소)을 못 넘는다. 분할이 아니라 화각 문제.
- 신논현역 복공판(135~146초)의 real 2 · amb 칸은 모든 마스크 설정에서 유지된다.

### 23-4. 발표 (제안)

- "차선 분할 모델(YOLO11n-seg, AI Hub 로드마크 3,281장, 차선 mAP50 0.84)로 잴 곳을 진짜 차선으로 한정하자 **신호 대기 중 횡단보도 오탐이 26건에서 4건으로 85% 줄었다**. 하지만 주행 중에는 선명한 차선을 마모로 보는 경우가 남아 시연에서는 끈다" — 실패도 개선 시도의 근거로 쓸 수 있다.
- 슬라이드 그림: `data/lane_seg_eval/gn_b_3320-3620/sheets/` (레포 밖, YouTube 표준 라이선스 영상이라 출처 표시 필요) 의 정지 중 횡단보도 칸과 복공판 칸.
- 한계: 정답 라벨 없이 Claude 가 2초 칸 단위로 눈 분류, 클립 2개 · 7.4분 · 한 도로. 기준은 같은 클립에서 고름.

### 23-5. 정현우에게 전할 것

- 바뀌는 것 없음: **시연에서 `--lane-every 0` 유지** (17-6 그대로). 분할 모델 연결 요청은 하지 않는다.
- 참고: 다시 켜려면 분할이 프레임당 약 30 ms (CPU 1 스레드) 더 든다.

## 24. 실제 마모 등급 라벨(RMDD)로 차선 마모 판정 재기 (11/18, 결과: **밝기 대비는 닳은 차선을 가려낸다 (차선 박스 AUC 0.84) · 시연 점수 ★ 는 약함 (0.71) → 시연은 그대로 끔 (`--lane-every 0`)**)

> 17·23장 후속. 지금까지는 "눈에 보이는 마모"의 정답 라벨이 없어 손 라벨 143장(17장 전)·LTL-M 재귀반사(AUC 0.51, 밤 반사 성능이라 다른 것)로만 쟀다.
> Roboflow 조사에서 찾은 유일한 블랙박스 시점 + 마모 등급 데이터 RMDD 로 처음 실제 등급과 비교한다.
> 질문: **우리 점수가 실제로 닳은 차선과 멀쩡한 차선을 가르는가?**

### 24-1. 데이터

- Roboflow Universe `rmdd/road-marking-damage-detection` **v7 "original 1000"**, YOLOv8 내보내기 65 MB → `data/rmdd/` (git 제외, 출처는 `data/rmdd/README.md`).
  라이선스 표기 MIT, **원 출처·촬영자·등급 기준은 공개 안 됨**. 일본 도심·주택가, 차량 앞 카메라, 낮·맑음 위주. Roboflow 가 16:9 원본을 640×640 으로 늘려 저장.
- 클래스 끝자리 0 = 멀쩡 · 1 = 조금 닳음 · 2 = 많이 닳음. 박스 수 (사진 수):

| 클래스 | train 800장 | valid 100장 | test 100장 | 합계 |
|---|---|---|---|---|
| LD0 차선 멀쩡 | 815 (421) | 129 (60) | 126 (62) | **1,070** |
| LD1 차선 조금 닳음 | 460 (314) | 58 (41) | 87 (50) | **605** |
| LD2 차선 많이 닳음 | 234 (184) | 43 (35) | 42 (28) | **319** |
| BD0 / BD1 / BD2 횡단보도 | 311 / 253 / 166 | 42 / 37 / 21 | 41 / 38 / 26 | 394 / 328 / 213 |
| AD0-2 화살표 · WND0-2 글자 | 256·105·62 · 157·125·62 | 49·18·8 · 15·15·11 | 34·14·16 · 12·12·9 | 339·137·86 · 184·152·82 |

- 박스는 차선 **한 줄이 아니라 차선이 지나가는 영역**이다 (차선 한 줄의 앞쪽 토막, 때로는 두 줄).

### 24-2. 방법 (`ai/daboyeo_ai/eval/lane_wear_rmdd.py`)

```bash
uv run python -m daboyeo_ai.eval.lane_wear_rmdd --root data/rmdd --seg-model ai/models/lane_seg_v1.onnx   # 약 1분
# → experiments/lane_wear_rmdd.json, 박스·사진별 값은 data/rmdd_eval/*.csv (레포 밖)
```

- 사진을 640×360(16:9)으로 되돌린 뒤 **사진 전체를 시연과 똑같이 잰다** (`LaneWearMeter` 기본값). 박스만 잘라 재면 ROI·소실점 투표가 깨지기 때문.
  채택된 차선 선분 중 **가운데 점이 박스 안인 선분만** 모아 그 박스의 contrast · contrast_p25 · paint_len 을 낸다 (박스 하나엔 차선 토막뿐이라 길이 하한 160 화소는 박스에 적용 안 함).
  박스 안 선분이 없으면 "못 찾음". 눈으로 확인: 선분이 대부분 박스 안 실제 차선 위에 있고, 가끔 난간·횡단보도 끝선이 섞인다.
- **★ = contrast_p25 ÷ 기준 × paint_len** 의 "기준(주행 중앙값)"은 낱장 데이터라 세 가지로 근사:
  `dataset` (전체 사진 중앙값, 상수라 순위는 절대값 contrast_p25 × paint_len 과 같음) · `tracker` (파일 번호 순으로 `LaneWearTracker` 에 넣은 그 시점 기준 = 시연과 가장 가까움) · `image` (같은 사진의 차선 박스 중앙값).
- 사진 단위: 차선 박스가 있는 829장 중 LD2 가 하나라도 있으면 "닳음"(247장), 모두 LD0 이면 "멀쩡"(288장). 사진 전체 측정값(시연과 같은 값) 사용.
- AUC = 닳은 것 하나·멀쩡한 것 하나를 뽑았을 때 닳은 쪽 값이 더 낮을 확률 (0.5 무작위). 95% 구간은 사진 단위 부트스트랩 500번.
- 따로 떼어 둔 확인: 기준값을 **train 의 멀쩡한 박스 10%(또는 20%)만 마모로 오판하는 값**으로 정하고, valid+test 200장에서 센다. 못 찾은 박스는 마모 아님(시연과 같음).
- `seg` = 차선 분할 마스크(`lane_seg_v1`, 23장)를 함께 준 측정.

### 24-3. 결과

차선 박스 AUC (차선 찾은 박스만, 괄호 95% 구간). 마스크 없음 / 마스크:

| 값 | LD0 vs LD2 | LD0 vs LD1+LD2 | LD0 vs LD1 | LD1 vs LD2 |
|---|---|---|---|---|
| **contrast** (차선 대 옆 노면 밝기 대비) | **0.84 (0.81–0.87)** / 0.85 | 0.72 / 0.73 | 0.66 / 0.68 | 0.77 / 0.74 |
| **contrast_p25** (대비 하위 25%) | **0.84 (0.81–0.86)** / **0.87** | 0.70 / 0.74 | 0.63 / 0.67 | 0.79 / 0.80 |
| ★ 기준 `tracker` (시연 점수) | 0.71 (0.68–0.74) / 0.73 | 0.60 / 0.62 | 0.54 / 0.56 | 0.71 / 0.71 |
| ★ 기준 `dataset` | 0.73 / 0.75 | 0.60 / 0.63 | 0.53 / 0.56 | 0.74 / 0.72 |
| ★ 기준 `image` | 0.64 / 0.64 | 0.53 / 0.54 | 0.47 / 0.49 | 0.68 / 0.66 |
| paint_len (남은 선 길이) | 0.60 / 0.60 | 0.52 / 0.53 | 0.47 / 0.48 | 0.65 / 0.63 |

등급별 (마스크 없음): 박스에서 차선 찾은 비율 LD0 74% · LD1 82% · LD2 84%. contrast 중앙값 1.09 · 0.91 · 0.58.
시연 기준(★ < 1.0)으로 마모라 볼 비율 (차선 찾은 박스 중) LD0 28% · LD1 23% · LD2 46%.

따로 떼어 둔 200장 (valid+test), 기준은 train 에서 정함:

| 값 · 기준 | 많이 닳음 LD2 (85개) | 조금 닳음 LD1 (145개) | 멀쩡 LD0 (255개) |
|---|---|---|---|
| contrast < 0.68 (train 멀쩡 10% 오판) | **46개 (54%)** | 23개 (16%) | **27개 (11%)** |
| contrast_p25 < 0.47 (train 멀쩡 20% 오판) | 58개 (68%) | 30개 (21%) | 43개 (17%) |
| 마스크 + contrast_p25 < 0.42 (10%) | 43개 (51%) | 22개 (15%) | 19개 (7%) |
| ★ `tracker` < 0.23 (10%) | 19개 (22%) | 9개 (6%) | 21개 (8%) |

사진 단위 (사진 전체 측정 = 시연 값, LD2 있는 사진 vs 모두 LD0 인 사진):
contrast AUC 0.78 (0.74–0.81) · contrast_p25 0.72 · ★ `tracker` 0.66 · paint_len 0.52 (마스크: 0.80 · 0.80 · 0.68 · 0.51).
**지금 시연 기준(★ < 1.0) 그대로면 닳은 사진 12% · 멀쩡한 사진 11% 를 마모로 판정 → 사실상 구별 못 함** (마스크 23% · 11%).
차선 찾은 비율은 사진 단위 98~100% (맑은 낮 도심이라 gn_a 젖은 노면 2.8% 와 크게 다름).

교란 확인: 박스 높이·폭·면적·화면 위치만으로는 AUC 0.53~0.59. 박스 높이 3등분 안에서도 contrast AUC 0.80 · 0.86 · 0.87, split 별 train 0.84 · valid 0.88 · test 0.78 → 박스 크기·위치 효과가 아니다.

횡단보도(BD): 정면에서 본 횡단보도 줄은 가로라 측정기가 거의 안 잡는다 (박스에서 찾은 비율 11~28%, 마스크 5~16%). 잡힌 것은 줄 끝선 몇 개뿐(박스 45~60개)이라 AUC 는 해석하지 않는다 (★ 는 오히려 거꾸로 0.34~0.36).

### 24-4. 판정

- **측정값 자체(밝기 대비)는 실제 마모 등급과 맞는다.** 많이 닳은 차선 vs 멀쩡한 차선 AUC 0.84 (마스크 contrast_p25 0.87), 처음 보는 200장에서 많이 닳은 차선 절반 이상을 잡으면서 멀쩡한 차선은 11% 만 오판. 손 라벨 143장(0.56~0.79)·LTL-M(0.51)보다 강한 근거다.
- **조금 닳음(LD1)은 거의 못 가른다** (LD0 vs LD1 0.63~0.68). "닳았다/아니다" 두 단계로만 말할 수 있다.
- **시연 점수 ★ 는 측정값보다 약하다** (0.71 vs 0.84). 원인은 paint_len: 남은 선 길이는 마모와 거의 무관하고(0.60, 사진 단위 0.52) 박스·차로 수에 따라 흔들려 대비의 신호를 묽힌다. 시연 기준 1.0 은 이 데이터에선 사진 단위로 구별을 못 한다. 17-6·23장에서 p25_only(paint_len 빼기)가 나았던 것과 같은 방향.
- **시연은 그대로 끈다 (`--lane-every 0`).** 이 결과는 "차선이 어디인지 알 때 얼마나 닳았는지"를 잰 것이고, 시연 클립의 실패(17·23장)는 **차선을 찾는 단계**(횡단보도·글자를 차선으로 봄, 젖은 노면에서 못 찾음)였다. RMDD 는 그 단계를 시험하지 않는다 (맑은 낮, 차선 박스가 주어짐). 마감·동결(11/24) 전에 점수 식을 바꾸고 시연 클립을 다시 재 검증할 시간도 없다.
- 점수 식 변경(★ → contrast_p25 만)은 하지 않는다. 기록만 남긴다: 다음에 켠다면 RMDD train 으로 기준을 정하고 gn 클립 헛것을 다시 재는 것이 순서.

### 24-5. 발표 (제안)

- "일본 공개 데이터(RMDD)의 실제 마모 등급이 붙은 차선 1,994개로 재 보니, 우리 밝기 대비 값은 **많이 닳은 차선과 멀쩡한 차선을 84% 맞게 가렸다**(AUC 0.84). 학습·조정에 쓰지 않은 사진 200장에서 **많이 닳은 차선 85개 중 46개(54%)를 찾았고, 멀쩡한 차선 255개 중 27개(11%)만 잘못 짚었다.** 하지만 한국 주행 영상에서는 횡단보도·노면 글자를 차선으로 착각하는 문제가 남아 시연에서는 끈다."
- 짧게: "실제 마모 라벨로 검증한 측정 원리는 동작한다 → 남은 과제는 차선을 정확히 찾는 것".
- 그림: `data/rmdd/` 사진에 박스·선분을 그린 것 몇 장 (출처 "RMDD, Roboflow Universe, MIT 표기" 필수, 원 출처 불명이라 많이 쓰지 말 것).

### 24-6. 한계

- 일본 도로 1,000장, 낮·맑음 위주. 한국 차선 도색·젖은 노면·야간은 다르다.
- 박스는 차선 한 줄이 아니라 차선 영역 (두 줄이 들어 있기도). 박스 안 선분 가운데 점으로 배정해 가장자리 선분이 옆 박스로 갈 수 있다.
- 등급 0/1/2 의 기준(벗겨진 면적? 눈대중?)이 공개되지 않았고 라벨 작성자 1명인지도 모름. LD1 이 애매한 것은 라벨 쪽 문제일 수도 있다.
- 시연 판정의 주행 기준(최근 100 프레임 중앙값)을 낱장 사진 번호 순으로 흉내 냈다 — 번호가 실제 주행 순서인지는 모른다. 연속 프레임 투표(5 중 3)는 낱장이라 잴 수 없다.
- 차선 박스를 정답으로 주었으므로 "엉뚱한 곳(횡단보도·글자)을 차선으로 보는" 헛것은 이 숫자에 안 들어 있다.

### 24-7. 정현우에게 전할 것

- 바뀌는 것 없음: 시연 `--lane-every 0` 유지. 코드 기본값(`lane_wear.py`) 그대로.

### 24-8. 발표 자료 — 실시간 시연 대신 "측정 화면" (11/18 박겸 결정: 실시간은 끄고 녹화 영상 + RMDD 그림으로 보여 준다)

```bash
uv run python -m daboyeo_ai.eval.lane_wear_figs video --video data/videos/gn_b_3320-3620.mp4 \
    --start 128 --end 148 --out data/presentation/lane_wear/gn_b_128-148.mp4      # 약 1분
uv run python -m daboyeo_ai.eval.lane_wear_figs rmdd --root data/rmdd --boxes data/rmdd_eval/base_boxes.csv \
    --out data/presentation/lane_wear/rmdd_examples.jpg                           # 24-2 를 먼저 돌려 둘 것
```

결과물 (레포 밖 `data/presentation/lane_wear/`, Drive `daboyeo_data/` 에 올려 슬라이드에 넣는다):

- **`gn_b_128-148.mp4`** (20초, 1280×890, H.264, 17 MB): 강남대로 선명한 차선(128~138초) → 신논현역 복공판의 흐린 차선(139~148초).
  측정기가 찾은 차선 선을 대비(이 주행 보통 = 1.0)로 색칠 (초록 ≥ 0.8 · 노랑 0.5~0.8 · 빨강 < 0.5), 왼쪽 위 숫자, 아래 시간 그래프.
  그래프가 선명 구간 1.5~2.0 → 복공판 0.5~0.7 로 내려간다. 1,201 프레임 중 차선 찾음 1,194 · 0.8 아래 517 · 0.5 아래 229.
  **색 구간은 화면 설명용이지 판정 규칙이 아니다** (마모 신고를 내는 장면이 아님). 141초쯤 버스 광고 그림에 선 1개가 잘못 그려진다 — 질문 받으면 "그래서 실시간에서는 껐다"의 예로 쓴다.
  분할 마스크 판(`--seg-model`)도 만들어 봤지만 복공판의 흐린 차선을 마스크가 덮지 못해 그 구간이 "차선 못 찾음"이 되어 쓰지 않는다.
- **`rmdd_examples.jpg`**: 위 줄 라벨 "많이 닳음"(LD2) 3장 · 아래 줄 "멀쩡"(LD0) 3장 (valid+test, 각 등급 대비 중앙값에 가까운 큰 박스 — 골라낸 극단값 아님),
  보라 선 = 측정기가 찾은 차선, 오른쪽 막대 = 등급별 대비 중앙값 1.09 · 0.91 · 0.58.
- 출처 표기: 영상 "Seoul Walker (YouTube, 표준 라이선스)" (영상 안에 표기됨), 사진 "RMDD, Roboflow Universe (MIT 표기)".

슬라이드 한 장 구성 (제안):

- 제목: "차선 마모 — 측정 원리는 검증, 실시간 적용은 다음 과제"
- 왼쪽 영상, 오른쪽 RMDD 그림. 아래 세 줄:
  1. "차선 페인트가 옆 노면보다 얼마나 밝은지(대비)를 잰다 — 학습 없이 영상 처리만, CPU 프레임당 약 5 ms"
  2. "실제 마모 등급이 붙은 차선 1,994개(RMDD)로 확인: 많이 닳은 차선과 멀쩡한 차선을 84% 맞게 가림. 처음 보는 사진에서 많이 닳은 차선 85개 중 46개를 찾고 멀쩡한 차선 255개 중 27개만 잘못 짚음"
  3. "한국 주행 영상에서는 횡단보도·노면 글자를 차선으로 착각해 실시간 신고에서는 뺐다 (분할 모델로 횡단보도 오탐 26 → 4건까지 줄임)"

## 25. 표지판 파손 판정 다시 해 보기 — 공개 표지판 파손 사진(Roboflow) 추가 (11/18, 결과: **통과 못 함 → 표지판 파손 판정은 계속 안 씀 (9장 그대로). 공개 사진을 더해도 블랙박스 크롭 성능은 거의 그대로 (AUC 0.84 → 0.85)**)

> 9장 후속. 11/15 에 파손 판정을 뺀 이유는 KRID 파손 크롭이 139개뿐이었기 때문 (v1b test F1 0.43, 정상 오탐률 15.7%).
> Roboflow 조사(11/18)에서 찾은 공개 표지판 파손 사진을 학습에 더하면 시연 영상의 "실험 기능" 구간에 넣을 만큼 좋아지는지 본다.
> **판정은 v1b 와 똑같이 KRID val·test 로만** 한다 (블랙박스와 같은 도메인). 가드레일은 공개 파손 데이터가 사실상 없어 손대지 않는다.

### 25-1. 데이터

받은 것 (박겸 승인, 모두 CC BY 4.0, 합계 136 MB, `data/roboflow_signs/` · 출처·버전은 `data/roboflow_signs/sources.json`):

| Roboflow Universe | 버전·형식 | 사진 | 원래 클래스 (박스) |
|---|---|---|---|
| `sign-detection-x0fra/new-damaged-sign` | v2 yolov11 (640×640 늘림) | 643 | bent 44 · broken-sheet 38 · crack 23 · rust 131 · not-damaged 342 · graffiti 82 |
| `nick-g857q/damaged-road-sign-detection` | v3 yolov11 (원본 크기) | 1,677 | damaged 1,060 · healthy 778 |
| `isuru-gnkcl/smart-city-damage-sign` | v4 yolov5 (원본 크기, 대비 늘림) | 547 | bent 236 · broken sheet 69 · rust 76 · normal 408 · graffiti 56 |

- `matyworkspace/traffic-signs-damaged-healthy` 는 스톡 사진 워터마크라 안 받았다.
- **라벨 바꾸기** (`ai/daboyeo_ai/data/sign_damage.py` `RF_LABEL`): bent·broken sheet·crack·rust·damaged → **damaged**, not-damaged·normal·healthy → **normal**.
  **graffiti(낙서)는 뺐다** (138개): KRID 파손 표지판은 탈색·벗겨짐·휨이고 낙서는 없으며, 정상으로 넣기에도 표지판 면이 가려져 애매하다.
  단 nick 의 `damaged` 에는 낙서·스티커 붙은 표지판도 섞여 있어 다 걸러지지는 않는다.
- **길쭉한 박스 뺌** (195개): 세로/가로 > 2.5 또는 가로/세로 > 4. 이런 박스(기둥째 넘어진 표지판·기둥 조각, 주로 isuru `bent`)는 damaged 174 대 normal 5 라 "길쭉하면 파손"이라는 지름길이 된다. KRID 표지판 크롭은 99% 가 세로/가로 2.5 이하.
- **중복 정리**: 크롭 dHash(64비트) 해밍 거리 ≤ 4 를 같은 것으로 보고 하나만 남김 → **837개 제거**. 대부분 같은 원본 사진: nick 안에서 516, **x0fra 사진이 nick 에 다시 들어 있음 277**, isuru 연속 프레임 45. 그래서 세 출처 사이에 겹치는 사진은 남지 않는다. 중복끼리 라벨이 다른 경우 3.
- 남은 크롭: **damaged 959 · normal 1,170**. 원본 사진 단위로 15% 는 학습에 안 쓰고 같은 도메인 확인용(holdout: damaged 155 · normal 167), **train 에 damaged 804 · normal 1,003**.
- **도메인 차이 줄이기** (`degrade`): Roboflow 크롭은 가까이서 찍어 크고 선명하다 (긴 변 중앙값 224 px 이상). KRID 표지판 크롭은 긴 변 중앙값 약 90 px, 짧은 변 하위 10% 약 35 px, 흐리다.
  그래서 Roboflow 크롭마다 긴 변을 40~160 px(로그 균등)로 줄이고, 가우시안 블러 σ 0~1.2, JPEG 품질 45~90 으로 다시 압축한 사본만 쓴다 (크롭마다 고정된 난수).
- 크롭 규칙은 KRID 와 같다: `facility_crop` (pad 0.1, 긴 변 224 이하) → 학습·추론 때 `square_input` (정사각형 224).
- 묶음: `data/sign_damage_v2/sign_rf.zip` (7.7 MB, `sign_rf/{train,holdout}/{damaged,normal}/*.jpg` + `manifest.csv` + `sources.json`).

```bash
uv run python -m daboyeo_ai.data.sign_damage crops --rf-dir data/roboflow_signs --out data/sign_damage_v2 --zip data/sign_damage_v2/sign_rf.zip   # 약 10초
```

### 25-2. 방법

- 모델·학습 설정은 v1b 와 같다: YOLO11n-cls 224, 60 epoch, 조기 종료 없음, `last.pt`, 좌우 반전·scale 0.3·randaugment, seed 0.
- val·test 는 **KRID 만** (v1b 와 같은 크롭: val 파손 15·정상 150, test 파손 17·정상 170, 표지판만이면 파손 9·정상 90). Roboflow 는 train 에만.
- 데이터 조합 5가지를 **Mac 에서** 같은 코드로 학습했다 (`experiments/sign_damage_v2_local.py`, MPS, 조합 하나 10~25분). Colab v1b 와 기계·난수가 달라서 **KRID 만으로 다시 학습한 `krid_all` 이 이번 비교의 기준선**이다.

| 이름 | 시설물 | Roboflow | KRID train 정상 배수 |
|---|---|---|---|
| `krid_all` | 가드레일+표지판 | 없음 | 5 (= v1b 데이터) |
| `krid_sign` | 표지판 | 없음 | 5 |
| `rf_all` | 가드레일+표지판 | 있음 | 5 |
| `rf_sign` | 표지판 | 있음 | 5 |
| `rf_sign20` | 표지판 | 있음 | 20 (KRID 정상을 더 넣어 오탐 줄이기 시도) |

- 지표 (v1b 와 같음): 임계값 = KRID val F1 최대 → KRID test F1·P·R, test 정상 크롭 **전부**(4,848장, 표지판 3,303 · 가드레일 1,545)의 오탐률.
- 보조 지표 (`experiments/sign_damage_v2_fpr5.py`): val 파손 표지판이 9개뿐이라 F1 최대 임계값이 0.000~0.882 로 크게 흔들린다. 그래서
  임계값 없는 **AUC** (test 표지판 파손 9 · 정상 3,303)와, **KRID val 정상 표지판 전부(4,356장)의 5% 만 넘는 점수**를 임계값으로 한 test 결과도 낸다.

```bash
uv run --with ultralytics --with onnx --with onnxslim --with onnxruntime python experiments/sign_damage_v2_local.py \
    --krid-cls data/sign_damage_v2/krid/cls --rf-root data/sign_damage_v2 --work data/sign_damage_v2/runs \
    --json experiments/sign_damage_v2_local.json            # krid.zip 의 cls/ 를 data/sign_damage_v2/krid/ 에 먼저 풀 것
uv run --with onnxruntime python experiments/sign_damage_v2_fpr5.py --krid-cls data/sign_damage_v2/krid/cls \
    --work data/sign_damage_v2/runs --json experiments/sign_damage_v2_local.json
```

### 25-3. 결과 (KRID test, 표지판)

| 모델 | 임계값 (val F1 최대) | 표지판 F1 (P / R) | 파손 9개 중 | 정상 표지판 오탐률 (3,303장) | 가드레일 오탐률 | AUC | val 오탐 5% 기준: 파손 9개 중 · test 오탐 |
|---|---|---|---|---|---|---|---|
| v1b (11/15, Colab) | — | 전체 0.43 (0.32 / 0.65)¹ | — | **13.5%** | 20.5% | — | — |
| `krid_all` (기준선) | 0.189 | 0.31 (0.24 / 0.44) | 4 | 7.3% | 19.4% | 0.84 | 3 · 4.9% |
| `krid_sign` | 0.006 | 0.36 (0.25 / 0.67) | 6 | 24.0% | — | 0.84 | 4 · 4.9% |
| `rf_all` | 0.008 | 0.35 (0.23 / 0.78) | 7 | 20.2% | 42.3% | **0.86** | 4 · 4.2% |
| `rf_sign` | 0.882 | 0.27 (0.33 / 0.22) | 2 | **2.7%** | — | 0.85 | 4 · 5.4% |
| `rf_sign20` | 0.000 | **0.50** (0.37 / 0.78) | 7 | 19.4% | — | 0.83 | **5** · 4.0% |

¹ v1b 는 가드레일+표지판 전체 값 (파손 17개). 표지판만 따로 낸 F1 은 기록이 없다. 숫자는 `experiments/sign_damage_v2_local.json`.

같은 도메인(Roboflow holdout, 학습에 안 쓴 공개 사진 322장, 각 모델 임계값):

| 모델 | F1 (P / R) |
|---|---|
| `krid_all` (Roboflow 안 씀) | 0.33 (0.58 / 0.23) |
| `rf_sign` | **0.86 (0.98 / 0.77)** — 파손 155개 중 119개, 정상 167개 중 3개 오판 |
| `rf_all` · `rf_sign20` | 0.85 · 0.82 |

### 25-4. 판정 — **통과 못 함, 표지판 파손 판정은 계속 안 쓴다**

- 기준(F1 ≥ 0.6 그리고 정상 오탐률 ≤ 5%)을 **둘 다 넘은 모델은 없다.** F1 이 가장 높은 `rf_sign20`(0.50)은 정상 표지판 19% 를 의심으로 짚고, 오탐이 가장 적은 `rf_sign`(2.7%)은 파손 9개 중 2개만 잡는다.
- **공개 사진을 더해도 블랙박스 크롭 성능은 거의 그대로다.** 임계값과 상관없는 AUC 가 0.84 → 0.85~0.86, 오탐 5% 로 맞추면 파손 9개 중 3개 → 4~5개 (1~2개 차이, 파손이 9개뿐이라 우연 범위). 그때 정밀도는 2~4% — 의심 판정 25~50번 중 1번만 진짜 파손.
- 같은 도메인(공개 사진)에서는 잘 된다 (F1 0.86, 정밀도 0.98). 즉 **모델이 '파손'을 못 배운 게 아니라, 가까이서 찍은 서양 표지판 사진의 파손(녹·휨·낙서)이 블랙박스에서 멀리 흐리게 찍힌 한국 표지판의 파손(탈색·벗겨짐)으로 옮겨 가지 않는다.** 작게·흐리게 만든 사본으로도 이 차이를 못 메웠다.
- 임계값이 val 파손 9개에 크게 흔들린다 (0.000 ~ 0.882). 같은 데이터(v1b ↔ `krid_all`)를 다시 학습해도 F1 이 0.43 → 0.27(전체)로 바뀐다. 이 규모에서는 F1 0.1~0.2 차이는 의미가 없다.
- **Colab 재학습은 하지 않아도 된다.** 노트북 03 은 v2 설정(`FACILITY="sign"`, `RF_PART="sign_rf.zip"`)으로 고쳐 두었지만, 로컬 5가지가 모두 기준에 한참 못 미쳐 Colab 에서 다시 돌려도 판단은 바뀌지 않는다.
- 시연·정현우 쪽 변화 없음: 실시간 시연은 v3 탐지만, `Detection.damage` 항상 `None` (9장 그대로). `contracts/` 변경 없음.

### 25-5. 발표 (제안)

시연 영상 "실험 기능" 구간에 넣는다면 **"블랙박스에서 파손 판정이 된다"가 아니라 "해 봤고 이런 이유로 안 됐다"** 로만:

- "공개 표지판 파손 사진 2,129장(중복 837장 정리)을 더해 표지판 파손 분류기를 다시 학습했다. **공개 사진에서는 파손 155개 중 119개(77%)를 잡고 정상 167개 중 3개만 잘못 짚었지만**, 우리 블랙박스 데이터(AI Hub)에서는 **정상 표지판의 5% 를 잘못 짚는 기준으로 파손 9개 중 4~5개**만 잡았다. 가까이서 찍은 사진과 달리고 있는 차에서 멀리 본 표지판은 다르다 → 블랙박스 해상도의 한국 표지판 파손 사진이 더 필요하다."
- 짧게: "공개 사진 77% → 블랙박스 9개 중 4개. 데이터의 '보는 거리'가 다르면 옮겨 가지 않는다."
- 그림: `data/presentation/sign_damage_v2/rf_holdout_{hit,miss,false,ok_normal}.jpg` (Roboflow holdout, `rf_sign`, 각 4장, `examples.json` 에 출처·점수). **출처 표기 필수**: "Roboflow Universe — sign-detection-x0fra/new-damaged-sign, nick-g857q/damaged-road-sign-detection, isuru-gnkcl/smart-city-damage-sign (CC BY 4.0)". 일부 원본 사진은 좌우가 뒤집혀 있다 (글자가 거울 글씨) — 슬라이드에서는 그런 장은 빼는 것이 좋다.
  **AI Hub(KRID) 크롭은 공개 자료에 쓰지 않는다.**

### 25-6. 한계

- KRID 파손 표지판이 val 9 · test 9개뿐이다. 9개 중 1개가 F1 을 0.05~0.1 움직인다.
- 학습은 Colab 이 아니라 Mac(MPS)에서 했다. 기준선 `krid_all` 과 v1b 가 같은 데이터인데도 숫자가 달라 (오탐률 7.3% vs 13.5%, 표지판 기준), 조합끼리 비교는 같은 Mac 학습끼리만 한다.
- Roboflow 라벨 기준은 출처마다 다르고 공개되지 않았다. nick 의 `damaged` 에 낙서·스티커가 섞여 있고, 일부 사진은 좌우 반전·640×640 늘림이 되어 있다.
- 크롭 하나씩의 판정이다. 실제 시연은 여러 프레임·여러 차량 투표(FR-7.2)라 크롭 단위 오탐보다는 덜 나쁠 수 있지만, 정밀도 2~4% 는 투표로도 메우기 어렵다.

### 25-7. 박겸이 할 것 · 정현우에게 전할 것

- 박겸: 할 것 없음 (Colab 재학습 불필요). 발표에 넣을지만 정한다 (25-5). 다시 해 보고 싶으면 노트북 03 을 열고 Drive `daboyeo_data/parts/` 에 `sign_rf.zip` 을 올린 뒤 위에서부터 실행 (설정 셀 `VERSION="v2"` 그대로).
- 정현우: 바뀌는 것 없음.

## 26. 도로 조각 추론을 균열에만 쓰기 — `road_tiles(2, bottom=0.8)` + `tile_classes=("crack",)` (11/19, 결과: **gn_b 균열 3→7/9 · 눈으로 분명한 오탐 +0 · 1스레드 64 ms · gn_a 변화 없음 · 부산 H900 영상은 앞유리 반사 오탐으로 끔 → 기본은 끔, 반사 없는 시연 영상에서만 켜 볼 만함**)

14장의 `road_tiles(2)`(전체 + 도로 2조각)는 작은 균열을 더 찾지만 포트홀 헛것과 보닛·반사 오탐이 같이 늘었다. 그래서 조각을 도로 높이 45–80%로 줄이고(보닛·앞유리 반사 띠 제외), 조각에서는 **균열만** 받는 변형을 쟀다. 포트홀·가드레일·표지판은 전체 프레임 패스 결과만 쓴다.

### 26-1. 바뀐 코드 (기본 동작은 그대로)

- `OnnxDetector(..., tile_classes=None)`: 전체 프레임(`FULL_FRAME`)이 아닌 창에서 받을 클래스 이름. None 이면 지금과 같다. 테스트는 `ai/tests/test_onnx_detector.py::test_tile_classes_filters_only_tile_windows`.
- `eval_detector --road-tiles 2 --tiles-bottom 0.8 --tile-classes crack`. 캐시 태그에 조건이 들어가서 다른 변형과 섞이지 않는다.
- `experiments/bbcsi_v3_check.py`: 정답 없는 새 영상을 빠르게 보는 스크립트다.
  - `run`(프레임별 박스 저장, `--crop-bottom`·`--road-tiles`·`--tiles-bottom`·`--tile-classes`), `summary`, `draw`(박스 그린 프레임)
  - `compare`(base와 조각 방식 비교. 정답 CSV가 있으면 `video_events` 규칙으로 이벤트 단위 평가. 조각에서만 나온 탐지의 크롭 모음)
- 숫자: `experiments/road_tiles_crack.json`

### 26-2. 결과 (v3, 기준 0.31/0.42/0.21/0.21, 0.5초, strict 정답, 18-3과 같은 방식)

| 영상 | 지표 | base (지금) | `road_tiles(2)` (14장) | **b0.8 + 균열만** |
|---|---|---|---|---|
| gn_b (182초) | 균열 정답 찾음 | 3/9 | 6/9 | **7/9** (+B_C01·C03·C04·C11) |
| | 균열 예측 이벤트 (맞음/헛것) | 6 (4/2) | 12 (7/5) | 11 (7/4) |
| | 포트홀 찾음 / 헛것 | 0/1 · 0 | 1/1 · 2 | 0/1 · 0 |
| gn_a (262초, 젖은 노면) | 균열 찾음 / 예측 이벤트 | 0/1 · 0 | — | 0/1 · 0 (늘어난 탐지 0) |
| 부산 NSUEkDQl3m4 (307초) | 균열 탐지 프레임 | 7 | 190 | 271 |
| 속도 (M4 Pro, 1080p) | 1스레드 / 2스레드 | 21 / 17 ms | 63 / 51 ms | **64 / 51 ms** |

- **gn_b의 새 헛것 2개**는 직접 봤다. 16.5–17초는 차로 가운데 세로 균열로, 실제 균열인데 정답에서 빠진 것이다. 117초는 amb 정답 B_C09 바로 앞의 작은 자국이라 애매로 분류했다. **눈으로 분명한 오탐 증가는 0이다.** 14장 방식에서 생겼던 횡단보도 페인트·포트홀 얼룩·복공판 오탐은 사라졌다.
- 잃은 것은 B_P01 포트홀(99초)이다. 조각에서 포트홀을 받아야만 잡힌다.
- **gn_a**에서는 두 방식 모두 균열 0이다. 고정 오탐이 없고, 자막(동 이름) 영역에서도 결함 탐지가 0이다. 해는 없지만 이득도 없다.
- **부산 영상(H900 4K 블랙박스)**: 왼쪽 대시보드 반사 오탐(14장 방식 40건)은 3건으로 줄었다. 대신 앞유리 반사 가장자리의 같은 자리(1080p 기준 x 540–650, y 700–860)에서 **고정 오탐이 41프레임** 나온다. 조각에서만 나온 탐지 79묶음을 직접 본 결과는 맞음 약 12, 애매 약 10, 오탐 약 57이다. 이 영상에서는 끈다.
  - 출처: YouTube **@bbcsi** (블랙박스 수사대), NSUEkDQl3m4 "[루카스 H900 4K] 여름 흐린날 주행영상 / 부산 사상구·부산진구"
  - **표준 라이선스이므로 녹화 시연에만 쓰고, 공개 게시하지 않으며, 출처를 표기한다.** 영상·프레임은 레포 밖(`data/videos/bbcsi/`, `data/presentation/bbcsi/`)에만 둔다.
  - v3 base로만 보면 이 영상은 시연용 결함 영상으로 약하다 (균열 7프레임 중 맞음 4, 포트홀 0). 화면 아래 OSD 줄과 "ITI" 워터마크, HUD에서는 오탐이 없어서 자를 필요가 없다.

### 26-3. 켜는 방법 (기본은 끔)

```python
from daboyeo_ai.inference import OnnxDetector, road_tiles, load_class_conf
det = OnnxDetector("ai/models/detector_v3.onnx", threads=1, conf=load_class_conf("ai/models/detector_v3.json"),
                   windows=road_tiles(2, bottom=0.8), tile_classes=("crack",))
```

```bash
# 시연 영상을 바꾸면 켜기 전에 한 번 확인 (조각에서만 나온 균열 크롭을 눈으로 보고 같은 자리 반복 오탐이 없는지)
uv run python experiments/bbcsi_v3_check.py run --video <영상> --model ai/models/detector_v3.onnx --out-dir <out>/base
uv run python experiments/bbcsi_v3_check.py run --video <영상> --model ai/models/detector_v3.onnx --out-dir <out>/b08 \
    --road-tiles 2 --tiles-bottom 0.8 --tile-classes crack
uv run python experiments/bbcsi_v3_check.py compare --base <out>/base/dets.json --tiles <out>/b08/dets.json --out-dir <out>/cmp \
    [--gt experiments/gt_events/<영상 이름>.csv]
```

### 26-4. 한계

- 정답이 있는 영상은 gn_a·gn_b 두 개(7.4분, 한 도로)뿐이다. 균열 정답 9개 중 +4는 불확실성이 크다.
- 반사 오탐은 차량과 촬영 방식에 따라 달라진다. 앞유리 반사가 있는 블랙박스에서는 조각의 아래 끝을 더 올리거나 꺼야 한다.
- 속도는 Mac(M4 Pro)에서 잰 값이다. 분석 한 번에 1스레드 기준 약 3배 느려진다.

### 26-5. 박겸이 할 것 · 정현우에게 전할 것

- 박겸: 시연 영상이 gn_b로 확정되면 켤지 정한다. 발표에서는 "조각 추론으로 균열 재현율 3→7/9" 실험 장면으로 쓸 수 있다.
- **[ ] 정현우** (켜기로 할 때만):
  - 블랙박스 CLI 플래그(예: `--road-tiles 2 --tiles-bottom 0.8 --tile-classes crack`)를 `edge/blackbox`에 추가하고, 탐지기를 26-3처럼 만든다. **기본값은 끔**으로 둔다.
  - **시연 노트북에서 속도를 다시 잰다.** Mac 기준으로 분석 한 번에 1스레드 약 64 ms다.
  - 시연 영상을 바꾸면 26-3의 확인 명령을 한 번 돌린다.

## 27. 친구 실제 주행 영상(asan_a)에서 v3 확인 — 발표 근거 전용 (11/21, 결과: **진짜 포트홀 1건 (132초, 1프레임) · 앞유리 얼룩 포트홀 고정 오탐 5이벤트 · 조각 균열 진짜 5 / 애매 6 / 오탐 3 · 가드레일 0 → e2e_smoke 에 넣지 않음 (11/21 박겸 확정). 시연은 11/22 결정으로 gn_b + asan_a 0~40초 (27-3). 판정 일부는 다시 확인 중 (27-4)**)

친구가 아산 외곽의 "도로 상태가 안 좋은 곳"에서 찍은 실제 주행 영상이다. 이 영상으로 AC-1 포트홀과 "블랙박스 시점에서 본 실제 결함" 근거를 채울 수 있는지 봤다.

- **친구 영상이라 공개 게시를 금지한다.** 학술제 발표·녹화 시연에만 쓴다. 영상·크롭은 git 에 올리지 않는다.
  - 영상 `data/videos/own/asan_a.mp4`, GPS `data/videos/own/asan_a.gps.csv` (정현우가 만듦)
  - 결과 `data/presentation/asan/`: `base/`·`tiles/` 의 `dets.json`, `crops/`, `ctx/`, 덱 후보 `examples/`
- 숫자와 이벤트별 판정: `experiments/asan_v3_check.json`
- 돌린 방법: 26-3 확인 명령 그대로 (`experiments/bbcsi_v3_check.py run/compare`, 0.5초 간격). 진짜·애매·오탐은 박스 크롭을 눈으로 보고 정했다.

### 27-1. 영상 조건

| 항목 | 값 |
|---|---|
| 형식 | 1920×1080, 30fps, H.264, **162초**, OSD 없음 |
| 촬영 | **휴대폰 촬영으로 보임.** 보닛이 화면 높이의 70~78% 쯤(가운데 기준)에서 시작해 아래 22~30% 를 가리고(gn_b 는 보닛이 안 보임), 화각이 흔들리며 옆을 보는 구간이 있다 |
| 앞유리 | 55~113초에 뿌옇고(역광) 오른쪽 가운데에 어두운 물체가 비친다 (처음에는 '앞유리 얼룩'으로 적음. 대시보드가 유리에 비친 것으로 보임, 27-4) |
| 노면·속도 | 농로와 주택가 골목. 평균 17 km/h, 최대 21 km/h. 125~140초 골목은 거북등·파손이 많다 |
| GPS | `t_s,lat,lon,speed_kmh` 326행, 0~162.3초 → 영상 길이와 맞는다 |

### 27-2. 결과 (v3, 기준 0.31/0.42/0.21/0.21, 0.5초 325프레임, 1초 이내 탐지를 한 이벤트로)

| 조건 | 포트홀 이벤트 (진짜/애매/오탐) | 균열 이벤트 (진짜/애매/오탐) | 가드레일 | 표지판 프레임 | 1스레드 ms |
|---|---|---|---|---|---|
| 640 기본 | 7 (**1**/0/6) | 5 (2/1/2) | 0 | 7 | 20.3 |
| 조각 `road_tiles(2, bottom=0.8)` + 균열만 | 7 (같음) | 14 (**5**/6/3) | 0 | 7 | 60.6 |

- **진짜 포트홀은 132.0초 한 프레임**뿐이다 (보닛 바로 앞 오른쪽의 작은 포트홀, conf 0.41). 125~140초 골목에 파손이 많은데도 다른 탐지는 없다.
- **고정 오탐 — 앞유리 얼룩을 포트홀로 봄.** 포트홀 오탐 6건 중 5건이다.
  - 59~113.5초 동안 1080p 기준 x 1030–1280, y 470–700 의 거의 같은 자리에서 나온다.
  - 0.31 이상 9프레임, 0.1 이상 15프레임이다. 기본 640 에서 나오므로 조각 추론과는 관계없다.
  - 나머지 1건은 흔들려 번진 맨홀이다 (78.5초).
- **균열 (조각)**: 9.5~26초의 차로 왼쪽 긴 세로 균열을 0.5~0.8 로 연속해서 잡는다 (진짜 이벤트 3개).
  - 오탐은 노란 중앙선(42~46초)과 보닛 끝선 2건이다. 이 영상은 보닛이 화면 70~78% 높이에서 시작해 `bottom=0.8` 조각에 보닛이 조금 들어간다.
  - 수십 프레임씩 나오는 반사 오탐은 없다. 26장 기준으로는 조각을 켜도 되는 영상이다.
- **가드레일 0**: 10~18초에 가드레일이 오른쪽에 크게 보이는데도 0.1 이상 탐지가 없다. 옆에서 가까이 본 시골 가드레일이다. v3 가드레일 재현율의 한계 사례다.
- 표지판 19~23초는 파란 도로 안내판으로, 진짜다.

### 27-3. 판정 (11/21 박겸 확정)

- **e2e_smoke 에는 넣지 않는다.**
  - 진짜 포트홀이 1프레임뿐이라 `--detect-every 10` 과 변형 기기에서 매번 잡힌다는 보장이 없다.
  - 같은 영상의 앞유리 얼룩 오탐 5이벤트가 오히려 포트홀 관측을 채운다.
- **시연 영상은 gn_b 를 유지한다.** asan_a 는 162초로 짧고, 휴대폰 화각과 앞유리 얼룩 구간이 있다.
- **갱신 (11/22 박겸 결정, 정현우 #73 구현)**: 시연은 **gn_b + asan_a 0~40초, 각 3대 = 6대** (`scripts/demo.sh both`, gn_b bb-01~03 · 아산 bb-04~06).
  - 아산 차량은 블랙박스 화면에 내보내지 않는다 (번호판). e2e_smoke 에는 여전히 넣지 않는다.
  - **23~26초 균열의 결함 클립은 열지 않는다** (번호판이 읽히는 차가 나오는 구간). 신고 장면은 13.5초 근처를 쓴다.
  - 11/24 현재: 친구의 사용 동의는 **미확인**, asan 0~40초 replay 기록(`data/replay/asan_a_0-40.v3tiles.jsonl`)은 **아직 없음**, #73 시험에서 나온 아산 결함 9건(모두 균열)은 **눈으로 판정하기 전**이다.
- **발표 근거로만 쓴다.**
  - 실제 노면 세로 균열을 연속으로 잡는 장면 (13.5초)
  - 실제 포트홀 탐지 (132초)
  - 앞유리 얼룩 오탐 (95초): 한 대의 오탐을 여러 차량 관측으로 걸러야 하는 이유로 쓴다
- **AC-1 정리 문구** (11/24 정정. 전에 적은 "시연 영상에는 포트홀이 없어"는 틀림): 시연 영상 gn_b 의 확실한 포트홀 1곳(B_P01, 99초)을 모델(v3)이 놓쳐 영상 기준 AC-1 포트홀 미충족. gn_a 의 2곳(A_P01·A_P02)은 애매(amb)라 정답에서 뺐다 (18-2·18-3). 포트홀 탐지는 RDD2022 test(P 0.91/R 0.55, 22장 사진 단위 값)에서 확인했다. 실제 주행 영상 132초 1프레임(27-2)은 다시 확인하기 전에는 근거로 쓰지 않는다 (27-4).
- 덱에 쓸 때는 **가려야** 한다.
  - 23~25초와 44~48초 프레임에는 차량 번호판이 원본 해상도로 읽힌다. 오토바이 운전자도 보인다.
  - 48.5초와 78.5초에는 간판 전화번호가 보인다.
  - 13.5·95·104·132초 예시는 그대로 써도 된다.

### 27-4. 11/24 다시 확인 — 고친 숫자와 결정 대기인 판정

이 장을 쓴 세션(Claude)이 `dets.json` 과 원본 프레임을 다시 보고 찾은 것이다.

바로 고친 것 (잰 값이 틀렸거나 재지 않은 값이었다):

- 고정 오탐의 0.31 이상 프레임 수 10 → **9**.
- 보닛 비율 "30~35%", "65% 높이에서 시작" → 가운데 기준 70~78% 에서 시작 (아래 22~30%).
- "gn_b 는 약 15%" 는 **재지 않고 적은 값**이었다. gn_b 프레임(20·60·125·170초)에는 보닛이 안 보인다.

결정 대기 (27-2 표와 `experiments/asan_v3_check.json` 의 판정은 아직 그대로):

| 대상 | 지금 적힌 판정 | 다시 보니 |
|---|---|---|
| 포트홀 오탐 5이벤트 | 앞유리 얼룩 | 둘레에 대시보드 가죽 무늬가 같이 비친다 → **대시보드 반사**로 보인다. 유리를 닦아서는 없어지지 않는다. 94.5·104.0초 "보닛 끝선"도 같은 반사의 가장자리로 보인다 |
| 132.0초 포트홀 | 진짜 | 130.5~131.5초 프레임에서 이어지는 **거친 포장(벗겨진 곳)의 끝부분**이 보닛 위로 보인 것 → 포트홀이라고 확정하기 어렵다 (애매) |
| 9.5~20.5초 균열 | 진짜 (세로 균열) | 닳은 노란 중앙선과 겹치거나 바로 옆을 지나는 긴 줄. 13.5초에는 검은 갈라짐이 뚜렷하지만 16.5~17.5초 박스는 닳은 노란 페인트 위다. 포장 이음매일 수도 있다 |
| 42~46초 균열 | 오탐 (노란 중앙선) | 43.0초에는 노란 선 바로 옆에 **검은 갈라짐이 실제로 보인다** → 오탐으로 단정할 수 없다. 9.5~20.5초와 같은 기준으로 다시 봐야 한다 |
| 156초 표지판 2프레임 | 세지 않고 합침 | 확인하지 않음 (19~23초 5프레임만 봄) |

## 28. 960·1280 입력 실험 — v3 를 640 보다 큰 입력으로 (11/19, 코드 변경 없음, 결과: **640 + 조각(균열만) 유지. 1280 은 균열을 같은 수(7/9) 찾지만 헛것이 늘고 gn_b 가드레일·표지판을 거의 잃는다**)

26장의 조각 추론 대신 전체 프레임을 960·1280 으로 넣으면 어떤지 쟀다. 원 보고서는 임시 폴더와 함께 사라져서, 기획 세션이 남긴 표를 남은 결과 파일과 대조해 다시 적었다 (11/24).

- 조건: v3, 기준 0.31/0.42/0.21/0.21, 0.5초 간격, gn_b·gn_a strict 정답 (26장과 같음). ms 는 1스레드/2스레드, 박겸 Mac.
- 남은 파일 (레포 밖): `data/demo_gt/eval/res_960/`·`res_1280/` (`summary.csv`·`json`), `data/presentation/bbcsi/resolution/` (`{gn_b,gn_a,bbcsi}_{960,1280}/dets.json`, `bbcsi_*_cmp/`, `frames_*`), 모델 `ai/models/detector_v3_960.onnx`·`detector_v3_1280.onnx`.

| 조건 | gn_b 균열 /9 | gn_b 헛것 이벤트 균열/포트홀 | base 대비 눈으로 분명한 오탐 ※ | gn_b 가드레일/표지판 | gn_a 균열 헛것 · 고정 오탐 | 부산 균열 프레임 · 앞유리 고정 오탐 | ms ※ |
|---|---|---|---|---|---|---|---|
| 640 기본 | 3 | 2/0 | — | 2/2 · 7/7 | 0 · 없음 | 7 · 없음 | 21/17 |
| **640 + 조각 (0.8, 균열만)** | **7** | 4/0 | +0 | 2/2 · 7/7 | 0 · 없음 | 271 · 41프레임 | 63/49 |
| 960 전체 | 4 | 3/0 | +1 (178.5초 횡단보도 페인트) | 2/2 · 7/7 | 1 · 없음 | 130 · 없음 | 48/37 |
| 1280 전체 | 7 | 6/1 | +2 이상 (25초 포트홀 얼룩, 횡단보도 페인트 등) | 0/2 · 1/7 | 8 · **건물 외벽 239프레임** | 148 · 없음 | 94–97/66–69 |

※ 이 두 열은 남은 파일로 확인할 수 없어 원 보고서 값을 옮긴 것이다 (눈 판정과 속도. 조각의 속도는 26-2 에는 64/51 로 적혀 있다). 나머지 열은 `summary.csv`·`dets.json`·26장과 대조했다.

- **옮겨 받은 값과 파일이 다른 것**:
  - gn_a 1280 고정 오탐은 "217프레임 (122~180초 정차)"으로 전해졌지만, `gn_a_1280/dets.json` 에서는 건물 외벽 자리(박스 중심 x 250–620, y < 600) 균열 탐지가 **239프레임, 122.5~244.0초**다. 겹침(IoU ≥ 0.3)으로 묶은 가장 큰 자리는 205프레임이다. 217 을 어떻게 셌는지는 남아 있지 않다. 프레임(130·170·200·240초)에서 붉은 벽돌 건물의 외벽인 것을 확인했다.
  - 부산 960 의 "정차 중 넓은 박스 약 20프레임"은 파일에서 [449–861, 583–892] 자리 33프레임 (81.5~135초, IoU ≥ 0.5)으로 나온다.
- 부산 960·1280 에는 80~134초 (앞차 뒤 정차) 같은 자리 [1332–1462, 713–849] 탐지가 109·106프레임 있지만, 크롭을 보면 **실제 거북등 균열**이다. 그래서 앞유리 고정 오탐은 "없음"이 맞다. 프레임 수 130·148 의 대부분은 이 정차 한 곳이다.
- **판정: 640 + 조각을 유지한다.**
- 기준값을 낮춘 참고 (판정에 안 씀. gn_b 균열만, `summary.csv` 와 같음): 960 은 0.35 → 5/9 · 헛것 8, 0.30 → 6/9 · 14. 1280 은 0.35 → 7/9 · 8, 0.30 → 8/9 · 10. 조각은 0.35 → 8/9 · 5, 0.30 → 9/9 · 11.
- 한계: 1280 의 107·113초 헛것은 프레임이 0.017초 어긋나 크롭을 확인하지 못했다. 960·1280 모델은 v3 를 입력 크기만 바꿔 다시 내보낸 ONNX 다 (메타데이터: 같은 `dataset_v3`, 11/19 내보냄). 정답 영상은 gn_a·gn_b 두 개뿐이다.
