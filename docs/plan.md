# 다보여 MVP 실행 계획 (팀 전체)

| 항목 | 내용 |
|---|---|
| 기반 | [PRD.md](../PRD.md) **v0.5** (v0.4 + 시설물 파손 판정 대체안 확정, 줄 번호는 v0.4 와 같음) |
| 계획 버전 | r3.11 (2026-11-15) — r1은 PRD v0.2, r2는 v0.3 기준 |
| 완성 기한 | 2026-11-27 (금), D-14 |
| 상태 | 팀 검토 중 — 0.1(레포·`contracts/`)은 완료 |

> PRD 인용은 `PRD.md:줄번호`(v0.4 기준). `contracts/`, `server/sql/`, `docker-compose.yml` 외의 파일 경로는 아직 없는 **제안 경로**다.

---

## 0-A. PRD v0.3에서 바뀐 것과 계획에 미친 영향

| PRD 변경 | 위치 | 계획에서 바뀐 것 |
|---|---|---|
| **직접 촬영 → 공개 주행 영상(유튜브 국내 블랙박스·드라이브 영상)** | `PRD.md:372`, `PRD.md:423-438` | M0의 "주행 촬영" 작업 삭제 → **영상 선정 + GPS 경로 재구성**으로 교체 (정현우) |
| **GPS 경로 재구성** (GPX로 경로 그리기 + 기준점 시각 + 거리 비례 보간) | `PRD.md:440-447` | 새 도구 `tools/gps/` 추가, 재구성 오차를 재는 검증 단계 추가 |
| **GPS 입력 = `시각, 위도, 경도` CSV 하나** (FR-4.4) | `PRD.md:273` | 계약에 GPS CSV 형식 고정 (`t_s,lat,lon`), 실제 GPS·재구성 경로 모두 같은 파서 |
| **AI Hub 도로장애물/표면 인지 영상(수도권) 추가** — RDD2022에 한국 도로 없음 | `PRD.md:373` | 박겸: **오늘 KRID와 함께 다운로드 신청**, 국내 도로 포트홀·균열 보강용으로 데이터셋 v2에 합침. r1의 "RDD2022 한국 서브셋" 언급은 잘못이라 삭제 |
| **BDD100K(실제 GPS) 보조, Mapillary 백업** | `PRD.md:430-431` | GPS 동기화·중복 정리를 실제 GPS로 검증하는 **보조 경로(P2)** 추가 |
| **다중 차량 = 같은 도로를 지나는 서로 다른 업로더 영상 2개 이상**, 못 구하면 한 영상 + 좌표 노이즈 | `PRD.md:435`, `PRD.md:666` | `run_fleet.py`에 `--gps-noise-m` 옵션, 중복 정리 실험 설계 변경 |
| 중복 묶기 거리 참고: **20 m에서 동일 지점 인식률 90%+** 연구 | `PRD.md:714` | 시작 반경 15 m → **20 m**, 비교 10/20/30 m |
| 오픈 이슈 #13 주행 영상 출처 (11/14) | `PRD.md:707` | 9장 오픈 이슈 표에 추가 |
| 수용 기준·지표의 평가 영상 = "공개 영상(8.5)" | `PRD.md:227`, `PRD.md:579-581` | 평가용 수동 라벨을 **선정한 공개 영상**에 붙임 |
| **(v0.4) KRID = AI Hub「전국 도로시설물 영상정보 데이터」에 `정상/파손` 라벨이 있음** | `PRD.md:370` | 파손 분류 학습 데이터를 KRID 라벨로 먼저 확보 → 2.7 작업이 "수집"에서 "라벨 추출 + 학습"으로 바뀜, 11/20 결정 위험 감소 |
| **(v0.4) AI Hub 다운로드 범위·클래스 매핑 확정** (라벨 먼저, 이미지는 필요한 zip만) | `PRD.md:375-394` | 0.2에 라벨 다운로드·집계 단계 추가, 변환 스크립트에 폴리곤→박스(가드레일)·마스크→박스(균열) 추가 |

## 0-B. 이 계획이 PRD에 더하는 것

1. **`contracts/` 패키지를 첫날 만든다.** 클래스 순서·추론 결과·관측 전송·**GPS CSV** 형식을 코드로 고정해서, 11/18 인터페이스 맞춤(`PRD.md:629`)을 "import 한 줄"로 만든다.
2. **11/16에 가짜 탐지기로 걷는 골격을 먼저 띄운다.** 모델 전에 영상 → 전송 → 중복 정리 → 화면이 돌게 하고, 11/21 end-to-end(`PRD.md:630`)는 탐지기 교체만 남긴다.
3. **블랙박스 안에서 연속 프레임 탐지를 '결함 1건 = 관측 1건'으로 합친다.** 관측 차량 수·통신량 지표(`PRD.md:584-585`)가 이것에 좌우된다.
4. **재구성 GPS의 오차를 숫자로 잰다.** 기준점을 일부 빼고(hold-out) 보간 위치와 실제 위치의 거리를 측정한다. 이 값이 중복 묶기 반경(#2)의 근거가 되고 발표에서 한계로 밝힐 수치가 된다.
5. **시연 안전장치: replay 모드.** 현장에서 실시간 추론이 끊기면 미리 계산한 탐지 결과(JSONL)로 같은 흐름을 재생한다.
6. **macOS CPU 제한 방법.** `taskset`이 없으므로 ONNX Runtime 스레드 수 제한 + Docker `--cpus`로 측정한다(`PRD.md:419`).

---

## 1. 요구사항 요약

**끝까지 지킬 흐름** (`PRD.md:652`): 탐지 → 전송 → 중복 정리 → 주행 후 확인·신고 → 신고서 → 지도 반영.

| 기능 | 우선순위 | 담당 | PRD |
|---|---|---|---|
| F1 결함 탐지 (4클래스 YOLO + 차선 마모 + 시설물 파손 분류) | P0 (파손 분류 P1) | 박겸 | `PRD.md:213-231` |
| F2 결과 수집 | P0 | 정현우 | `PRD.md:235-246` |
| F3 영상 품질 판별 | P0 | 정현우 | `PRD.md:250-262` |
| F4 위치·위험도 (+ GPS CSV 입력) | P0 | 정현우 | `PRD.md:266-284` |
| F5 중복 정리·신고 상태 | P0 | 정현우 | `PRD.md:288-300` |
| F6 주행 후 확인·신고 | P0 | 정현우 | `PRD.md:304-317` |
| F7 오신고 방지 (노출 기준) | P0 | 정현우 (+박겸 임계값) | `PRD.md:321-330` |
| F8 포상 현황 | P1 | 정현우 | `PRD.md:334-345` |
| F9 관리자 지도 | P0 | 정현우 | `PRD.md:349-359` |
| 주행 영상·GPS 확보 (8.5) | P0 (선행 조건) | 정현우 (+박겸 라벨) | `PRD.md:423-447` |

---

## 2. 기술 결정 (권장안 — 팀 확인 필요)

| 결정 | 권장안 | 이유 | 관련 이슈 |
|---|---|---|---|
| 레포 형태 | **모노레포** (`ai/`, `edge/`, `server/`, `ui/`, `contracts/`, `tools/`) | 두 명·2주, 계약 패키지를 양쪽에서 import | — |
| 언어/환경 | Python 3.11, `uv` 워크스페이스 (또는 폴더별 requirements) | 전 구성요소가 Python | — |
| **주행 영상 출처** | **유튜브 국내 영상 + 경로 재구성을 메인**, BDD100K 몇 개를 실제 GPS 검증용 보조 | PRD 추천안 (`PRD.md:429-430`), 서울 포상 도로 시나리오와 맞음 | **#13** (`PRD.md:707`) |
| GPS CSV | `t_s,lat,lon` (선택 열 `speed_kmh`), `t_s` = 영상 시작 기준 초 | FR-4.4 (`PRD.md:273`) | — |
| 관측 시각 | `observed_at = trip.started_at + t_s` (가상 주행 시작 시각 기준) | 공개 영상에는 실제 촬영 시각이 없음 | — |
| 탐지 모델 | Ultralytics YOLO **n급**(필요 시 s급), 입력 640 | Colab 시간·CPU 속도 (`PRD.md:671`) | — |
| 균열 클래스 | RDD2022 D00·D10·D20 → `crack`, D40 → `pothole`. AI Hub 도로장애물 데이터도 같은 2클래스로 매핑 | 4클래스 단일 모델 유지 | #1 (`PRD.md:706`) |
| 화면 | **Streamlit 하나** (시민 화면 = 좁은 레이아웃 페이지, 관리자 = `streamlit-folium` 지도) | PRD 권장안 (`PRD.md:640`) | #11 |
| DB | PostgreSQL 16 + PostGIS, `docker-compose` | `PRD.md:159` | — |
| DB 마이그레이션 | `server/sql/001_init.sql` 하나 (Alembic 생략) | 2주, 스키마 변경 적음 | — |
| "먼저 확인한 시민" | **먼저 [신고]를 누른 시민** (`reports.defect_id UNIQUE`) | 가장 단순하고 동시성 테스트로 검증 가능 | #4 |
| 넘기기한 결함 | 같은 시민에게 다시 묻지 않음 | 시연 단순화 | #8 |
| 클립 | 탐지 전후 2초, 480p, H.264 mp4 | 통신량 절감률을 보여주기 좋음 | #10 |
| 포상 대상 도로 | 선정한 영상 경로 중 일부를 **GeoJSON 폴리곤으로 지정** → `ST_Covers` | PRD 대안 (`PRD.md:713`) | #7 |
| 중복 묶기 반경 | **시작 20 m**, 10/20/30 m 비교 후 확정. 재구성 GPS 오차 측정값(0-B 4번)을 근거로 조정 | `PRD.md:714` 참고 연구 + 재구성 오차 | #2 |
| 위험도 | 우선 유형만으로 (`PRD.md:275-281`), 크기 반영은 11/21 판단 | `PRD.md:643` | #5 |
| CPU 제한 | (a) ORT `intra_op_num_threads=1/2`, `inter_op=1` (b) Docker `--cpus=1/2` | macOS에 `taskset` 없음 | — |
| 통신량 기준값 | 두 가지를 함께 제시: (a) 내려받은 영상 파일 크기 (b) **일반 블랙박스 1080p 비트레이트(가정값, 예 10 Mbps) × 주행 시간** | 유튜브 영상은 재압축돼 원본보다 작음 → (a)만 쓰면 절감률이 과소 평가됨 | — |

---

## 3. 레포 구조 (제안)

```
daboyeo/
├── PRD.md
├── README.md                     # 실행법 (데모 명령 3줄)
├── docker-compose.yml            # postgis, server, ui
├── contracts/                    # ★ 공동 소유, 변경 시 둘 다 확인
│   └── daboyeo_contracts/
│       ├── classes.py            # CLASS_NAMES = ["pothole","crack","guardrail","sign"] (순서 고정)
│       ├── detection.py          # Detection, Detector 프로토콜
│       ├── observation.py        # 서버 전송용 ObservationIn (pydantic)
│       └── gps.py                # GPS CSV 읽기/쓰기 + 시각→좌표 보간 (FR-4.4)
├── tools/                        # 정현우 (영상·GPS 준비)
│   ├── video/
│   │   ├── sources.csv           # 영상 목록: id, URL, 업로더, 라이선스, 도로명, 날씨/시간대, 용도
│   │   └── fetch.py              # 받기 + 1080p/30fps 정규화 (ffmpeg)
│   └── gps/
│       ├── anchors/<video_id>.csv  # 기준점: t_s, lat, lon, 설명(교차로명 등)
│       ├── routes/<video_id>.gpx   # gpx.studio에서 그린 경로
│       ├── reconstruct.py        # GPX + 기준점 → 거리 비례 보간 → <video_id>.gps.csv
│       ├── holdout_error.py      # 기준점 1개씩 빼고 보간 오차(m) 측정
│       ├── bdd_to_csv.py         # (P2) BDD100K info JSON → 같은 CSV
│       └── preview_map.py        # 경로·기준점을 오프라인 HTML(SVG, data/basemap 도로선)로 확인
├── ai/                           # 박겸
│   ├── daboyeo_ai/data/          # ★ 구현됨
│   │   ├── aihub.py              # AI Hub 파일 트리·다운로드·분할 병합 (aihubshell 과 같은 API)
│   │   ├── sources.py            # 라벨 파서: KRID(폴리곤→박스, 정상/파손), 도로장애물(COCO형, 보수완료 포트홀·맨홀=배경), RDD2022(XML)
│   │   ├── yolo.py               # YOLO 탐지·분류 데이터셋 쓰기, 640 축소, 영상 구간 단위 train/val/test 분할
│   │   ├── pipeline.py           # zip 단위 변환 + 라벨 집계(count_label_zip). 파손 프레임은 stride 로 버리지 않음
│   │   ├── damage_cls.py         # ★ 구현됨 — 파손 크롭 → 시설물 종류 복원(KRID 라벨), normal 해시 줄이기 + 정사각형 PNG(subsample_cls)
│   │   └── krid_crop_types.csv.gz # 크롭 이미지 stem → 시설물 종류 코드열 (Mac 에서 라벨로 생성, 54 KB)
│   ├── notebooks/
│   │   ├── 01_data_prep.ipynb    # ★ 구현됨 — Colab: 라벨 먼저 → 집계 → 이미지 zip 하나씩 변환 → Drive parts/*.zip
│   │   ├── 02_train_detector.ipynb # ★ 구현됨 — Colab GPU: parts 풀기 → YOLO 학습(끊기면 이어서) → 평가 → ONNX + 일치 확인 → Drive models/
│   │   └── 03_train_damage_cls.ipynb # ★ 구현됨 — Colab GPU: krid.zip cls/ 풀기 → normal 줄이기 → YOLO11n-cls 학습(이어하기) → ONNX → val 최대 F1 임계값 → test 시설물별 지표 → 일치 확인 → Drive models/
│   ├── daboyeo_ai/inference/onnx_detector.py # ★ 구현됨 — OnnxDetector (Detector 계약, letterbox·NMS, 스레드 제한). Ultralytics 와 박스·신뢰도 일치 확인
│   ├── daboyeo_ai/inference/damage_classifier.py # ★ 구현됨 — DamageClassifier (카드 .json 의 클래스 순서·임계값). OnnxDetector(damage_model=...) 가 guardrail·sign 에 damage 채움
│   ├── daboyeo_ai/eval/damage_metrics.py # ★ 구현됨 — 임계값(val 최대 F1), precision·recall·F1·normal 오탐률, 시설물별
│   ├── daboyeo_ai/lane_wear.py   # ★ 구현됨 — LaneWearMeter: 하단 ROI → 양옆보다 밝은 줄무늬 → Hough 선분 → 원근·소실점 조건 → 차선/바로 옆 노면 밝기 대비 (기준값 TBD, #6)
│   ├── daboyeo_ai/eval/lane_wear_calib.py # ★ 구현됨 — 실제 프레임 contrast 분포 CSV·백분위수 (#6 기준값 보정용)
│   ├── daboyeo_ai/eval/{eval_detector,bench_cpu,interval_sweep}.py # 예정
│   └── models/                   # .gitignore — detector_vN.onnx + .json, damage_cls_vN.onnx + .json
├── edge/blackbox/                # 정현우 — 가상 블랙박스
│   ├── main.py                   # --video --gps --vehicle --detector {fake,onnx,replay} --gps-noise-m
│   ├── frame_source.py
│   ├── quality.py
│   ├── scheduler.py              # PRD 8.3 분석 주기
│   ├── track_merge.py            # 연속 프레임 → 결함 1건
│   ├── clipper.py                # ★ 구현됨 — 결함 전후 2초 480p H.264 클립 (앞 클립과 겹치면 이어 자름)
│   ├── sender.py                 # ★ 구현됨 (uploader) — 전송 + 바이트 집계(메타·사진·클립), 결함 구간 따로 저장
│   ├── fake_detector.py          # 수동 라벨 CSV를 탐지처럼 내보냄
│   └── metrics.py                # 단계별 시간 JSONL
├── server/
│   ├── sql/001_init.sql          # PRD 9장 스키마
│   ├── app/{main,db,config}.py
│   ├── app/routers/{trips,observations,defects,reports,users}.py
│   ├── app/services/{clustering,risk,exposure,reporting,reward}.py
│   ├── data/reward_roads.geojson
│   └── tests/
├── ui/{Home.py, pages/1_시민_확인.py, pages/2_관리자_지도.py}
├── scripts/{run_fleet.py, e2e_smoke.py, reset_demo.py}
├── data/                         # .gitignore — 영상·GPS CSV·클립 (용량 큼)
└── experiments/                  # 결과 CSV·그래프 (발표 원천)
    └── gt_events/<video_id>.csv  # 수동 라벨 (평가 정답). 형식 제안: docs/ai-handoff.md 12장
```

---

## 4. 인터페이스 계약 (11/14까지 확정)

PRD 14장 합의 지점(`PRD.md:614-616`)을 코드로 고정한다.

**박겸 → 정현우: 탐지기**

```python
class Detection(BaseModel):
    class_id: int                       # CLASS_NAMES 인덱스
    class_name: str                     # "pothole" | "crack" | "guardrail" | "sign"
    conf: float                         # 0~1
    bbox: tuple[float, float, float, float]   # x1,y1,x2,y2, 0~1 정규화
    damage: Literal["normal", "suspected"] | None = None   # 시설물만
    damage_conf: float | None = None

class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...
```

- ONNX: `ai/models/detector_vN.onnx`, 입력 `1×3×640×640` float32 RGB 0~1. **전처리·NMS는 `ai/daboyeo_ai/inference/onnx_detector.py` 책임** (구현됨) → 정현우는 `from daboyeo_ai.inference import OnnxDetector` 후 `OnnxDetector(path, threads=1).detect(frame)`만 호출. 출력은 v1부터 4클래스 고정(v1은 guardrail·sign 데이터만 없음)이라 v1→v2 교체 때 블랙박스 코드 변경 없음.
- 함께 두는 `detector_vN.json`: 클래스, 입력 크기, 권장 임계값, 학습 데이터 버전, val mAP.

**GPS (FR-4.4)**

```csv
t_s,lat,lon
0.0,37.55012,126.92341
1.0,37.55020,126.92355
```

`contracts/gps.py`의 `GpsTrack.from_csv(path).at(t_s) -> (lat, lon)` 하나로 재구성 경로·BDD100K·실제 블랙박스 모두 처리.

**정현우 → 박겸: 측정 로그**

```jsonl
{"frame":120,"t":4.00,"stage":"quality","ran":true,"ms":1.8,"result":"good"}
{"frame":120,"t":4.00,"stage":"detect","ran":true,"ms":95.2,"n":2}
{"frame":121,"t":4.03,"stage":"detect","ran":false}
```

**블랙박스 → 서버: 관측** — `POST /api/v1/observations` multipart (JSON 메타 + 캡처 jpg + 클립 mp4). 메타 = PRD 9장 observations 필드(`PRD.md:478-492`). 관측 1건 = `track_merge.py`가 합친 결함 1건.

---

## 5. 일정과 단계별 작업

PRD 15.1(`PRD.md:626-634`)을 따르고, 각 작업에 **완료 확인 방법**을 붙였다.

### M0 준비 — 11/13(금) ~ 11/15(일)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 0.1 | 공동 | 레포 생성, 폴더 구조, `contracts/` (GPS CSV 포함), README | 두 사람 모두 `import daboyeo_contracts` 성공 |
| 0.2 | 박겸 | **오늘: AI Hub「전국 도로시설물 영상정보」+「도로장애물/표면 인지 영상(수도권)」 다운로드 신청** (`PRD.md:375-394`) | 신청 완료 화면 |
| 0.2b | 박겸 | 승인 후 **라벨만 먼저** 받기 (도로시설물 `TL_*`·`VL_*` 약 440 MB, 도로장애물 `Annotations` 약 1.2 GB) → zip별 포트홀 수, 가드레일·표지판 수, **파손 객체 수** 집계 → 받을 이미지 zip 결정. **`ai/notebooks/01_data_prep.ipynb` 2~3단계** (전부 Colab 에서, 노트북 PC 에는 받지 않음) | Drive `label_counts_*.csv`, 받을 zip 목록 |
| 0.3 | 박겸 | RDD2022 다운로드·변환 — 같은 노트북 5단계 (Figshare, 신청 불필요) | Drive `parts/rdd2022.zip`, 클래스별 박스 수 |
| 0.4 | 박겸 | 데이터셋 v1 (RDD2022, 2클래스) — 노트북 6단계로 `data.yaml` 생성 | Colab에서 1 epoch 실행 |
| 0.5 | 정현우 | **영상 후보 찾기** — 기준 `PRD.md:433-438`. 목표: 같은 도로 지나는 서로 다른 업로더 영상 **2개 이상** + 야간/우천 1~2개. `sources.csv`에 URL·업로더·라이선스·도로명 기록 | `sources.csv` 5행 이상, 이슈 #13 확정 (11/14) |
| 0.6 | 정현우 | `fetch.py`로 받기 + 1080p/30fps 정규화. 라이선스 확인(CC 우선, 아니면 업로더 허락), 영상은 레포에 올리지 않음 | `data/videos/`에 파일, 길이·fps 로그 |
| 0.7 | 정현우 | **GPS 재구성** (`PRD.md:440-445`): gpx.studio로 경로 → 기준점 영상당 **8개 이상**(교차로·육교·표지판) → `reconstruct.py` | `<video_id>.gps.csv` 생성, `preview_map.py`로 경로가 도로 위에 있음 |
| 0.8 | 정현우 | `holdout_error.py` — 기준점을 하나씩 빼고 보간 위치 오차(m) 측정 | 영상별 평균·최대 오차 표 (`experiments/gps_error.csv`) |
| 0.9 | 정현우 | `docker-compose.yml` + `001_init.sql` — **완료** (골격 커밋에 포함) | `psql`로 테이블·PostGIS 확인 |
| 0.10 | 공동 | 선정 영상에 **수동 라벨** `gt_events/<video_id>.csv` (시각·유형). 같은 도로 영상끼리는 **같은 결함에 같은 ID** | 결함 20건 이상, 두 영상에 공통으로 보이는 결함 5건 이상 |

> 영상·GPS가 모든 작업의 선행 조건이다 (`PRD.md:669`). 11/15까지 "같은 도로 2개"를 못 구하면 **한 영상 + 좌표 노이즈** 대체안(`PRD.md:666`)으로 확정하고 넘어간다.

### M1 골격 — 11/16(월) ~ 11/18(수)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 1.1 | 정현우 | **걷는 골격 (11/16 목표)**: `main.py --detector fake` → 업로드 → DB 저장. 가짜 탐지기는 `gt_events` CSV를 탐지처럼 내보냄 | 영상 1개 실행 후 `observations` 행 = 라벨 수 |
| 1.2 | 정현우 | `/trips`, `/trips/{id}/end`, 차량별 구분 (FR-2.2), `--gps-noise-m` | `run_fleet.py --n 3` 후 차량별 행 수 일치 |
| 1.3 | 정현우 | `clipper.py`, 전송 바이트 집계 (기준값 2가지, 2장 표). **11/16 구현**: `edge/blackbox/clipper.py`(전후 2초·480p·H.264 30fps, ffmpeg libx264 CRF 28, 없으면 OpenCV), `sender.py` 가 메타·사진·클립 바이트를 따로 세고 `--keep-dir`(기본 `server_data/blackbox/<기기>/<주행>/`)에 남김, 서버가 `clip` 을 받아 `observations.clip_path·clip_bytes` 저장 → `/api/v1/stats` 합계에 포함. 기준값 2가지는 `experiments/bandwidth.py` 가 계산 (11장) | 로그에 원본 크기 vs 전송량 |
| 1.4 | 박겸 | 탐지 v1 학습 (RDD 2클래스) → 데이터 도착 시 v2 (+AI Hub 도로장애물, +KRID 4클래스). **`02_train_detector.ipynb`** (`VERSION`, `PARTS` 만 바꿈) | val/test 클래스별 mAP@0.5 표, Drive `models/detector_vN.json` |
| 1.5 | 박겸 | ONNX 변환 + `OnnxDetector` — 같은 노트북 4단계 | val 10장에서 `OnnxDetector` 와 Ultralytics(같은 ONNX) 박스 일치 ≥ 95% |
| 1.6 | 공동 | **11/18 인터페이스 맞춤**: `--detector onnx`로 선정 영상 실행 | 탐지 박스 오버레이가 보임 |

**11/18 Go/No-go** (`PRD.md:641`): ONNX가 안 돌면 PyTorch로 같은 `Detector` 프로토콜을 구현해 연결, ONNX는 속도 측정용.

### M2 통합 — 11/19(목) ~ 11/21(토)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 2.1 | 정현우 | `quality.py` (밝기 평균, 라플라시안 분산) | 야간/우천 영상 구간에서 `detect ran:false` (F3) |
| 2.2 | 정현우 | `scheduler.py` — 품질 매 프레임, 탐지 N프레임, 차선 M프레임. **구현됨 (r3.17)**: `run.py --quality-every 1 --detect-every N --lane-every M`, 탐지·차선 프레임은 늘 품질 먼저, 끝날 때 `단계별 실행 품질 a/n (%, ms) · 탐지 … · 차선 …` 출력, `drive()` 가 `stages` 로 돌려줌 | 로그상 단계별 실행 비율 |
| 2.3 | 정현우 | `track_merge.py` — 같은 클래스·IoU>0.3이 T초 안에 이어지면 1건. **구현됨 (r3.17)**: IoU>0.3 **또는 움직임 문**(새 박스 중심이 앞 박스에서 가로 ±2.5·S, 세로 −0.5·S~+2.5·S, S=긴 변), T=`--merge-gap` 1초, 트랙 최대 3초, 끝나면 가장 신뢰도 높은 프레임 1장 전송, 주행 끝에 모두 전송. 차선 마모는 `LaneWearTracker` 그대로. `--no-merge` 로 비교 | 같은 포트홀 → 관측 1건 |
| 2.4 | 정현우 | `clustering.py` — `ST_DWithin(20m)` 같은 유형에 붙이거나 신규, `vehicle_count`·대표 좌표 갱신. 한 트랜잭션 + `FOR UPDATE` | 테스트: 3대 같은 지점 → 결함 1건, vehicle_count=3 |
| 2.5 | 정현우 | `risk.py` | 모든 결함에 risk_level (F4) |
| 2.6 | 박겸 | `lane_wear.py` — 하단 ROI → 흰/노랑 마스크 → 차선 대 노면 밝기 대비 | 선정 영상 수동 라벨 대비 정확도 |
| 2.7 | 박겸 | Mac `prepare_local` 이 만든 **KRID 파손 크롭**(`krid.zip` 의 `cls/{train,val,test}/{normal,damaged}`)으로 → `03_train_damage_cls.ipynb` 에서 가드레일+표지판 하나의 이진 분류(YOLO11n-cls, 224). normal 은 train 5배·val/test 10배로 줄임, 임계값은 val 최대 F1. 파손이 부족할 때만 공개 영상 프레임·공개 이미지로 보충 (`PRD.md:371`). 크롭 수(파손): train 107 / val 15 / test 17 — 가드레일 55·표지판 84. **결과(11/15, `damage_cls_v1b`, 60 epoch): test F1 0.43 (P 0.32 / R 0.65), test 정상 크롭 4,848장 오탐률 15.7% (가드레일 20.5%·표지판 13.5%), val 최대 F1 0.275 → No-go, 대체안 확정** | test precision·recall·F1·normal 오탐률 (시설물별), `damage_cls_v1.json` |
| 2.8 | 공동 | **11/21 첫 end-to-end** — `e2e_smoke.py` | 영상 2개(또는 노이즈 3대) → DB에서 공통 결함이 묶임, 라벨 대비 결함 수 출력 |

**11/20 Go/No-go** (`PRD.md:642`): **11/15 결정 — 대체안(위치 탐지까지만) 확정.** 크롭 수 기준(파손 139장)은 넘었지만 학습 결과가 기준(test 파손 F1 ≥ 0.6, 정상 오탐률 ≤ 5%)에 크게 못 미침. 코드(`OnnxDetector(damage_model=...)`)는 끈 채로 남겨 두고, `Detection.damage` 는 항상 `None`. 남는 시간은 차선 마모 보정(#6)과 통합에.
**11/21 Go/No-go** (`PRD.md:643`): E2E가 늦으면 위험도는 유형만으로.

### M3 화면 · M4 실험 — 11/22(일) ~ 11/24(화)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 3.1 | 정현우 | `exposure.py` — `conf ≥ θ OR vehicle_count ≥ k`, 시설물 의심은 `vehicle_count ≥ k`만 | 기준 미달 결함이 목록에 없음 |
| 3.2 | 정현우 | `/trips/{id}/findings`, `/defects/{id}/report`, `/skip` | API 테스트 |
| 3.3 | 정현우 | `reporting.py` — `INSERT … ON CONFLICT (defect_id) DO NOTHING RETURNING id` | 동시 요청 20개 → 1건 |
| 3.4 | 정현우 | `reward.py` + `reward_roads.geojson` | 포상 도로 결함 신고 → 유효 건수 +1 |
| 3.5 | 정현우 | 시민 화면 (캡처·유형·위치·위험도·신뢰도·차량 수·포상 도로, [신고]/[넘기기]) | 시연 5~6단계 |
| 3.6 | 정현우 | 관리자 지도 (상태별 마커, 상세, '보수 중' 변경) | 신고 후 마커 '신고됨' |
| 3.7 | 정현우 | `--detector replay` 모드 | CPU 추론 없이도 시연 흐름 동작 |
| 4.1 | 박겸 | `eval_detector.py` — 공개 테스트셋 / **선정 공개 영상** 분리 평가 (`PRD.md:579`) | 12장 표 탐지 행 |
| 4.2 | 박겸 | `bench_cpu.py` — 1·2코어 × 주기 적용 전/후 | FPS 표 |
| 4.3 | 박겸 | `interval_sweep.py` — 간격 1/3/5/10/15 × 누락률 | 그래프 |
| 4.4 | 공동 | 노출 기준 θ, k 결정 (#3) | `server/app/config.py`에 값 |
| 4.5 | 정현우 | 통신량 절감률(기준값 2가지), 중복 정리 비율, **반경 10/20/30 m별 묶임 비율**. **11/16 잠정 측정**(실제 영상 gn_a + 가짜 탐지 + 가짜 GPS) → 11장. 모델·재구성 GPS·블랙박스 합치기(2.3)가 들어오면 11-4 의 명령으로 다시 잼 | 12장 표 + #2 확정 |
| 4.6 | 정현우 (P2) | BDD100K 클립 몇 개로 실제 GPS 경로 동작 확인 | `bdd_to_csv.py` → 같은 파이프라인 실행 성공 |

### 동결·리허설 — 11/25(수) ~ 11/27(금)

| 날짜 | 작업 | 완료 확인 |
|---|---|---|
| 11/25 | 기능 동결, 버그 수정, 12장 표 채우기 | 12장 표에 TBD 없음 |
| 11/26 | 리허설: `reset_demo.py` → `run_fleet.py` → 시민 화면 → 지도. 실시간·replay 두 경로. 발표 자료에 **영상 출처 표기**와 "GPS는 재구성했고 오차는 X m" 명시 (`PRD.md:438`, `PRD.md:447`) | 끊김 없는 시연 2회 이상 (`PRD.md:633`) |
| 11/27 | 완성 | — |

**시연 기본 설정 (11/16)**: 차선 마모는 끈다 (`edge/blackbox/run.py` 기본 `--lane-every 0`). 시연 클립 gn_b 에서 확정 58건 중 54건이 헛것(25건은 신호 대기 중 횡단보도), gn_a 는 젖은 노면이라 차선을 2.8% 프레임에서만 찾았다 (`docs/ai-handoff.md` 17장). 박겸이 11/20까지 개선해 보고, 통과하면 `--lane-every 3` 으로 켠다.

**발표에서 쓰는 말 — '여러 차량 확인'** (11/16)

| | 내용 |
|---|---|
| 사실 | `edge/blackbox/variants.py` 의 기기 2·3대째는 **같은 영상**에 GPS 오차·밝기·시작 시점만 바꾼 것이다. 서버의 관측 차량 수(`vehicle_count`)는 이때 "보낸 기기 수"이고 **독립 확인이 아니다.** 같은 헛것도 기기 수만큼 늘어난다 |
| 하지 않는 말 | "여러 차량이 확인했다", "N대가 따로 발견했다", 차선 마모 서버 확정 규칙(2대·3건)을 교차 검증으로 설명하기 |
| 하는 말 | "같은 결함을 GPS 오차가 있는 여러 기기가 보내도 서버가 한 건으로 묶는다" (중복 정리 검증). "실서비스에서는 서로 다른 차량의 관측이 쌓여 신뢰도가 오른다" (설계 설명, 시연 결과로 말하지 않음) |
| 독립 관측 | gn_a·gn_b 는 다른 날 다른 사람이 찍은 영상이라 독립 관측이 맞다. 다만 v3 는 gn_a(젖은 노면)에서 균열 0건이라, 두 영상이 같은 결함을 함께 본 사례는 아직 없다 (handoff 18장) |
| 노출 기준 | 노출 기준 `vehicle_count ≥ k`(k=2)는 변형 기기끼리는 의미가 없다. 신뢰도가 θ 아래인 헛것도 변형 2대가 보내면 시민 목록에 뜬다. 시연에서 k 를 어떻게 둘지는 #3 에서 정한다 |

---

## 6. 수용 기준 (테스트 가능한 형태)

| ID | 기준 | 확인 방법 |
|---|---|---|
| AC-1 | 선정 공개 영상에서 4클래스 각각 1건 이상 탐지 (대체안 시 시설물은 위치만) (`PRD.md:227`) | `e2e_smoke.py` 클래스별 결함 수 |
| AC-2 | CPU 1·2코어 제한에서 시연 영상 처리 완료, 주기 적용 전/후 FPS 비교 표 | `experiments/bench_cpu.csv` |
| AC-3 | 품질 불량 구간에서 탐지 미실행 로그 + 적용 전/후 오탐 수 비교 | metrics JSONL + `experiments/quality_ab.csv` |
| AC-4 | 가상 블랙박스 3대 동시 실행 시 전송 관측 수 = 저장 관측 수 | `e2e_smoke.py` assert |
| AC-5 | 통신량 절감률이 기준값 2가지로 계산되어 있음 | `experiments/bandwidth.py` → `experiments/bandwidth_*.json` (11장) |
| AC-6 | 서로 다른 영상(또는 노이즈 차량)에 공통으로 보이는 라벨 결함 중 하나로 묶인 비율 제시 | `experiments/dedup_distance.py --replay` (11장) |
| AC-7 | 같은 결함 동시 신고 20건 → `reports` 1행, 나머지 `already_reported` | `test_report_concurrency.py` |
| AC-8 | 노출 기준 미달 결함은 `/findings`에 없음 | `test_exposure.py` |
| AC-9 | 신고서 payload에 사진·위치·유형·신뢰도·관측 차량 수·신고자 | `test_reporting.py` |
| AC-10 | 포상 도로 결함 신고 → `quarter_valid_count` +1, 비대상 도로는 변화 없음 | `test_reward.py` |
| AC-11 | 시민 신고 직후 지도 마커 '신고됨' | 리허설 체크리스트 |
| AC-12 | 두 번째 차량 시민에게 이미 신고된 결함이 안 보임 (시연 8단계) | `test_findings.py` |
| AC-13 | 재구성 GPS의 hold-out 오차가 영상별로 기록되어 있음 | `experiments/gps_error.csv` |
| AC-14 | 같은 GPS 파서가 재구성 CSV와 (P2) BDD100K CSV를 모두 읽음 | `test_gps.py` |

---

## 7. 리스크와 대응

| 리스크 | 신호 | 대응 |
|---|---|---|
| **같은 도로를 지나는 영상 2개를 못 구함** | 11/15까지 후보 1개 | 한 영상을 여러 가상 블랙박스에 넣고 `--gps-noise-m 5~10` (`PRD.md:666`). 발표에서 방식 명시 |
| **두 영상의 촬영 시점이 달라 결함이 다름** (보수됨·새로 생김) | 공통 라벨 결함 5건 미만 | 공통 결함만으로 중복 정리 비율 계산, 나머지는 "단독 관측"으로 분리 집계 |
| **재구성 GPS 오차가 큼** | hold-out 평균 오차 > 20 m | 기준점 추가(가로등·건물 입구), 속도 표시 영상은 OCR 누적 거리(`PRD.md:445`), 반경 조정 |
| **영상 라이선스** | CC 영상 없음 | 업로더에게 허락 요청, 영상은 레포·발표 자료에 원본 배포 안 하고 캡처만 출처와 함께 사용 |
| **유튜브 영상 압축·화각 문제** (노면이 뭉개짐, 화면 오버레이) | 탐지 결과가 공개셋보다 크게 낮음 | 1080p 이상만 선정, 화면 하단 오버레이 영역 마스킹, AI Hub 국내 데이터로 보강 |
| **RDD2022(해외 도로) → 국내 도로 성능 저하** | 선정 영상 평가 mAP 저조 | AI Hub 도로장애물/표면 데이터로 v2 학습 (`PRD.md:373`), 시간 되면 선정 영상 프레임 일부 라벨링해 파인튜닝 |
| **AI Hub 데이터(전국 도로시설물·도로장애물) 승인 지연·대용량** (각 1 TB 이상) | 11/16까지 미승인 | RDD 2클래스로 진행. 승인 후 라벨만 먼저 받아(0.2b) 필요한 이미지 zip만 다운로드, 클래스당 수천 장 샘플링 (`PRD.md:670`, `PRD.md:375-394`) |
| **KRID 파손 라벨이 생각보다 적거나 품질이 낮음** | 0.2b 집계에서 가드레일·표지판 파손 각 100개 미만 | `Validation` 이미지·다른 도로 종류 zip 추가, 공개 이미지 보충, 그래도 부족하면 11/20 대체안 — **11/15 현실화: 대체안 확정** |
| 연속 프레임 중복으로 관측 폭증 | 결함 수 ≫ 라벨 수 | `track_merge.py` + `clustering.py` 2단계 |
| 동시 신고 경쟁 | AC-7 실패 | UNIQUE + `ON CONFLICT`, `FOR UPDATE` |
| 시연 현장 성능·네트워크 | 리허설 중 FPS 하락 | 전부 로컬 docker, replay 모드 대기 |
| 인터페이스 변경으로 서로 깨짐 | import 에러 | `contracts/` 변경은 둘 다 확인, 모델은 버전 파일명으로만 교체 |

---

## 8. 검증 단계

1. **단위 테스트** (`pytest`): `gps` 파서·보간(재구성/BDD CSV), `reconstruct` 거리 비례 보간, `scheduler`, `quality`, `track_merge`, `clustering`, `exposure`, `reporting` 동시성, `reward`.
2. **데이터 검증**: `preview_map.py`로 영상별 경로 육안 확인, `holdout_error.py` 오차 표.
3. **통합 테스트**: `scripts/e2e_smoke.py` — 영상 2개(또는 노이즈 3대) → DB 검증 (AC-1, AC-4, AC-6). 11/21부터 매일 1회.
4. **성능 실험**: `ai/eval/*` → `experiments/` CSV → 발표 그래프.
5. **시연 체크리스트**: PRD 13장 1~8단계(`PRD.md:593-603`)를 실시간·replay 두 경로로 통과.

---

## 9. 오픈 이슈 처리 계획

| # | 이슈 | 권장 | 확정 시점 |
|---|---|---|---|
| 1 | 균열 클래스 | `crack` 하나로 병합 | 11/14 |
| **13** | **주행 영상 출처** | **유튜브 국내 + 경로 재구성 (메인), BDD100K (보조)** | 11/14 |
| 11 | 화면 방식 | Streamlit | 11/15 |
| 12 | 비식별화 | 이번 버전 제외 (컷 라인 1순위) | 11/15 |
| 4 | 먼저 확인한 시민 | 먼저 [신고] 누른 시민 | 11/18 |
| 8 | 넘기기 | 같은 시민에게 다시 묻지 않음 | 11/18 |
| 10 | 클립 | 전후 2초, 480p mp4 | 11/18 |
| 7 | 포상 도로 | 선정 영상 경로 일부를 GeoJSON 지정 | 11/20 |
| 2 | 묶기 반경 | 20 m 시작, 10/20/30 m + 재구성 오차로 확정. **11/16 잠정(11-3)**: 관측마다 독립 GPS 오차 7 m 이면 20 m 는 나뉨이 생기고(합성 바르게 69%, 실제 파이프라인 재생 결함 5개 → 10건) 30 m 가 낫다(78%, 5건). 오차 3~5 m 면 20 m 가 가장 낫다(91~93%). 재구성 GPS 오차(AC-13)가 나오면 그 행으로 정함 | 11/21 |
| 5 | 크기→위험도 | 유형만 (시간 되면 하단 노면 기준 정규화) | 11/21 |
| 6 | 차선 마모 확정 | 3회 이상 관측 + 대비 기준값은 실측 분포로. **제안(r3.8)**: 프레임 `score < 1.0` → 블랙박스 연속 5 중 3 투표로 관측 1건 → 서버는 한 지점에 차량 2대 이상 · 관측 3건 이상일 때 확정 (서버 규칙은 정현우 확인 필요, `docs/ai-handoff.md` 8장). r3.12: 실측 재귀반사도(AI Hub 654 LTL-M)와는 맞지 않음(AUC 0.51) — 기준은 그대로, 시연 영상으로 확정. **r3.15: 시연 클립에서 헛것 93% → 시연 기본은 끔(`--lane-every 0`), 11/20까지 개선해 보고 통과하면 켬.** 서버 2대·3건 규칙은 변형 기기끼리는 독립 확인이 아님 (5장 "발표에서 쓰는 말") | 11/20 |
| 9 | 보수 중 전환 | 관리자 지도에서 수동 | 11/22 |
| 3 | 노출 기준 θ, k | 선정 영상 정밀도 기반 | 11/24 |

---

## 10. 바로 시작할 일 (오늘·내일)

- [ ] 박겸: **AI Hub「전국 도로시설물 영상정보」+「도로장애물/표면 인지 영상(수도권)」 다운로드 신청**, RDD2022 다운로드 시작. 승인되면 라벨만 먼저 받아 집계 (0.2b)
- [ ] 정현우: 유튜브 영상 후보 검색 (`PRD.md:437` 검색어), `sources.csv` 작성 → 11/14 #13 확정
- [ ] 공동: 2장 기술 결정 확인 → 레포 생성 + `contracts/` 작성

---

## 11. 측정 결과 (PRD 12장 통신량·중복 정리 행, 계획 4.5)

**모두 잠정이다 (11/16).** 실제 탐지 모델·재구성 GPS·블랙박스 합치기(2.3)가 아직 없어서 아래 입력으로 측정 절차를 끝까지 한 번 돌렸다. 수치는 낮게 나와도 그대로 적고, 입력이 바뀌면 11-4 의 같은 명령으로 다시 잰다.

| 입력 | 이번에 쓴 것 | 결과에 미치는 영향 |
|---|---|---|
| 영상 | **실제 영상** `gn_a` (유튜브 원본 그대로, 1080p 59.94fps VP9, 2,093초, 789.4 MB ≈ 3.0 Mbps). 시연 구간 20:50~25:10 은 같은 스트림을 복사한 `gn_a_2050-2510.mp4` (262초, 97.7 MB) | 사진·클립 바이트는 실제 화면으로 잰 값 |
| 탐지 | **가짜 탐지** (`bandwidth.py --fake-every 60`): 5초부터 60초마다 결함 1개가 2초 동안 화면 아래쪽(가로 40~55%, 세로 70~82%)에 보인다고 치고, 분석하는 프레임마다 1건씩 냄. 유형은 포트홀·균열 번갈아. 품질 판별·차선 마모는 끔. 모델 `ai/models/detector_v2.onnx` 가 이 Mac 에 없어서 | **보낸 용량은 결함 수(1분 1개 가정)에 거의 비례한다.** 실제 탐지 수로 다시 재야 함 |
| 분석 주기 | `--detect-every 10` (60fps 영상이라 초당 6번) | 블랙박스 합치기 전이라 결함 1개를 2초 동안 12번 보냄 → 사진이 용량의 64% |
| GPS | **가짜 GPS**: 2점 직선·일정 속도 (전체: 도곡동→한남대교 / 시연 구간: 강남역→신논현역, 약 2.9 m/s) | 통신량에는 거의 영향 없음(메타 몇 바이트). 중복 정리에는 영향 큼 |
| 여러 차량 | 한 영상을 `edge/blackbox/variants.py` 의 3대 조건으로 차례로 재생: bb-01 원본, bb-02 GPS 오차 7 m·밝기 0.85, bb-03 GPS 오차 7 m·2프레임 늦게. **GPS 오차는 관측마다 독립**(실제 GPS 오차는 시간에 따라 이어지므로 이쪽이 더 나쁜 경우) | 같은 결함이 여러 건으로 나뉘는 쪽으로 치우침 |
| 바이트 | 메타 JSON + 사진 JPEG + 클립 mp4 의 합. HTTP·multipart 머리는 뺌 | 실제 전송량보다 조금 작음 |

### 11-1. 통신량 절감률 (AC-5, PRD 12장 통신량 행)

절감률 = 1 − 보낸 용량 ÷ 기준값. 기준값 (a) 내려받은 영상 파일 크기, (b) 일반 블랙박스 1080p **10 Mbps(가정값)** × 재생 시간 (2장).

| 입력 | 재생 시간 | 기기 | 보낸 관측 (1대) | 보낸 용량 (1대) | 내역 (1대) | (a) 파일 | 절감률 (a) | (b) 10 Mbps | 절감률 (b) |
|---|---|---|---|---|---|---|---|---|---|
| `gn_a.mp4` 전체 | 2,093초 | 1 | 420건 (결함 35개) | 18.10 MB | 메타 0.17 MB · 사진 11.62 MB · 클립 35개 6.32 MB | 789.4 MB | **97.71%** | 2,616 MB | **99.31%** |
| `gn_a_2050-2510.mp4` 시연 구간 | 262초 | 3 (평균) | 60건 (결함 5개) | 2.83 MB | 메타 0.02 MB · 사진 1.59~1.72 MB · 클립 5개 1.13 MB | 97.7 MB | **97.10%** | 328 MB | **99.14%** |

- 클립: 854×480 H.264 30fps, 4초 121장, 한 개 57~362 KB (평균 약 180 KB). 같은 결함의 뒤 탐지 11번은 앞 클립 안이라 다시 자르지 않음 (`clips_skipped_overlap`).
- (a) 는 유튜브가 3.0 Mbps 로 다시 압축한 파일이라 (b) 보다 절감률이 낮게 나온다 (2장에서 예상한 그대로).
- 블랙박스 합치기(2.3)가 들어오면 결함 1개 = 관측 1건이 되어 사진이 결함당 12장 → 1장으로 줄어든다. 이 효과는 아직 재지 않았다.
- 원자료: `experiments/bandwidth_gn_a_full_fake.json`, `experiments/bandwidth_gn_a_demo_fake.json`.

### 11-2. 중복 정리 비율 (AC-6, PRD 12장 중복 정리 행)

중복 정리 비율 = 1 − 서버 결함 수 ÷ 서버로 간 관측 수. 시연 구간, 3대, 서버 반경 20 m (지금 기본값).

| 입력 | 서버로 간 관측 | 서버 결함 | 정답 결함 | 중복 정리 비율 | 2대 이상이 본 결함 |
|---|---|---|---|---|---|
| 시연 구간 + 가짜 탐지 + 가짜 GPS, 3대 차례로 | 180건 | 9건 | 5개 | **95.0%** (정답대로 묶였다면 97.2%) | 7건 |

- 늘어난 4건은 GPS 오차 7 m 를 받은 관측 일부가 큰 결함에서 떨어져 나가 따로 생긴 작은 결함이다 (같은 조건의 첫 측정에서 각 관측 1~4건, 차량 1~2대).
- 사진·클립은 서버에 8.42 MB 저장됨(`/api/v1/stats` 와 같은 합계).

### 11-3. 반경별 묶임 (오픈 이슈 #2)

`experiments/dedup_distance.py`. "바르게" = 실제 결함 중 서버 결함 하나로만 묶이고 다른 결함과 섞이지 않은 비율.

**(가) 실제 파이프라인이 보낸 관측 재생** (`--replay experiments/bandwidth_gn_a_demo_fake.json`, 위 11-2 의 관측 180건을 영상 시각 순으로 다시 넣음):

| 반경 | 서버 결함 (정답 5개) | 나뉨 | 합쳐짐 | 바르게 | 중복 정리 비율 |
|---|---|---|---|---|---|
| 10 m | 27건 | 22 | 0 | 0% | 85.0% |
| 20 m | 10건 | 5 | 0 | 20% | 94.4% |
| 30 m | 5건 | 0 | 0 | 100% | 97.2% |

11-2 의 실제 서버(20 m)는 9건, 재생(20 m)은 10건이다. 넣는 순서가 달라서다 (실제: 기기별로 차례로, 재생: 영상 시각 순으로 섞어서). 결함이 60초 = 약 170 m 간격이라 이 입력으로는 '합쳐짐'을 볼 수 없다.

**(나) 합성** (결함 30개를 반지름 400 m 안에 흩뿌림 — 가장 가까운 결함까지 평균 약 65 m, 일부는 20~40 m. 차량 3대가 결함마다 1번씩, 씨앗 5개 평균):

| GPS 오차 (m) | 반경 10 m | 반경 20 m | 반경 30 m |
|---|---|---|---|
| 0 | 29.8건 (나뉨 0.0, 합쳐짐 0.2, 바르게 99%) | 28.8건 (나뉨 0.0, 합쳐짐 1.2, 바르게 92%) | 27.8건 (나뉨 0.0, 합쳐짐 2.2, 바르게 85%) |
| 3 | 32.0건 (나뉨 2.2, 합쳐짐 0.2, 바르게 91%) | 29.0건 (나뉨 0.0, 합쳐짐 1.0, 바르게 93%) | 27.2건 (나뉨 0.0, 합쳐짐 2.8, 바르게 81%) |
| 5 | 46.0건 (나뉨 16.8, 합쳐짐 0.8, 바르게 48%) | 30.2건 (나뉨 1.2, 합쳐짐 1.0, 바르게 91%) | 27.2건 (나뉨 0.0, 합쳐짐 2.8, 바르게 81%) |
| 7 | 59.0건 (나뉨 30.4, 합쳐짐 1.4, 바르게 18%) | 35.6건 (나뉨 7.4, 합쳐짐 1.8, 바르게 69%) | 27.6건 (나뉨 1.0, 합쳐짐 3.4, 바르게 78%) |
| 10 | 69.0건 (나뉨 40.8, 합쳐짐 1.8, 바르게 7%) | 44.2건 (나뉨 17.6, 합쳐짐 3.4, 바르게 41%) | 32.0건 (나뉨 6.2, 합쳐짐 4.2, 바르게 65%) |

**잠정 판단 (#2 확정 아님)**: 관측 GPS 오차가 3~5 m 면 20 m 가 가장 낫고, 7 m 이상이면 30 m 가 낫다 (대신 가까운 결함끼리 합쳐짐이 늘어남). 확정은 재구성 GPS 의 hold-out 오차(AC-13, `gps-reconstruct`)와, 블랙박스 합치기 뒤 실제 탐지로 다시 잰 (가) 표로 한다.

### 11-4. 다시 재는 방법

영상·GPS·모델·블랙박스 합치기가 들어오면 같은 명령으로 다시 재고 이 장의 표를 바꾼다. 11/16 에는 시연용 DB 를 건드리지 않으려고 따로 만든 DB(`DATABASE_URL=…/daboyeo_bw`)와 포트 8010 서버로 쟀다.

```bash
# 통신량만 (서버 없이). 모델이 있으면 --fake-every 60 대신 --model ai/models/detector_v2.onnx
uv run python experiments/bandwidth.py --video data/videos/gn_a.mp4 --gps <gn_a GPS CSV> --fake-every 60 \
    --devices 1 --dry-run --detect-every 10 --keep-dir "" --json experiments/bandwidth_gn_a_full_fake.json
# 3대 + 서버 (통신량 + 중복 정리). 서버를 먼저 띄운다 (scripts/demo.sh up 이면 --server 기본값 8000)
uv run python experiments/bandwidth.py --video data/videos/gn_a_2050-2510.mp4 --gps <구간 GPS CSV> --fake-every 60 \
    --devices 3 --detect-every 10 --server http://localhost:8010 --json experiments/bandwidth_gn_a_demo_fake.json
# 반경 10/20/30 m (DB 를 비운다): 위 결과 재생, 합성
uv run python experiments/dedup_distance.py --yes --replay experiments/bandwidth_gn_a_demo_fake.json --distances 10,20,30
uv run python experiments/dedup_distance.py --yes --distances 10,20,30
scripts/demo.sh reset      # 시연용 DB 로 쟀다면
```

- 모델이 생기면 결과 파일 이름의 `_fake` 를 `_v2` 등으로 바꾸고, 블랙박스 합치기(2.3) 전·후를 둘 다 남긴다.
- 정답이 있는 비교(나뉨·합쳐짐)는 지금 가짜 탐지에서만 나온다. 실제 탐지로는 `experiments/gt_events/` 라벨이 있어야 한다.
- 영상 두 개(gn_a·gn_b)로 '같은 도로 2대'를 재려면 두 영상의 GPS 가 같은 좌표계로 재구성되어야 한다 (지금은 한 영상 + 오차 차량).

---

## 변경 이력

- **r3.18 (2026-11-17)** — 주행 종료 모의 문자와 가입 없는 링크 신고 구현 ([notify-link-plan.md](notify-link-plan.md), 정현우). 주행을 끝내면 확인할 결함이 있는 시민에게 문자(링크 포함)를 만들어 `notifications` 표에 저장한다(**실제 발송 없음, 전부 모의**). 링크의 토큰이 신원 확인을 대신하고(72시간), 링크 화면에서 [신고] 한 번으로 신고서가 만들어진다. 화면에 메시지함(모의) 추가. `contracts/`·`ai/`·`edge/` 변경 없음. PRD F6 문구(화면 알림 → 문자 링크)는 박겸 확인 전이라 고치지 않았다.
- **r3.17 (2026-11-17)** — 블랙박스 연속 프레임 합치기(2.3)·분석 주기(2.2) 구현, 정현우. `edge/blackbox/track_merge.py`·`scheduler.py`, `run.py` 는 탐지 결과를 보내는 부분만 바뀜, 계약·추론 함수는 그대로. **IoU>0.3 만으로는 안 됨**: 노면 결함은 땅에 붙어 있어, 차가 결함 길이(포트홀 0.3~0.6 m)보다 많이 나아가면 앞뒤 박스가 세로로 겹치지 않는다(핀홀: 거리 Z 의 땅 위 점은 지평선 아래 f·h/Z). 핀홀 모의(결함 폭 0.4/0.8 m · 길이 0.3/0.6 m · 옆 위치 4곳, 30 fps)에서 결함 1개당 관측 수가 IoU 만이면 50 km/h 3·5·10 프레임마다 6.5·4.0·1.5건, 30 km/h 3프레임마다 8.9건 → IoU+움직임 문은 모든 속도(30·50·80 km/h)·주기(1~15 프레임)에서 1.00건, 탐지 20% 누락에도 1건. 1.5 m 간격으로 나란한 포트홀 2개는 2건으로 따로 남음. 가까운 같은 유형 결함이 문 때문에 합쳐져도 서버가 20 m 안 같은 유형을 어차피 한 결함으로 묶으므로 결함 수는 같다. 긴 균열이 1건으로 뭉개지지 않게 트랙은 최대 3초(50 km/h 약 42 m). **합성 영상 비교** (640×360 · 30 fps · 50 km/h · 60초, 포트홀 8개 + 30 m 균열 1개, ONNX 대신 색 문턱값 탐지기, `--dry-run`): 보낸 관측이 탐지 1·3·10 프레임마다 379·131·38건 → **10·10·10건** (포트홀 8 + 균열 2: 균열이 4.4초 보여 3초에서 잘림), 통신량 1,366·462·137 KB → 29·31·36 KB. 단계별 실행(10프레임마다): 품질 1800/1800 (100%) · 탐지 180/1800 (10%). 실제 주행 영상·`detector_v2.onnx` 로는 아직 안 돌림 — 영상이 오면 `run.py … --dry-run` 과 `--no-merge` 로 같은 비교. 블랙박스 화면의 '서버로 보낸 탐지' → '서버로 보낸 관측' (트랙이 끝난 뒤 올라감, 도움말에 탐지 횟수).
- **r3.16 (2026-11-17)** — 1.3 클립·전송 바이트 집계 구현과 4.5 잠정 측정 (11장 새로 씀). 블랙박스가 결함 전후 2초를 480p H.264 로 잘라(`edge/blackbox/clipper.py`) 사진과 함께 보내고 따로 남김, 서버가 `clip` 을 받아 `clip_path·clip_bytes` 저장. 실제 영상 gn_a + **가짜 탐지(1분 1개) + 가짜 GPS(직선)** 로: 통신량 절감률 (a) 내려받은 파일 기준 97.71% · (b) 10 Mbps 기준 99.31% (전체 2,093초, 1대 18.10 MB), 시연 구간 3대 중복 정리 비율 95.0% (관측 180 → 결함 9, 정답 5). 반경: 관측마다 독립 7 m 오차면 20 m 는 나뉨(재생 10건), 30 m 는 정답대로 5건. 합성에서는 오차 3~5 m 면 20 m 가 가장 나음. 모델·재구성 GPS·합치기 전이라 **#2 는 아직 확정하지 않음**. 측정 스크립트 `experiments/bandwidth.py` 를 로드마크 구간에서 영상 입력으로 바꾸고 `dedup_distance.py` 에 `--replay`·바르게 묶임 비율 추가.
- **r3.15 (2026-11-16)** — 시연 기본에서 차선 마모를 끔 (#6, 정현우). `edge/blackbox/run.py` 의 `--lane-every` 기본값 3 → 0. 근거는 시연 클립 결과 (handoff 17장: gn_b 확정 58건 중 헛것 54, 그중 25건이 신호 대기 중 횡단보도, gn_a 는 차선 찾은 프레임 2.8%). 박겸이 11/20까지 개선해 보고 통과하면 `--lane-every 3`. 5장에 "발표에서 쓰는 말": 변형 기기의 관측 차량 수는 독립 확인이 아니므로 "여러 차량이 확인했다"고 말하지 않는다. 노출 기준 k 가 변형 기기에서 헛것을 통과시키는 문제는 #3 에서 정한다.
- **r3.14 (2026-11-16)** — 시연 모델을 탐지 v3 로 확정 (박겸). 시연 클립 gn_a·gn_b 889프레임에서 v2 는 진짜 균열 0, v3 는 gn_b 에서 4곳 (균열 기준 0.42 에서 real 19·헛것 1). 클래스별 기준 `{"pothole": 0.31, "crack": 0.42, "guardrail": 0.21, "sign": 0.21}` = `ai/models/detector_v3.json` `class_conf_draft`. 자막 마스킹 불필요, 품질 게이트 통과 100%. `docs/ai-handoff.md` 16장. 정현우: 블랙박스 모델 경로·기준 교체.
- **r3.13 (2026-11-15)** — 차선 마모 LTL-M 검증 보강 (정현우 11/12 결과 반영, #6). '30 미만 = 마모'(정현우가 표본 사이에서 고른 값, 매뉴얼 기준 아님)로도 잼: ★ AUC 낮 0.62 · 밤 0.72, 잠정 1.0 정확도 낮 70.6% < '전부 정상' 76.9% (정현우 이전 판정 79.4% < 80.3% 와 같은 모양) — 정확도 대신 AUC·F1 로 판단. 같은 곳 다시 지날 때 30 미만이 되풀이될 확률 19% ≈ 기저 14% → 30 미만은 장소(닳은 페인트)보다 순간(측정기가 페인트를 벗어남)에 가까움. 측정하는 선은 쪽별 비교로도 정할 수 없음(왼쪽을 약하게 가리킬 뿐). GPS 파서가 A·B·C 형식과 잘못된 행을 처리. 기준·규칙은 그대로. `docs/ai-handoff.md` 8-1.
- **r3.12 (2026-11-15)** — 차선 마모 판정을 실측 재귀반사도로 검증 (2.6, #6). AI Hub 로드마크(654) VS4.비온전(복합)+VL4, 이미지·LTL-M 이 모두 있는 1,996장, 한 프레임 판정. 도구 `ai/daboyeo_ai/eval/lane_wear_roadmark.py`, 집계 `experiments/lane_wear_roadmark.json`. 마모 = LTL-M < 100 mcd/(m²·lx)(유지 최소 백색 100 · 황색 70), 30 미만은 '측정기 아래 페인트 없음'으로 뺌. 결과: 낮 ★ score AUC 0.51 · 2-fold F1 0.40(전부 마모라고 부른 것과 같음), contrast 는 거꾸로(AUC 0.32). LTL-M 은 이웃 프레임끼리 일관되지만(Spearman 0.90) 같은 곳을 밤에 재면 31 높고, 차선이 아닌 노면 밝기와 같이 움직임(낮·맑음 Spearman 0.91) → 이 데이터의 LTL-M 은 깨끗한 마모 정답이 아님. **기준 1.0·투표 규칙은 그대로**, 발표에서 '재귀반사도를 잰다'고 하지 않음. 차선 찾기는 VL4 선 위치와 맞음(자기 차로 선 있는 프레임 96% 찾음, 채택 선분 72% 가 라벨 선 위). AI Hub 데이터는 재배포 금지라 이미지·프레임별 값은 레포에 두지 않음, 발표에 프레임을 보여도 되는지는 미확인. `docs/ai-handoff.md` 8-1.
- **r3.11 (2026-11-15)** — 실제 주행 영상 평가 도구 (4.1·4.3, 시연 영상·라벨이 오는 날 바로 돌리기 위해). `ai/daboyeo_ai/eval/eval_detector.py`: 영상을 N 프레임마다 탐지 → 유형별 이벤트로 묶어 정답 창과 맞춤 → conf 0.25~0.6(+ 클래스별 `--conf-json`) 마다 이벤트 단위 정밀도·재현율·F1·분당 이벤트 수, 차선 마모는 `LaneWearTracker.confirmed`, 시설물은 위치 찾기로 따로(결함 합계에서 뺌), 공개 테스트셋 지표(`detector_v2.json`)를 나란히 출력. `interval_sweep.py`: 모든 프레임 1번 탐지·캐시 후 간격 1/3/5/10/15 로 골라 누락률·분석 프레임 수 (`--with matplotlib` 이면 그래프). 공통 `video_events.py`. 수동 라벨 형식 `defect_id,t_start_s,t_end_s,defect_type,memo` 제안 (`docs/ai-handoff.md` 12장, 정현우 확인 필요 — 1.1 가짜 탐지기도 같은 파일을 읽음). 실제 영상·라벨로는 아직 안 돌림.
- **r3.10 (2026-11-15)** — 클래스별 탐지 신뢰도 기준 초안 (#3 노출 기준 θ 준비). `ai/daboyeo_ai/eval/conf_thresholds.py`: test 이미지에서 정밀도·재현율 곡선을 내고 정밀도 ≥ 0.7 인 가장 낮은 기준을 고름 (없으면 F1 최대) → `experiments/conf_thresholds.{csv,json}`. 결과 pothole 0.22 · guardrail 0.26 · sign 0.24 (0.25 와 거의 같음, 0.8 목표면 0.33 · 0.42 · 0.37), crack 은 RDD2022 test 가 로컬에 없어 0.25 그대로. `OnnxDetector(conf={...})` 로 클래스별 기준 지원(기본값 그대로), 모델 카드에 `class_conf_draft` 키 추가. 블랙박스 기록 기준으로 쓸지 서버 θ 로 쓸지는 정현우와 정할 것 — `docs/ai-handoff.md` 4-1.
- **r3.9 (2026-11-15)** — CPU 처리 시간 측정 `ai/daboyeo_ai/eval/bench_cpu.py` (4.2, AC-2) 와 첫 결과 `experiments/bench_cpu.csv`·`.json`. `run`: 탐지(`OnnxDetector.detect`, v2)·차선 마모(`measure`+`tracker.update`, 1스레드)를 threads 1·2 × 640×360·1280×720 로 재고, 30 fps 영상 1분을 1·3·5·10·15 프레임마다 분석할 때의 계산 시간을 계산(실제 주기 스케줄러는 정현우). `log`: 정현우 측정 로그 JSONL 을 주기 적용 전/후로 요약 (사용법 `docs/ai-handoff.md` 11장). Mac M4 Pro 참고값(실제 프레임 300장 × 3회, 다른 작업 부하 있음): 탐지 median 1스레드 21~23 ms · 2스레드 17~18 ms, 차선 마모 4~5 ms, 탐지+차선 1스레드 매 프레임이면 영상 1분당 51~53 초(실시간 이내), 5 프레임마다 약 11 초. Colab CPU(1스레드 134 ms)보다 약 6배 빠르므로 **공식 수치는 노트북에서 같은 명령으로** 다시 잰다 (Linux 는 코어 고정까지, macOS·Windows 는 스레드 수 제한만).
- **r3.8 (2026-11-15)** — 차선 마모 판정을 엄격하게 (2.6, #6). 기준 1.7 에서 차선 찾은 프레임의 41% 를 마모로, 선명 차선의 18% 를 마모로 봤음. 1단계: 잠정 기준 `PROVISIONAL_WEAR_SCORE` 1.7 → **1.0** (마모 비율 28%, 선명→마모 12%; 한 프레임 정밀도는 기준값과 상관없이 ~0.5). 2단계: `LaneWearTracker` 에 연속 프레임 투표 — 최근 판정 프레임 5 개 중 3 개 이상 마모면 `WearJudgement.confirmed=True` (관측 1건), 확정 뒤 창이 비워질 때까지(또는 `cooldown`) 다시 내지 않음, 차선 못 찾은 프레임은 건너뜀, `t` 로 5 초 넘는 끊김 뒤 창 비움 (`votes`·`n_voted`·`confirmed` 필드 추가, 기본값 있음). 블랙박스는 `j.confirmed` 일 때만 `ObservationIn(LANE_WEAR)` 1건. 3단계(제안, 정현우 확인 필요): 서버는 한 지점에 차량 2대 이상 · 관측 3건 이상일 때 확정. `lane_wear_eval` 에 기준값별 마모 비율·선명→마모 비율 표 추가. 투표 효과는 독립 프레임 라벨로 잴 수 없어 시연 영상으로 확인 (11/22).
- **r3.7 (2026-11-15)** — 차선 마모 판정을 절대 대비 기준값 하나에서 **상대 대비 + 페인트 양**으로 바꿈 (2.6, #6). 손 라벨 143장(선명 112 / 마모 31, KRID·AI Hub 도로장애물)에서 절대 contrast 는 AUC 0.56 · best F1 0.37 로 카메라·조명에 더 좌우됨. 새 점수 `score = (contrast_p25 ÷ 같은 주행의 중앙값) × paint_len` (행마다 잰 대비의 하위 25% · 남은 차선 길이) 은 AUC 0.79 · best F1 0.58 · 2-fold F1 0.58 (두 카메라 각각 AUC 0.79). 잠정 기준 `score < 1.7` (TBD, 시연 영상으로 재확인). `LaneWearResult` 에 `contrast_p25`·`paint_len` 추가(기본값 있음), 주행마다 쓰는 `LaneWearTracker`(최근 100 프레임 중앙값, warmup 20), 비교 도구 `ai/daboyeo_ai/eval/lane_wear_eval.py`. `measure()` 와 `PROVISIONAL_THRESHOLD` 는 그대로. tracker 를 블랙박스에 둘지 서버(차량별)에 둘지는 정현우와 정할 것 — `docs/ai-handoff.md` 8장.
- **r3.6 (2026-11-15)** — 차선 마모 측정을 실제 프레임에 맞게 고침 (2.6, #6 준비). 실제 블랙박스 프레임 300장(KRID 150 + AI Hub 도로장애물 150)에서 잡힌 선분 대부분이 차선이 아니라 균열·이음매·가드레일·앞차였음 (contrast 32% 가 음수). 변경: 양옆 노면보다 밝은 줄무늬만 후보(극성), 원근·소실점 투표로 선분 거르기, 대비는 ROI 전체가 아니라 선분 바로 옆 노면과 비교, 밝기는 V(=max RGB, 노란 차선용), 화소 값은 폭 640 기준으로 해상도 비례, ROI 기본값을 화면 0.35~0.92 육각형으로. 결과: 차선 찾음 92% → 73% (잘못 찾던 프레임이 빠짐), 음수 contrast 32% → 0%, 중앙값 0.10 → 0.50, 1280×720 약 4 ms. 기준값(잠정 0.3)은 그대로 — 손 라벨로 #6 에서 정한다. `docs/ai-handoff.md` 8장 ROI 설명 갱신.
- **r3.5 (2026-11-15)** — 탐지 v2 확정 (RDD test crack 0.584 ≥ v1 0.570, `ai/models/detector_v2.json`). 시설물 파손 분류 (2.7) 학습 결과 No-go → **대체안(위치 탐지까지만) 확정**, PRD v0.5. 첫 학습은 Ultralytics 가 val 정확도로 best.pt 를 골라 '전부 정상' epoch 1 모델이 뽑혔다(val 이 정상 91%) → 조기 종료 끄고 last.pt 로 다시 학습(v1b)해 판단. 노트북 03 도 같은 방식으로 수정.
- **r3.4 (2026-11-15)** — 시설물 파손 분류 (2.7) 구현: `ai/daboyeo_ai/data/damage_cls.py`(크롭 → 시설물 종류 표 `krid_crop_types.csv.gz`, normal 줄이기), `ai/daboyeo_ai/eval/damage_metrics.py`, `ai/daboyeo_ai/inference/damage_classifier.py`, `OnnxDetector(damage_model=...)`, `ai/notebooks/03_train_damage_cls.ipynb`. 학습 크롭과 추론 크롭은 같은 `facility_crop`(CropWriter 에서 분리) + `square_input`(정사각형, Ultralytics 가운데 자르기에 가드레일이 잘리지 않게). 로컬 Mac 에서 2 epoch 모의 학습·ONNX 일치 확인. 사용법은 `docs/ai-handoff.md` 9장(정현우 확인 필요).

- **r3.3 (2026-11-14)** — 차선 마모 측정 `ai/daboyeo_ai/lane_wear.py`(`LaneWearMeter.measure(frame) -> LaneWearResult`)와 보정 도구 `ai/daboyeo_ai/eval/lane_wear_calib.py` 구현 (2.6). 합성 프레임 시험 통과, 1280×720 에서 약 1.5 ms. 기준값은 잠정 0.3 — 실제 프레임 분포로 #6 에서 결정. 사용법은 `docs/ai-handoff.md` 8장(정현우 확인 필요). 3장 레포 구조의 경로를 실제 위치로 갱신.
- **r3.2 (2026-11-13)** — 탐지 학습 노트북 `02_train_detector.ipynb`·`OnnxDetector` 구현 (1.4·1.5). 확인: 사전학습 YOLO11n ONNX 로 Ultralytics 와 박스·신뢰도 완전 일치, 노트북 전체를 CPU 1 epoch 로 모의 실행, 중단된 학습 이어하기·끝난 학습 건너뛰기 (끝난 last.pt 를 resume 하면 Ultralytics 가 COCO 샘플로 새 학습을 시작하는 문제를 막음).
- **r3.1 (2026-11-13)** — 데이터 준비를 Colab 에서 하도록 확정: `ai/daboyeo_ai/data/*`, `ai/notebooks/01_data_prep.ipynb` 구현. 원본은 Colab 로컬 디스크에서만 다루고 Drive 에는 변환 결과만 저장. 0.2b·0.3·0.4·2.7 작업 설명 갱신.
- **r3 (2026-11-13)** — PRD v0.4 반영: AI Hub 상세 페이지 확인 결과. KRID(전국 도로시설물 영상정보)에 정상/파손 라벨 있음 → 파손 분류 데이터를 KRID에서 추출(2.7, `prepare_damage_crops.py`), 라벨 먼저 받아 집계하는 0.2b 단계, 폴리곤→박스·마스크→박스 변환, 리스크 1건 추가, 0.9 완료 표시, PRD 줄번호 v0.4 기준으로 갱신.
- **r2 (2026-11-13)** — PRD v0.3 반영: 직접 촬영 → 공개 영상 + GPS 재구성(`tools/video`, `tools/gps`, M0 0.5~0.8), GPS CSV 계약, AI Hub 도로장애물 데이터 추가, BDD100K 보조 경로(4.6), 다중 차량 대체안(`--gps-noise-m`), 반경 20 m 시작, 재구성 오차 측정(AC-13), 통신량 기준값 2가지, 영상 관련 리스크 5건, 이슈 #13. r1의 "RDD2022 한국 서브셋" 오류 삭제.
- r1 (2026-11-13) — PRD v0.2 기준 최초 작성.
