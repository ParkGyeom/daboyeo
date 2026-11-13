# 다보여 MVP 실행 계획 (팀 전체)

| 항목 | 내용 |
|---|---|
| 기반 | [PRD.md](../PRD.md) **v0.3** (주행 영상 확보 방식 반영) |
| 계획 버전 | r2 (2026-11-13) — r1은 PRD v0.2 기준 |
| 완성 기한 | 2026-11-27 (금), D-14 |
| 상태 | 팀 검토 중 — 0.1(레포·`contracts/`)은 완료 |

> PRD 인용은 `PRD.md:줄번호`(v0.3 기준). `contracts/`, `server/sql/`, `docker-compose.yml` 외의 파일 경로는 아직 없는 **제안 경로**다.

---

## 0-A. PRD v0.3에서 바뀐 것과 계획에 미친 영향

| PRD 변경 | 위치 | 계획에서 바뀐 것 |
|---|---|---|
| **직접 촬영 → 공개 주행 영상(유튜브 국내 블랙박스·드라이브 영상)** | `PRD.md:372`, `PRD.md:402-417` | M0의 "주행 촬영" 작업 삭제 → **영상 선정 + GPS 경로 재구성**으로 교체 (정현우) |
| **GPS 경로 재구성** (GPX로 경로 그리기 + 기준점 시각 + 거리 비례 보간) | `PRD.md:419-426` | 새 도구 `tools/gps/` 추가, 재구성 오차를 재는 검증 단계 추가 |
| **GPS 입력 = `시각, 위도, 경도` CSV 하나** (FR-4.4) | `PRD.md:273` | 계약에 GPS CSV 형식 고정 (`t_s,lat,lon`), 실제 GPS·재구성 경로 모두 같은 파서 |
| **AI Hub 도로장애물/표면 인지 영상(수도권) 추가** — RDD2022에 한국 도로 없음 | `PRD.md:373` | 박겸: **오늘 KRID와 함께 다운로드 신청**, 국내 도로 포트홀·균열 보강용으로 데이터셋 v2에 합침. r1의 "RDD2022 한국 서브셋" 언급은 잘못이라 삭제 |
| **BDD100K(실제 GPS) 보조, Mapillary 백업** | `PRD.md:409-410` | GPS 동기화·중복 정리를 실제 GPS로 검증하는 **보조 경로(P2)** 추가 |
| **다중 차량 = 같은 도로를 지나는 서로 다른 업로더 영상 2개 이상**, 못 구하면 한 영상 + 좌표 노이즈 | `PRD.md:414`, `PRD.md:645` | `run_fleet.py`에 `--gps-noise-m` 옵션, 중복 정리 실험 설계 변경 |
| 중복 묶기 거리 참고: **20 m에서 동일 지점 인식률 90%+** 연구 | `PRD.md:693` | 시작 반경 15 m → **20 m**, 비교 10/20/30 m |
| 오픈 이슈 #13 주행 영상 출처 (11/14) | `PRD.md:686` | 9장 오픈 이슈 표에 추가 |
| 수용 기준·지표의 평가 영상 = "공개 영상(8.5)" | `PRD.md:227`, `PRD.md:558-560` | 평가용 수동 라벨을 **선정한 공개 영상**에 붙임 |

## 0-B. 이 계획이 PRD에 더하는 것

1. **`contracts/` 패키지를 첫날 만든다.** 클래스 순서·추론 결과·관측 전송·**GPS CSV** 형식을 코드로 고정해서, 11/18 인터페이스 맞춤(`PRD.md:608`)을 "import 한 줄"로 만든다.
2. **11/16에 가짜 탐지기로 걷는 골격을 먼저 띄운다.** 모델 전에 영상 → 전송 → 중복 정리 → 화면이 돌게 하고, 11/21 end-to-end(`PRD.md:609`)는 탐지기 교체만 남긴다.
3. **블랙박스 안에서 연속 프레임 탐지를 '결함 1건 = 관측 1건'으로 합친다.** 관측 차량 수·통신량 지표(`PRD.md:563-564`)가 이것에 좌우된다.
4. **재구성 GPS의 오차를 숫자로 잰다.** 기준점을 일부 빼고(hold-out) 보간 위치와 실제 위치의 거리를 측정한다. 이 값이 중복 묶기 반경(#2)의 근거가 되고 발표에서 한계로 밝힐 수치가 된다.
5. **시연 안전장치: replay 모드.** 현장에서 실시간 추론이 끊기면 미리 계산한 탐지 결과(JSONL)로 같은 흐름을 재생한다.
6. **macOS CPU 제한 방법.** `taskset`이 없으므로 ONNX Runtime 스레드 수 제한 + Docker `--cpus`로 측정한다(`PRD.md:398`).

---

## 1. 요구사항 요약

**끝까지 지킬 흐름** (`PRD.md:631`): 탐지 → 전송 → 중복 정리 → 주행 후 확인·신고 → 신고서 → 지도 반영.

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
| 주행 영상·GPS 확보 (8.5) | P0 (선행 조건) | 정현우 (+박겸 라벨) | `PRD.md:402-426` |

---

## 2. 기술 결정 (권장안 — 팀 확인 필요)

| 결정 | 권장안 | 이유 | 관련 이슈 |
|---|---|---|---|
| 레포 형태 | **모노레포** (`ai/`, `edge/`, `server/`, `ui/`, `contracts/`, `tools/`) | 두 명·2주, 계약 패키지를 양쪽에서 import | — |
| 언어/환경 | Python 3.11, `uv` 워크스페이스 (또는 폴더별 requirements) | 전 구성요소가 Python | — |
| **주행 영상 출처** | **유튜브 국내 영상 + 경로 재구성을 메인**, BDD100K 몇 개를 실제 GPS 검증용 보조 | PRD 추천안 (`PRD.md:408-409`), 서울 포상 도로 시나리오와 맞음 | **#13** (`PRD.md:686`) |
| GPS CSV | `t_s,lat,lon` (선택 열 `speed_kmh`), `t_s` = 영상 시작 기준 초 | FR-4.4 (`PRD.md:273`) | — |
| 관측 시각 | `observed_at = trip.started_at + t_s` (가상 주행 시작 시각 기준) | 공개 영상에는 실제 촬영 시각이 없음 | — |
| 탐지 모델 | Ultralytics YOLO **n급**(필요 시 s급), 입력 640 | Colab 시간·CPU 속도 (`PRD.md:650`) | — |
| 균열 클래스 | RDD2022 D00·D10·D20 → `crack`, D40 → `pothole`. AI Hub 도로장애물 데이터도 같은 2클래스로 매핑 | 4클래스 단일 모델 유지 | #1 (`PRD.md:685`) |
| 화면 | **Streamlit 하나** (시민 화면 = 좁은 레이아웃 페이지, 관리자 = `streamlit-folium` 지도) | PRD 권장안 (`PRD.md:619`) | #11 |
| DB | PostgreSQL 16 + PostGIS, `docker-compose` | `PRD.md:159` | — |
| DB 마이그레이션 | `server/sql/001_init.sql` 하나 (Alembic 생략) | 2주, 스키마 변경 적음 | — |
| "먼저 확인한 시민" | **먼저 [신고]를 누른 시민** (`reports.defect_id UNIQUE`) | 가장 단순하고 동시성 테스트로 검증 가능 | #4 |
| 넘기기한 결함 | 같은 시민에게 다시 묻지 않음 | 시연 단순화 | #8 |
| 클립 | 탐지 전후 2초, 480p, H.264 mp4 | 통신량 절감률을 보여주기 좋음 | #10 |
| 포상 대상 도로 | 선정한 영상 경로 중 일부를 **GeoJSON 폴리곤으로 지정** → `ST_Covers` | PRD 대안 (`PRD.md:692`) | #7 |
| 중복 묶기 반경 | **시작 20 m**, 10/20/30 m 비교 후 확정. 재구성 GPS 오차 측정값(0-B 4번)을 근거로 조정 | `PRD.md:693` 참고 연구 + 재구성 오차 | #2 |
| 위험도 | 우선 유형만으로 (`PRD.md:275-281`), 크기 반영은 11/21 판단 | `PRD.md:622` | #5 |
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
│       └── preview_map.py        # 경로·기준점을 folium 지도로 확인
├── ai/                           # 박겸
│   ├── data/
│   │   ├── prepare_rdd2022.py    # XML→YOLO txt, 클래스 병합
│   │   ├── prepare_aihub_road.py # AI Hub 도로장애물 13종 → pothole/crack 매핑
│   │   ├── prepare_krid.py       # KRID → guardrail/sign
│   │   ├── merge_dataset.py      # 합치기 + 분할 + 샘플링
│   │   └── dataset.yaml
│   ├── notebooks/{train_detector,train_damage_cls}.ipynb   # Colab
│   ├── export/export_onnx.py
│   ├── inference/{onnx_detector,damage_classifier}.py
│   ├── lane_wear/lane_wear.py
│   ├── eval/{eval_detector,bench_cpu,interval_sweep}.py
│   └── models/                   # .gitignore — detector_vN.onnx + detector_vN.json
├── edge/blackbox/                # 정현우 — 가상 블랙박스
│   ├── main.py                   # --video --gps --vehicle --detector {fake,onnx,replay} --gps-noise-m
│   ├── frame_source.py
│   ├── quality.py
│   ├── scheduler.py              # PRD 8.3 분석 주기
│   ├── track_merge.py            # 연속 프레임 → 결함 1건
│   ├── clipper.py
│   ├── uploader.py               # 전송 + 바이트 집계
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
    └── gt_events/<video_id>.csv  # 수동 라벨: t_s, 유형, 메모 (평가 정답)
```

---

## 4. 인터페이스 계약 (11/14까지 확정)

PRD 14장 합의 지점(`PRD.md:593-595`)을 코드로 고정한다.

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

- ONNX: `ai/models/detector_vN.onnx`, 입력 `1×3×640×640` float32 RGB 0~1. **전처리·NMS는 `ai/inference/onnx_detector.py` 책임** → 정현우는 `OnnxDetector(path, threads=1).detect(frame)`만 호출.
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

**블랙박스 → 서버: 관측** — `POST /api/v1/observations` multipart (JSON 메타 + 캡처 jpg + 클립 mp4). 메타 = PRD 9장 observations 필드(`PRD.md:457-471`). 관측 1건 = `track_merge.py`가 합친 결함 1건.

---

## 5. 일정과 단계별 작업

PRD 15.1(`PRD.md:605-613`)을 따르고, 각 작업에 **완료 확인 방법**을 붙였다.

### M0 준비 — 11/13(금) ~ 11/15(일)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 0.1 | 공동 | 레포 생성, 폴더 구조, `contracts/` (GPS CSV 포함), README | 두 사람 모두 `import daboyeo_contracts` 성공 |
| 0.2 | 박겸 | **오늘: KRID + AI Hub 도로장애물/표면 인지 영상 다운로드 신청** | 신청 완료 화면 |
| 0.3 | 박겸 | RDD2022 다운로드 + `prepare_rdd2022.py` (클래스 병합) | 클래스별 박스 수 표 |
| 0.4 | 박겸 | 데이터셋 v1 (RDD2022, 2클래스) + `dataset.yaml` | Colab에서 1 epoch 실행 |
| 0.5 | 정현우 | **영상 후보 찾기** — 기준 `PRD.md:412-417`. 목표: 같은 도로 지나는 서로 다른 업로더 영상 **2개 이상** + 야간/우천 1~2개. `sources.csv`에 URL·업로더·라이선스·도로명 기록 | `sources.csv` 5행 이상, 이슈 #13 확정 (11/14) |
| 0.6 | 정현우 | `fetch.py`로 받기 + 1080p/30fps 정규화. 라이선스 확인(CC 우선, 아니면 업로더 허락), 영상은 레포에 올리지 않음 | `data/videos/`에 파일, 길이·fps 로그 |
| 0.7 | 정현우 | **GPS 재구성** (`PRD.md:419-424`): gpx.studio로 경로 → 기준점 영상당 **8개 이상**(교차로·육교·표지판) → `reconstruct.py` | `<video_id>.gps.csv` 생성, `preview_map.py`로 경로가 도로 위에 있음 |
| 0.8 | 정현우 | `holdout_error.py` — 기준점을 하나씩 빼고 보간 위치 오차(m) 측정 | 영상별 평균·최대 오차 표 (`experiments/gps_error.csv`) |
| 0.9 | 정현우 | `docker-compose.yml` + `001_init.sql` | `psql`로 테이블·PostGIS 확인 |
| 0.10 | 공동 | 선정 영상에 **수동 라벨** `gt_events/<video_id>.csv` (시각·유형). 같은 도로 영상끼리는 **같은 결함에 같은 ID** | 결함 20건 이상, 두 영상에 공통으로 보이는 결함 5건 이상 |

> 영상·GPS가 모든 작업의 선행 조건이다 (`PRD.md:648`). 11/15까지 "같은 도로 2개"를 못 구하면 **한 영상 + 좌표 노이즈** 대체안(`PRD.md:645`)으로 확정하고 넘어간다.

### M1 골격 — 11/16(월) ~ 11/18(수)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 1.1 | 정현우 | **걷는 골격 (11/16 목표)**: `main.py --detector fake` → 업로드 → DB 저장. 가짜 탐지기는 `gt_events` CSV를 탐지처럼 내보냄 | 영상 1개 실행 후 `observations` 행 = 라벨 수 |
| 1.2 | 정현우 | `/trips`, `/trips/{id}/end`, 차량별 구분 (FR-2.2), `--gps-noise-m` | `run_fleet.py --n 3` 후 차량별 행 수 일치 |
| 1.3 | 정현우 | `clipper.py`, 전송 바이트 집계 (기준값 2가지, 2장 표) | 로그에 원본 크기 vs 전송량 |
| 1.4 | 박겸 | 탐지 v1 학습 (RDD 2클래스) → 데이터 도착 시 v2 (+AI Hub 도로장애물, +KRID 4클래스) | val mAP@0.5 기록 |
| 1.5 | 박겸 | `export_onnx.py` + `onnx_detector.py` | 같은 이미지 10장에서 PyTorch·ONNX 박스 일치 |
| 1.6 | 공동 | **11/18 인터페이스 맞춤**: `--detector onnx`로 선정 영상 실행 | 탐지 박스 오버레이가 보임 |

**11/18 Go/No-go** (`PRD.md:620`): ONNX가 안 돌면 PyTorch로 같은 `Detector` 프로토콜을 구현해 연결, ONNX는 속도 측정용.

### M2 통합 — 11/19(목) ~ 11/21(토)

| # | 담당 | 작업 | 완료 확인 |
|---|---|---|---|
| 2.1 | 정현우 | `quality.py` (밝기 평균, 라플라시안 분산) | 야간/우천 영상 구간에서 `detect ran:false` (F3) |
| 2.2 | 정현우 | `scheduler.py` — 품질 매 프레임, 탐지 N프레임, 차선 M프레임 | 로그상 단계별 실행 비율 |
| 2.3 | 정현우 | `track_merge.py` — 같은 클래스·IoU>0.3이 T초 안에 이어지면 1건 | 같은 포트홀 → 관측 1건 |
| 2.4 | 정현우 | `clustering.py` — `ST_DWithin(20m)` 같은 유형에 붙이거나 신규, `vehicle_count`·대표 좌표 갱신. 한 트랜잭션 + `FOR UPDATE` | 테스트: 3대 같은 지점 → 결함 1건, vehicle_count=3 |
| 2.5 | 정현우 | `risk.py` | 모든 결함에 risk_level (F4) |
| 2.6 | 박겸 | `lane_wear.py` — 하단 ROI → 흰/노랑 마스크 → 차선 대 노면 밝기 대비 | 선정 영상 수동 라벨 대비 정확도 |
| 2.7 | 박겸 | 파손 이미지 수집(공개 영상 프레임·공개 이미지, `PRD.md:371`) + 분류 학습 시도 | 이미지 수, 홀드아웃 정확도 |
| 2.8 | 공동 | **11/21 첫 end-to-end** — `e2e_smoke.py` | 영상 2개(또는 노이즈 3대) → DB에서 공통 결함이 묶임, 라벨 대비 결함 수 출력 |

**11/20 Go/No-go** (`PRD.md:621`): 파손 이미지가 클래스당 100장 미만(제안값)이면 위치 탐지만으로 확정.
**11/21 Go/No-go** (`PRD.md:622`): E2E가 늦으면 위험도는 유형만으로.

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
| 4.1 | 박겸 | `eval_detector.py` — 공개 테스트셋 / **선정 공개 영상** 분리 평가 (`PRD.md:558`) | 12장 표 탐지 행 |
| 4.2 | 박겸 | `bench_cpu.py` — 1·2코어 × 주기 적용 전/후 | FPS 표 |
| 4.3 | 박겸 | `interval_sweep.py` — 간격 1/3/5/10/15 × 누락률 | 그래프 |
| 4.4 | 공동 | 노출 기준 θ, k 결정 (#3) | `server/app/config.py`에 값 |
| 4.5 | 정현우 | 통신량 절감률(기준값 2가지), 중복 정리 비율, **반경 10/20/30 m별 묶임 비율** | 12장 표 + #2 확정 |
| 4.6 | 정현우 (P2) | BDD100K 클립 몇 개로 실제 GPS 경로 동작 확인 | `bdd_to_csv.py` → 같은 파이프라인 실행 성공 |

### 동결·리허설 — 11/25(수) ~ 11/27(금)

| 날짜 | 작업 | 완료 확인 |
|---|---|---|
| 11/25 | 기능 동결, 버그 수정, 12장 표 채우기 | 12장 표에 TBD 없음 |
| 11/26 | 리허설: `reset_demo.py` → `run_fleet.py` → 시민 화면 → 지도. 실시간·replay 두 경로. 발표 자료에 **영상 출처 표기**와 "GPS는 재구성했고 오차는 X m" 명시 (`PRD.md:417`, `PRD.md:426`) | 끊김 없는 시연 2회 이상 (`PRD.md:612`) |
| 11/27 | 완성 | — |

---

## 6. 수용 기준 (테스트 가능한 형태)

| ID | 기준 | 확인 방법 |
|---|---|---|
| AC-1 | 선정 공개 영상에서 4클래스 각각 1건 이상 탐지 (대체안 시 시설물은 위치만) (`PRD.md:227`) | `e2e_smoke.py` 클래스별 결함 수 |
| AC-2 | CPU 1·2코어 제한에서 시연 영상 처리 완료, 주기 적용 전/후 FPS 비교 표 | `experiments/bench_cpu.csv` |
| AC-3 | 품질 불량 구간에서 탐지 미실행 로그 + 적용 전/후 오탐 수 비교 | metrics JSONL + `experiments/quality_ab.csv` |
| AC-4 | 가상 블랙박스 3대 동시 실행 시 전송 관측 수 = 저장 관측 수 | `e2e_smoke.py` assert |
| AC-5 | 통신량 절감률이 기준값 2가지로 계산되어 있음 | uploader 집계 |
| AC-6 | 서로 다른 영상(또는 노이즈 차량)에 공통으로 보이는 라벨 결함 중 하나로 묶인 비율 제시 | `experiments/dedup.csv` |
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
| **같은 도로를 지나는 영상 2개를 못 구함** | 11/15까지 후보 1개 | 한 영상을 여러 가상 블랙박스에 넣고 `--gps-noise-m 5~10` (`PRD.md:645`). 발표에서 방식 명시 |
| **두 영상의 촬영 시점이 달라 결함이 다름** (보수됨·새로 생김) | 공통 라벨 결함 5건 미만 | 공통 결함만으로 중복 정리 비율 계산, 나머지는 "단독 관측"으로 분리 집계 |
| **재구성 GPS 오차가 큼** | hold-out 평균 오차 > 20 m | 기준점 추가(가로등·건물 입구), 속도 표시 영상은 OCR 누적 거리(`PRD.md:424`), 반경 조정 |
| **영상 라이선스** | CC 영상 없음 | 업로더에게 허락 요청, 영상은 레포·발표 자료에 원본 배포 안 하고 캡처만 출처와 함께 사용 |
| **유튜브 영상 압축·화각 문제** (노면이 뭉개짐, 화면 오버레이) | 탐지 결과가 공개셋보다 크게 낮음 | 1080p 이상만 선정, 화면 하단 오버레이 영역 마스킹, AI Hub 국내 데이터로 보강 |
| **RDD2022(해외 도로) → 국내 도로 성능 저하** | 선정 영상 평가 mAP 저조 | AI Hub 도로장애물/표면 데이터로 v2 학습 (`PRD.md:373`), 시간 되면 선정 영상 프레임 일부 라벨링해 파인튜닝 |
| **AI Hub 데이터(KRID·도로장애물) 승인 지연·대용량** | 11/16까지 미도착 | RDD 2클래스로 진행, 도착 즉시 클래스당 수천 장만 샘플링 (`PRD.md:649`) |
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
5. **시연 체크리스트**: PRD 13장 1~8단계(`PRD.md:572-582`)를 실시간·replay 두 경로로 통과.

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
| 2 | 묶기 반경 | 20 m 시작, 10/20/30 m + 재구성 오차로 확정 | 11/21 |
| 5 | 크기→위험도 | 유형만 (시간 되면 하단 노면 기준 정규화) | 11/21 |
| 6 | 차선 마모 확정 | 3회 이상 관측 + 대비 기준값은 실측 분포로 | 11/22 |
| 9 | 보수 중 전환 | 관리자 지도에서 수동 | 11/22 |
| 3 | 노출 기준 θ, k | 선정 영상 정밀도 기반 | 11/24 |

---

## 10. 바로 시작할 일 (오늘·내일)

- [ ] 박겸: **AI Hub KRID + 도로장애물/표면 인지 영상(수도권) 다운로드 신청**, RDD2022 다운로드 시작
- [ ] 정현우: 유튜브 영상 후보 검색 (`PRD.md:416` 검색어), `sources.csv` 작성 → 11/14 #13 확정
- [ ] 공동: 2장 기술 결정 확인 → 레포 생성 + `contracts/` 작성

---

## 변경 이력

- **r2 (2026-11-13)** — PRD v0.3 반영: 직접 촬영 → 공개 영상 + GPS 재구성(`tools/video`, `tools/gps`, M0 0.5~0.8), GPS CSV 계약, AI Hub 도로장애물 데이터 추가, BDD100K 보조 경로(4.6), 다중 차량 대체안(`--gps-noise-m`), 반경 20 m 시작, 재구성 오차 측정(AC-13), 통신량 기준값 2가지, 영상 관련 리스크 5건, 이슈 #13. r1의 "RDD2022 한국 서브셋" 오류 삭제.
- r1 (2026-11-13) — PRD v0.2 기준 최초 작성.
