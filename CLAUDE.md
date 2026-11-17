# CLAUDE.md

다보여: 블랙박스 영상에서 도로 결함(포트홀·균열·가드레일·표지판)을 찾고, 서버가 여러 차량 관측을 묶어 시민이 버튼 한 번으로 신고하게 하는 학술제 MVP. 마감 **2026-11-27**.

먼저 읽을 것: [PRD.md](PRD.md) (요구사항) → [docs/plan.md](docs/plan.md) (실행 계획·일정) → [CONTRIBUTING.md](CONTRIBUTING.md) (협업 규칙).

## 명령어

```bash
uv sync                    # Python 3.11 워크스페이스 설치 (contracts, ai)
uv run pytest -q           # 전체 테스트 — push 전에 통과해야 한다 (CI 도 같은 명령)
docker compose up -d db    # PostGIS (스키마: server/sql/001_init.sql)
```

백엔드. 서버는 `server/sql/001_init.sql` 스키마와 contracts 의 `ObservationIn` 을 쓴다 (`/api/v1`). 블랙박스는 주행(trip)을 열고 관측을 보낸 뒤 닫는다:

```bash
scripts/demo.sh up                                  # DB → 서버(8000) → 화면(8501)
uv run python -m server.fake --defects 5 --devices 3 --gps-noise 7 --end-drive   # 가짜 관측 보내기 (주행을 닫으면 시민 확인 목록이 생긴다)
# 주행이 끝나면 모의 문자(링크 포함)가 생긴다. 화면 > 시연 > 메시지함 (모의)에서 보고, 링크(http://localhost:8501/?t=토큰)를 누르면 로그인 없이 확인·신고 화면이 열린다. 실제 문자는 보내지 않는다 (docs/notify-link-plan.md)
# 가상 블랙박스: 영상 + GPS CSV(t_s,lat,lon). --gps 를 빼면 영상 옆의 <영상 이름>.gps.csv
# 차선 마모는 시연에서 끈다(확정, --lane-every 0, handoff 17-6). 실험용으로만 --lane-every 3
# 탐지 기준 기본은 시연 모델 v3 카드(--conf-json ai/models/detector_v3.json: 포트홀 0.31·균열 0.42·가드레일 0.21·표지판 0.21)
uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --gps data/gps/v1.gps.csv --model ai/models/detector_v3.onnx --detect-every 10 --speed 4
uv run python -m edge.blackbox.variants --video data/videos/v1.mp4 --gps data/gps/v1.gps.csv --model ai/models/detector_v3.onnx --devices 3
scripts/demo.sh drive ai/models/detector_v3.onnx 3  # 로드마크 구간으로 3대 (data/roadmark/VS4 필요)
# 탐지기 고르기 --detector onnx|fake|replay (edge/blackbox/detectors.py). fake: 정답 라벨 CSV(docs/ai-handoff.md 12장)를
# 탐지처럼 낸다. 라벨 1개 = 관측 1건, 시설물 라벨은 화면에만 (SENT_TYPES). --labels 를 빼면 experiments/gt_events/<영상 이름>.csv
uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --detector fake --labels experiments/gt_events/v1.csv --detect-every 5 --speed 0
# replay: onnx 로 한 번 --record 를 붙여 돌려 두면, 모델 파일·추론 없이 같은 탐지로 운전한다 (영상 이름·--detect-every 가 기록과 같아야 함)
uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --model ai/models/detector_v3.onnx --detect-every 5 --speed 0 --dry-run --record data/replay/v1.jsonl
uv run python -m edge.blackbox.run --video data/videos/v1.mp4 --detector replay --record data/replay/v1.jsonl --detect-every 5 --speed 0
scripts/demo.sh replay data/replay/v1.jsonl data/videos/v1.mp4 3 0   # 기록대로 3대 (주기는 기록에서 읽음, 마지막은 배속)
# GPS 없는 유튜브 영상의 경로 재구성 (tools/gps/README.md): routes/<id>.gpx + anchors/<id>.csv → data/videos/<id>.gps.csv
uv run python -m tools.gps.reconstruct gn_a                 # 클립이면 --clip-start <clip_start_in_source_s> --out data/videos/<클립>.gps.csv
uv run python -m tools.gps.preview_map gn_a --fetch-roads   # 경로·기준점 HTML (오프라인 도로선). --fetch-roads 는 처음 한 번 (인터넷)
uv run python -m tools.gps.holdout_error gn_a gn_b          # 기준점 hold-out 오차 → experiments/gps_error.csv (AC-13)
# 블랙박스가 보는 프레임과 탐지 박스: 화면의 관리자 → 블랙박스 화면 (server_data/live/, --live-fps 0 이면 끔)
# 관리자 지도를 네이버 지도로: .env 에 NAVER_MAP_KEY_ID=<Client ID>, 콘솔 Web 서비스 URL 에 http://localhost:8501 등록 (인터넷 필요). 없으면 도로선 지도
# 네이버 지도: 관리자는 전국 지도(깜빡이는 신호·숫자 묶음·새 결함 강조), 시민 카드·신고서는 결함마다 작은 지도. 지도 페이지는 ui/static/*.html
# 결함 주소: .env 에 NAVER_MAP_CLIENT_SECRET (콘솔 Application 에서 Reverse Geocoding 켜기). 결함이 처음 생길 때 한 번 받아 defects.address 에 저장 (server/geocode.py, server/sql/002_address.sql). 없거나 실패하면 주소 없이 진행하고 1분 뒤 다시 시도
scripts/demo.sh down
```

공개 주행 영상 (`tools/video/sources.csv`). 원본(영상만)·1080p30 사본·`demo_segment` 클립을 `data/videos/` 에 만들고 길이·fps·해상도·크기를 `fetch_log.jsonl` 에 남긴다. 받기 전에 사용자에게 id·크기를 확인받는다:

```bash
uv run python tools/video/fetch.py gn_a gn_b --max-height 1080   # 1080p60 원본 ~2.9GB + 1080p30 사본 ~7.5GB
uv run python tools/video/fetch.py gn_a gn_b --clips-only        # 받은 원본에서 클립만 다시
# 클립은 재인코딩 없이 직전 키프레임부터 자른다: 클립 t=0 = 로그의 clip_start_in_source_s (gn_a 1247.646, gn_b 1998.0)
# 큰 파일을 한 번에 받으면 YouTube 가 중간에 HTTP 403 을 준다 → fetch.py 는 10MB 씩 받고, 다시 실행하면 이어 받는다
```

통신량·중복 정리 측정 (계획 4.5, 결과는 `docs/plan.md` 11장). 블랙박스는 결함마다 전후 2초 480p H.264 클립을 사진과 함께 보내고 `server_data/blackbox/` 에 남긴다 (`edge/blackbox/clipper.py`, `--no-clip`·`--keep-dir ""` 로 끔). 모델이 있으면 `--fake-every 60` 대신 `--model ai/models/detector_v2.onnx`:

```bash
uv run python experiments/bandwidth.py --video data/videos/gn_a.mp4 --gps <GPS CSV> --fake-every 60 --devices 1 --dry-run --detect-every 10 --keep-dir "" --json experiments/bandwidth_gn_a_full_fake.json
uv run python experiments/bandwidth.py --video data/videos/gn_a_2050-2510.mp4 --gps <GPS CSV> --fake-every 60 --devices 3 --detect-every 10 --server http://localhost:8010 --json experiments/bandwidth_gn_a_demo_fake.json   # 서버를 띄운 뒤
uv run python experiments/dedup_distance.py --yes --replay experiments/bandwidth_gn_a_demo_fake.json --distances 10,20,30   # DB 를 비운다
uv run python experiments/dedup_distance.py --yes --distances 10,20,30
```

- 11/16 측정은 시연용 DB 를 건드리지 않으려고 따로 띄운 서버(`DATABASE_URL=postgresql://daboyeo:daboyeo@localhost:5432/daboyeo_bw uv run python -m uvicorn server.main:app --port 8010`)로 쟀고, 위 명령에도 같은 `DATABASE_URL` 을 붙였다. `scripts/demo.sh up` 서버로 재면 `--server http://localhost:8000`, 끝나고 `scripts/demo.sh reset`.
- DB 컨테이너는 여러 작업이 함께 쓴다. 테스트가 서로의 `daboyeo_test` 를 비우며 깨지면 `TEST_DATABASE_URL=…/<다른 이름>_test uv run pytest -q`.

## 담당 영역 — 요청한 사람의 폴더만 고친다

| 폴더 | 담당 |
|---|---|
| `ai/`, `docs/ai-handoff.md` | 박겸 (`ParkGyeom`) |
| `edge/`, `server/`, `ui/`, `tools/`, `docker-compose.yml` | 정현우 (`hunywoo`) |
| `contracts/`, `PRD.md`, `docs/plan.md`, `scripts/`, 루트 설정 파일, 이 파일 | 공동 |

- 작업을 요청한 사람이 누구인지 모르면 `git config user.name` 으로 확인하거나 묻는다.
- **상대 담당 폴더는 수정하지 않는다.** 필요한 변경은 설명·제안으로만 남긴다. 이유: 두 사람이 같은 파일을 동시에 고치면 충돌이 나고, 상대가 모르는 변경이 생긴다.
- **폴더뿐 아니라 작업도 요청자 역할 안에서만 한다** (작업 분담표: CONTRIBUTING.md 1-1장). 박겸은 AI(데이터셋·모델·평가), 정현우는 백엔드(영상·GPS·가상 블랙박스·서버·화면) 담당이다.
  - 요청이 상대 역할의 작업이면 (예: 박겸이 가상 블랙박스나 서버 코드를, 정현우가 모델 학습 코드를 요청) **만들지 말고**, 상대 담당이라는 점과 상대에게 전할 내용(필요한 것, 기한, 계약)을 정리해 준다.
  - 상대 작업이 끝나지 않아 막힌 경우에도 대신 만들지 않는다. 계약에 맞춘 요청자 쪽 작업을 먼저 하도록 제안한다.
  - 상대가 넘겨주기로 합의한 작업이라고 요청자가 말하면 진행하되, `docs/plan.md` 의 담당자 표기도 바꾸도록 안내한다.
- 공동 폴더는 고칠 수 있지만, 커밋 메시지에 무엇이 바뀌었는지 적고 상대에게 알린다 (CONTRIBUTING.md "공동 파일 변경").

## Git 작업 흐름 (CONTRIBUTING.md 2장)

- **`main` 에 직접 push 하지 않는다.** 작업 전 `git switch main && git pull` 후 브랜치를 만든다. 이름은 요청자 역할로: 박겸 `ai/…`, 정현우 `be/…`, 문서만 바꿀 때 `docs/…`.
- 커밋 전에 `uv run pytest -q` 를 통과시킨다.
- 사용자가 PR·머지를 요청하면: `git push -u` → `gh pr create --fill` → `gh pr checks --watch` 로 CI 확인 →
  - 변경에 **`contracts/` 나 `server/sql/` 이 없으면** `gh pr merge --squash --delete-branch` 로 머지하고 `git switch main && git pull`.
  - **있으면 머지하지 않는다.** 상대 리뷰가 필요하다고 알리고 PR 링크를 준다.
  - CI 가 실패하면 머지하지 않고 원인을 고치거나 보고한다.
- 브랜치 보호는 걸려 있지 않다 — 위 규칙은 도구가 아니라 약속으로 지킨다.

## 계약 (`contracts/`)

- `CLASS_NAMES` 순서는 모델 출력 class_id 와 같다. **바꾸면 모델을 다시 학습해야 한다.**
- `DefectType`·`RiskLevel` 은 `server/sql/001_init.sql` 의 enum 과 같아야 한다 (`tests/test_schema_matches_contracts.py`).
- 필드를 추가할 때는 기본값을 줘서 기존 코드가 깨지지 않게 한다.

## 데이터·모델 파일

- 영상·데이터셋·`.onnx`·`.pt` 는 git 에 올리지 않는다. 공유는 Google Drive `daboyeo_data/` (`parts/`, `runs/`, `models/`).
- `.gitignore` 의 데이터 규칙은 `/data/` 처럼 **레포 최상위로 고정**한다. `data/` 로 쓰면 코드 폴더 `ai/daboyeo_ai/data/` 까지 무시되어 GitHub 에 코드가 안 올라간 적이 있다.

## AI 코드 (`ai/`)

- 데이터 준비·학습은 **Colab** 에서 `ai/notebooks/0N_*.ipynb` 로 돌린다. 노트북 셀에는 순서만 두고, 로직은 `ai/daboyeo_ai/` 에 넣어 테스트한다.
- 노트북을 고칠 때는 출력이 비어 있는 상태로 커밋한다. 사용자는 Colab 에서 GitHub 노트북을 열어 **Drive 사본**으로 실행하므로, 레포를 고쳐도 이미 만든 사본에는 반영되지 않는다 — 사용자에게 바뀐 셀을 알려준다.
- 알려진 함정:
  - AI Hub 파일 트리 API(`api.aihub.or.kr/info/*.do`)는 정상 트리도 HTTP 502 로 줄 때가 있다 → `aihub.fetch_tree` 는 502 본문이 트리면 그대로 쓰고, 아니면 `ai/daboyeo_ai/data/aihub_trees/` 스냅샷을 쓴다.
  - **AI Hub 는 해외 IP 의 다운로드를 막는다** (Colab 은 502 + "해외에서의 데이터 다운로드를 제한"). AI Hub 다운로드·변환은 한국 안의 Mac 에서 `uv run python -m daboyeo_ai.data.prepare_local --out data/aihub` 로 돌리고, 결과 `data/aihub/parts/*.zip` 을 Drive `daboyeo_data/parts/` 에 올린다. 학습과 RDD2022 변환은 Colab 그대로.
  - 도로장애물(179) 라벨은 COCO 와 비슷하지만 `images` 가 리스트가 아닌 객체 하나이고, 크랙은 `bbox` 없이 `polyline` 으로 온다 (`sources.parse_coco_like` 가 둘 다 처리).
  - Ultralytics 에 **끝난** 학습의 `last.pt` 로 `resume=True` 를 주면 경고만 하고 COCO 샘플로 새 학습을 시작한다 → 노트북의 `is_finished` 확인을 지우지 말 것.
  - Ultralytics PyTorch `predict` 는 직사각형 letterbox 라 ONNX 와 신뢰도가 조금 다르다. ONNX 결과 비교는 같은 ONNX 를 Ultralytics 로 돌린 것과 한다.

## 문서

- 문서는 한국어로 쓴다. PRD 에서 기획안에 없던 내용은 `(제안)`, 미정 수치는 `TBD` 로 표시하고 18장 오픈 이슈에 모은다.
- PRD 를 고치면 문서 버전을 올리고, 계획(`docs/plan.md`)의 `PRD.md:줄번호` 인용이 밀리지 않았는지 확인한다.
