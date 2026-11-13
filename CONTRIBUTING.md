# 협업 규칙

> 두 사람·2주 프로젝트에 맞춘 가벼운 규칙. 확정된 규칙이다 (2026-11-14). 바꿀 일이 생기면 이 파일을 고치고 상대에게 알린다.

## 1. 역할과 담당 폴더

| | 박겸 (`ParkGyeom`) — AI | 정현우 (`hunywoo`) — 백엔드 |
|---|---|---|
| 폴더 | `ai/`, `docs/ai-handoff.md` | `edge/`, `server/`, `ui/`, `tools/`, `docker-compose.yml` |
| 공동 | `contracts/`, `PRD.md`, `docs/plan.md`, `scripts/`, 루트 설정(`pyproject.toml`, `uv.lock`, `.gitignore`, `.github/`), `README.md`, `CLAUDE.md`, 이 파일 | ← 같음 |

**상대 담당 폴더는 고치지 않는다.** 고칠 게 보이면 이슈나 메시지로 알린다.

## 1-1. 각자 자기 역할의 작업만 한다

폴더뿐 아니라 **작업 자체**도 나눈다. 상대 작업을 대신 하면 같은 일을 두 번 하거나, 상대가 모르는 방식으로 만들어져 다시 맞춰야 한다.

| 박겸이 하는 작업 (AI) | 정현우가 하는 작업 (백엔드) |
|---|---|
| 데이터셋 다운로드·변환·라벨링 (RDD2022, AI Hub) | 공개 주행 영상 선정·다운로드, GPS 경로 재구성 |
| 통합 탐지 모델 학습·ONNX 변환 (`OnnxDetector`) | 가상 블랙박스 (프레임 추출, GPS 동기화, 분석 주기, 결과·클립 전송) |
| 시설물 파손 분류 모델 | 영상 품질 판별 |
| 차선 마모 판별 (OpenCV) | 가짜·재생(replay) 탐지기 등 블랙박스 쪽 도구 |
| 모델 성능 평가, CPU 속도 측정, 분석 간격 실험 | 서버·DB (수집 API, 중복 묶기, 위험도, 신고서, 포상 집계) |
| 모델 사용법 문서 ([docs/ai-handoff.md](docs/ai-handoff.md)) | 시민 확인·신고 화면, 관리자 지도 |
| | 시연 영상 수동 라벨 정리, 통신량·중복 정리 실험 |

함께 하는 것: `contracts/` 변경, PRD·계획 갱신, 노출 기준 θ·k 결정(#3), 위험도 크기 반영(#5), 리허설.

**규칙**

1. **상대 작업이 막혀서 내가 기다리게 돼도 대신 만들지 않는다.** 무엇이 필요한지, 언제까지 필요한지 상대에게 알린다. 그동안은 계약(`contracts/`)에 맞춘 내 쪽 작업을 먼저 한다.
2. **도와주고 싶으면 먼저 묻는다.** 상대가 "그거 해 줘"라고 한 것만 한다. 그 작업은 넘겨받은 사람 담당이 되고, 계획서([docs/plan.md](docs/plan.md)) 담당자 칸도 바꾼다.
3. **상대 결과물에서 문제를 찾으면 고치지 말고 알린다.** 재현 방법이나 로그를 같이 보낸다.
4. 표에 없는 새 작업이 생기면 누가 할지 먼저 정하고 이 표에 추가한다.
5. **Claude Code 도 같은 규칙을 따른다.** 내 Claude 에게 상대 작업을 시키지 않는다 (CLAUDE.md 에도 적혀 있다).

## 2. 브랜치와 PR

**모든 변경은 브랜치 → PR → CI 통과 → 머지.** `main` 에 직접 push 하지 않는다.
이유: CI(테스트)가 `main` 에 들어가기 **전에** 돌아서 `main` 은 항상 돌아가는 상태로 남고, 기능마다 PR 기록이 남는다.

| 바꾸는 곳 | 머지 조건 |
|---|---|
| 내 담당 폴더, 문서(PRD·계획·규칙), 루트 설정 | **CI 통과하면 본인이 바로 머지** (리뷰 기다리지 않음) |
| **`contracts/`, `server/sql/`** (계약·DB 스키마) | CI 통과 + **상대 승인** 후 머지. `CODEOWNERS` 가 리뷰를 자동 요청한다 |

- 문서·규칙을 바꾼 PR 은 머지한 뒤 상대에게 한 줄로 알린다.
- 브랜치 이름: `ai/<내용>`, `be/<내용>`, `docs/<내용>` (예: `ai/v2-dataset`, `be/clustering`).
- PR 은 작게: 하루 작업 이하, 한 가지 목적. 커지면 나눠서 올린다.
- 머지는 **squash** (PR 하나 = `main` 커밋 하나)로 하고, 머지한 브랜치는 지운다.
- **11/25(수) 기능 동결** 이후에는 버그 수정 PR 만 (계획 15.1).

```bash
git switch main && git pull                 # 1. 최신 main 에서 시작
git switch -c ai/v2-dataset                 # 2. 브랜치
# ... 작업, uv run pytest -q, git commit ...
git push -u origin ai/v2-dataset            # 3. push
gh pr create --fill                         # 4. PR (제목·본문은 커밋 메시지로)
gh pr checks --watch                        # 5. CI 통과 확인
gh pr merge --squash --delete-branch        # 6. 머지 (contracts/·server/sql/ 이면 상대 승인 후)
git switch main && git pull                 # 7. 로컬 main 갱신
```

Claude Code 에 "PR 올리고 CI 통과하면 머지해줘"라고 하면 위 순서대로 한다 (CLAUDE.md).

## 3. 커밋 메시지

- 첫 줄: 무엇을 왜 바꿨는지 한 줄 (한국어·영어 모두 가능). 예: `Cap train background images at 10%`
- 공동 파일을 바꿨으면 본문에 영향을 적는다. 예: `contracts: ObservationIn 에 heading 필드 추가 (기본값 None, 기존 코드 영향 없음)`

## 4. 공동 파일 변경

### 계약 (`contracts/`)
1. 바꾸려는 이유와 내용을 먼저 상대에게 말한다.
2. PR 로 올리고 CI 통과 + 상대 승인 후 merge.
3. `CLASS_NAMES` 순서 변경 = 모델 재학습. 11/18 이후에는 하지 않는다.

### PRD·계획
- 오픈 이슈를 결정하면 PRD 18장 표와 해당 요구사항의 `TBD` 를 고치고, 문서 버전을 올린다.
- 일정·작업이 바뀌면 `docs/plan.md` 해당 표와 맨 아래 "변경 이력"에 한 줄 남긴다.

## 5. 파일 공유

| 무엇 | 어디 | git |
|---|---|---|
| 코드·문서·노트북 | 이 레포 | O |
| 학습 데이터 (`parts/*.zip`), 체크포인트 (`runs/`) | 박겸 Google Drive `daboyeo_data/` | X |
| 모델 (`detector_vN.onnx` + `.json`) | Drive `daboyeo_data/models/` → 받아서 `ai/models/` | `.json` 만 O |
| 시연 영상·GPS CSV | 정현우가 정함 (`data/` 아래, git 제외) | X |
| 실험 결과 CSV·그래프 | `experiments/` | O |

- 영상은 라이선스 때문에 레포에 올리지 않는다. 출처는 `tools/video/sources.csv` 에 남긴다 (PRD 8.5).
- 비밀값(API 키·토큰)은 커밋하지 않는다. Colab 은 🔑 보안 비밀, 로컬은 `.env` (git 제외).

## 6. 체크포인트

| 날짜 | 함께 확인할 것 |
|---|---|
| 11/18(수) | 인터페이스 맞춤 — v1 ONNX 를 가상 블랙박스에 연결 ([docs/ai-handoff.md](docs/ai-handoff.md)) |
| 11/21(토) | 첫 end-to-end — 영상 → 탐지 → 서버 DB 에서 결함이 묶임 |
| 11/24(화) | 노출 기준 θ·k 결정 (오픈 이슈 #3) |
| 11/25(수) | 기능 동결 |
| 11/26(목) | 리허설 2회 |

막히면 혼자 오래 붙잡지 말고 바로 공유한다. 일정이 밀리면 PRD 15.3 컷 라인 순서대로 줄인다.

## 7. Claude Code 를 쓸 때

- 레포 루트의 [CLAUDE.md](CLAUDE.md) 를 Claude 가 자동으로 읽는다. 담당 폴더 규칙도 거기 있다.
- Claude 가 만든 변경도 내가 만든 것과 같이 취급한다: 테스트 통과, 담당 폴더 규칙, 브랜치 → PR → CI → 머지.
