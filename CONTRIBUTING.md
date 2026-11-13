# 협업 규칙

> 두 사람·2주 프로젝트에 맞춘 가벼운 규칙. **(제안)** 표시는 팀 합의 후 확정한다.

## 1. 역할과 담당 폴더

| | 박겸 (`ParkGyeom`) — AI | 정현우 (`hunywoo`) — 백엔드 |
|---|---|---|
| 폴더 | `ai/`, `docs/ai-handoff.md` | `edge/`, `server/`, `ui/`, `tools/`, `docker-compose.yml` |
| 공동 | `contracts/`, `PRD.md`, `docs/plan.md`, `scripts/`, 루트 설정(`pyproject.toml`, `uv.lock`, `.gitignore`, `.github/`), `README.md`, `CLAUDE.md`, 이 파일 | ← 같음 |

**상대 담당 폴더는 고치지 않는다.** 고칠 게 보이면 이슈나 메시지로 알린다.

## 2. 브랜치와 push `(제안)`

| 바꾸는 곳 | 방법 |
|---|---|
| 내 담당 폴더만 | `main` 에 바로 push 해도 된다 |
| **공동 파일** (특히 `contracts/`, DB 스키마) | 브랜치 → PR → **상대 승인 후** merge. `CODEOWNERS` 가 리뷰를 자동 요청한다 |

- push 전에 `uv run pytest -q` 통과. GitHub Actions(CI)도 push·PR 마다 같은 테스트를 돌린다.
- push 전에 `git pull --rebase` 로 최신을 받는다.
- 브랜치 이름: `ai/<내용>`, `be/<내용>`, `docs/<내용>` (예: `ai/v2-dataset`, `be/clustering`).
- **11/25(수) 기능 동결** 이후에는 버그 수정만 (계획 15.1).

## 3. 커밋 메시지

- 첫 줄: 무엇을 왜 바꿨는지 한 줄 (한국어·영어 모두 가능). 예: `Cap train background images at 10%`
- 공동 파일을 바꿨으면 본문에 영향을 적는다. 예: `contracts: ObservationIn 에 heading 필드 추가 (기본값 None, 기존 코드 영향 없음)`

## 4. 공동 파일 변경

### 계약 (`contracts/`)
1. 바꾸려는 이유와 내용을 먼저 상대에게 말한다.
2. PR 로 올리고 상대가 승인하면 merge.
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
- Claude 가 만든 변경도 내가 만든 것과 같이 취급한다: 테스트 통과, 담당 폴더 규칙, 공동 파일은 PR.
