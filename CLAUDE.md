# CLAUDE.md

다보여: 블랙박스 영상에서 도로 결함(포트홀·균열·가드레일·표지판)을 찾고, 서버가 여러 차량 관측을 묶어 시민이 버튼 한 번으로 신고하게 하는 학술제 MVP. 마감 **2026-11-27**.

먼저 읽을 것: [PRD.md](PRD.md) (요구사항) → [docs/plan.md](docs/plan.md) (실행 계획·일정) → [CONTRIBUTING.md](CONTRIBUTING.md) (협업 규칙).

## 명령어

```bash
uv sync                    # Python 3.11 워크스페이스 설치 (contracts, ai)
uv run pytest -q           # 전체 테스트 — push 전에 통과해야 한다 (CI 도 같은 명령)
docker compose up -d db    # PostGIS (스키마: server/sql/001_init.sql)
```

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
