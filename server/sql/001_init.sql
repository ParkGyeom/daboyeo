-- 다보여 초기 스키마 (PRD 9장 데이터 모델 초안)
-- enum 값은 contracts/daboyeo_contracts/classes.py 와 같아야 한다 (tests/test_schema_matches_contracts.py 가 확인).

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TYPE defect_type AS ENUM ('pothole', 'crack', 'lane_wear', 'guardrail_damage', 'sign_damage');
CREATE TYPE risk_level AS ENUM ('urgent', 'routine');
CREATE TYPE defect_status AS ENUM ('unreported', 'reported', 'under_repair');
CREATE TYPE finding_response AS ENUM ('pending', 'reported', 'skipped', 'already_reported');
CREATE TYPE submission_status AS ENUM ('generated', 'mock_submitted');

-- 시민 (시연용 가상 사용자)
CREATE TABLE users (
    id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL  -- 신고서에 들어갈 본인 명의
);

-- 가상 블랙박스 1개 = 차량 1대
CREATE TABLE vehicles (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_user_id uuid NOT NULL REFERENCES users(id),
    label         text NOT NULL  -- 예: 차량 A
);

-- 주행 1회
CREATE TABLE trips (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    vehicle_id uuid NOT NULL REFERENCES vehicles(id),
    started_at timestamptz NOT NULL DEFAULT now(),
    ended_at   timestamptz  -- 채워지면 주행 후 확인 목록 생성
);

-- 중복을 정리한 결함 (지도·신고의 단위)
CREATE TABLE defects (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    defect_type          defect_type NOT NULL,
    location             geography(Point, 4326) NOT NULL,  -- 대표 좌표
    risk_level           risk_level NOT NULL,
    status               defect_status NOT NULL DEFAULT 'unreported',
    observation_count    int NOT NULL DEFAULT 0,
    vehicle_count        int NOT NULL DEFAULT 0,
    confidence           real NOT NULL,  -- 대표 신뢰도
    is_suspected         boolean NOT NULL DEFAULT false,  -- 시설물 '파손 의심'
    reward_eligible_road boolean NOT NULL DEFAULT false,  -- 포상 대상 도로
    first_seen_at        timestamptz NOT NULL,
    last_seen_at         timestamptz NOT NULL
);
CREATE INDEX defects_location_gix ON defects USING gist (location);
CREATE INDEX defects_type_status_idx ON defects (defect_type, status);

-- 블랙박스가 보낸 개별 관측
CREATE TABLE observations (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    client_observation_id uuid NOT NULL UNIQUE,  -- 재전송 중복 방지 (FR-2.4)
    trip_id               uuid NOT NULL REFERENCES trips(id),
    vehicle_id            uuid NOT NULL REFERENCES vehicles(id),
    defect_type           defect_type NOT NULL,
    confidence            real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    bbox                  jsonb,
    size_metric           real,
    is_suspected          boolean NOT NULL DEFAULT false,
    location              geography(Point, 4326) NOT NULL,  -- 탐지 시점 GPS 좌표
    t_s                   real NOT NULL,  -- 영상 시작 기준 초
    observed_at           timestamptz NOT NULL,
    quality_score         real,
    snapshot_path         text,
    clip_path             text,
    snapshot_bytes        int,  -- 통신량 집계용
    clip_bytes            int,
    defect_id             uuid REFERENCES defects(id)  -- 중복 묶기 결과
);
CREATE INDEX observations_location_gix ON observations USING gist (location);
CREATE INDEX observations_defect_idx ON observations (defect_id);
CREATE INDEX observations_trip_idx ON observations (trip_id);

-- 시민별 주행 후 확인 목록의 한 항목
CREATE TABLE findings (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    trip_id      uuid NOT NULL REFERENCES trips(id),
    user_id      uuid NOT NULL REFERENCES users(id),
    defect_id    uuid NOT NULL REFERENCES defects(id),
    response     finding_response NOT NULL DEFAULT 'pending',
    responded_at timestamptz,
    -- 같은 시민에게 같은 결함은 한 번만 묻는다 (오픈 이슈 #8 권장안: 넘기기 후 다시 묻지 않음)
    UNIQUE (user_id, defect_id)
);

-- 신고서
CREATE TABLE reports (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    defect_id         uuid NOT NULL UNIQUE REFERENCES defects(id),  -- 결함당 신고 1건 (FR-5.5)
    user_id           uuid NOT NULL REFERENCES users(id),  -- 본인 명의
    quarter           text NOT NULL,  -- 예: 2026-Q4
    payload           jsonb NOT NULL,  -- 사진·위치·유형·AI 신뢰도·관측 차량 수
    submission_status submission_status NOT NULL DEFAULT 'generated',
    reward_valid      boolean NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX reports_user_quarter_idx ON reports (user_id, quarter) WHERE reward_valid;
