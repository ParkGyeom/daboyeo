-- 주행 종료 문자(모의 발송). 001_init.sql 위에 겹쳐 적용해도 되도록 모두 IF NOT EXISTS 로 쓴다 (init_db 가 항상 적용한다).
-- 실제 문자·메일은 보내지 않는다. 보낼 문자를 이 표에 저장하는 것이 발송이고, 화면의 메시지함이 그것을 보여 준다.

-- 기기 초기 설정 때 등록된 연락처 (시연은 가짜 번호)
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone text;

CREATE TABLE IF NOT EXISTS notifications (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES users(id),
    trip_id    uuid NOT NULL REFERENCES trips(id),
    to_phone   text NOT NULL,
    body       text NOT NULL,                 -- 보낸 문구 (링크 포함)
    token      text NOT NULL UNIQUE,          -- 링크의 열쇠. 가입·로그인 없이 이 값으로 그 시민의 확인 목록을 연다
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL,
    opened_at  timestamptz,
    UNIQUE (trip_id, user_id)                 -- 한 주행에 같은 시민에게는 한 번만
);
CREATE INDEX IF NOT EXISTS notifications_created_idx ON notifications (created_at DESC);
