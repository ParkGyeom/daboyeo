-- 결함 위치의 주소 (제안, 정현우 → 박겸 리뷰 필요).
-- 결함이 처음 생길 때 서버가 네이버 클라우드 Reverse Geocoding 으로 한 번 받아 저장한다.
-- 시연장에 인터넷이 없어도 화면·신고서에 주소가 보이게 하기 위해서다. 받지 못하면 비워 두고 나중에 다시 시도한다.
-- 여러 번 실행해도 안전하다: docker compose 는 새 DB 에서 001 다음에 실행하고, 서버(db.init_db)는 켤 때마다 실행한다.
ALTER TABLE defects ADD COLUMN IF NOT EXISTS address text;                -- 예: "서울 강남구 강남대로 396"
ALTER TABLE defects ADD COLUMN IF NOT EXISTS address_tried_at timestamptz; -- 마지막으로 주소를 시도한 시각
