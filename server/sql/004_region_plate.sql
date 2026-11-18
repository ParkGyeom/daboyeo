-- 관리자 지도에서 지역·차량 번호로 나눠 보기 (제안, 정현우 → 박겸 리뷰 필요).
-- 여러 번 실행해도 안전하다: docker compose 는 새 DB 에서 001~003 다음에 실행하고, 서버(db.init_db)는 켤 때마다 실행한다.
-- 모두 비어 있어도 되는 열이라 기존 데이터는 그대로 남는다.

-- 결함 위치의 행정 구역. 주소(address)와 같이 네이버 Reverse Geocoding 으로 한 번 받아 저장한다 (server/geocode.py).
-- 도로명 주소("서울 강남구 강남대로 396")에는 동이 없어서 따로 둔다
ALTER TABLE defects ADD COLUMN IF NOT EXISTS region1 text;  -- 시·도      예: "서울특별시"
ALTER TABLE defects ADD COLUMN IF NOT EXISTS region2 text;  -- 시·군·구   예: "강남구"
ALTER TABLE defects ADD COLUMN IF NOT EXISTS region3 text;  -- 읍·면·동   예: "역삼동". 응답에 없으면 NULL
CREATE INDEX IF NOT EXISTS defects_region_idx ON defects (region1, region2, region3);

-- 차량 번호. 관리자가 차량을 번호로 구분한다. label 은 블랙박스 기기 이름(--device-id, bb-01)이라 그대로 둔다
ALTER TABLE vehicles ADD COLUMN IF NOT EXISTS plate text;     -- 예: "12가 3456" (시연 차량은 가짜 번호)
