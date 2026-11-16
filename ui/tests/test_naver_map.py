"""관리자 지도의 '새 결함' 강조 규칙과 키 없을 때의 작은 지도."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 화면 코드는 ui/ 를 기준으로 import 한다

import naver_map  # noqa: E402


def test_first_draw_is_not_fresh_then_new_ones_are():
    state = {}
    assert naver_map.fresh_ids([{"id": "a"}, {"id": "b"}], state, now=100.0) == set()
    assert naver_map.fresh_ids([{"id": "a"}, {"id": "b"}, {"id": "c"}], state, now=105.0) == {"c"}
    # 강조는 FRESH_S 초 동안만
    assert naver_map.fresh_ids([{"id": "c"}], state, now=105.0 + naver_map.FRESH_S - 1) == {"c"}
    assert naver_map.fresh_ids([{"id": "c"}], state, now=105.0 + naver_map.FRESH_S) == set()


def test_mini_without_key_falls_back_offline():
    from streamlit.testing.v1 import AppTest

    def app():
        import naver_map

        naver_map.key_id = lambda: None
        naver_map.mini(37.49899, 127.02713, "urgent")

    at = AppTest.from_function(app).run()
    assert not at.exception
    # 받아 둔 도로가 없으면 좌표 글자, 있으면 도로선 그림 + 출처
    texts = [c.value for c in at.caption]
    assert any("37.49899" in t or "OpenStreetMap" in t for t in texts)
