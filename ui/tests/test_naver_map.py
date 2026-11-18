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
        naver_map.mini(37.49899, 127.02713, "crack")

    at = AppTest.from_function(app).run()
    assert not at.exception
    # 받아 둔 도로가 없으면 좌표 글자, 있으면 도로선 그림 + 출처
    texts = [c.value for c in at.caption]
    assert any("37.49899" in t or "OpenStreetMap" in t for t in texts)


def test_short_place_keeps_road_and_number_without_id():
    assert naver_map.short_place("서울 강남구 봉은사로 102") == "봉은사로 102"
    assert naver_map.short_place("서울 강남구 테헤란로8길 21") == "테헤란로8길 21"
    assert naver_map.short_place("경기 성남시 분당구 정자동 1-2") == "정자동 1-2"  # 지번 주소
    assert naver_map.short_place(None) == "위치 확인 중"
    assert naver_map.short_place("") == "주소 없는 곳"
    row = {"id": "be844d5a-0000", "defect_type": "crack", "address": "서울 강남구 강남대로 396"}
    assert naver_map.title(row) == "강남대로 396 균열"


def test_type_colors_match_blackbox_live_view():
    from edge.blackbox.live import COLORS

    names = {"guardrail": "guardrail_damage", "sign": "sign_damage"}
    for t, (b, g, r) in COLORS.items():
        assert naver_map.TYPE_COLOR[names.get(t, t)] == f"#{r:02x}{g:02x}{b:02x}"
