"""메시지함이 모의 문자 본문을 보여 주는 방식 (ui/api.py sms_markdown)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 화면 코드는 ui/ 를 기준으로 import 한다

import api  # noqa: E402

URL = "http://localhost:8501/?t=Ab-c_9"


def test_lines_kept_link_clickable_and_symbols_literal():
    body = "[다보여] 도로 결함 확인 요청\n오후 12:54~12:58 주행\n· 균열 3건 — 봉은사로 102 부근\n확인·신고: " + URL
    md = api.sms_markdown(body, URL).split("  \n")
    assert md[0] == r"\[다보여\] 도로 결함 확인 요청"
    assert md[1] == r"오후 12:54\~12:58 주행"  # ~ 가 취소선이 되지 않게
    assert md[3].startswith("확인·신고: [") and md[3].endswith(f"]({URL})")


def test_blank_lines_split_paragraphs_and_link_alone_on_its_line():
    md = api.sms_markdown("제목\n\n▶ 확인·신고하기\n" + URL, URL)
    assert "\n  \n" in md  # 빈 줄 → 문단 사이 간격 (실제 문자처럼 덩어리로 나뉨)
    assert md.endswith(f"[{api._md_escape(URL)}]({URL})")


def test_body_without_link_still_renders():
    assert api.sms_markdown("한 줄", URL) == "한 줄"
