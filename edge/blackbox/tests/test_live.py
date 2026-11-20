"""블랙박스 화면의 박스 라벨: 한글 글꼴이 있으면 한국어, 없으면 영어로 떨어진다."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edge.blackbox import live

HAS_FONT = any(Path(p).exists() for p in live.FONT_PATHS)


@pytest.fixture(autouse=True)
def _fresh_cache():
    live._label_patch.cache_clear()
    yield
    live._label_patch.cache_clear()


def _det(t="sign", conf=0.31, bbox=(100, 100, 200, 180)):
    return SimpleNamespace(type=t, confidence=conf, bbox=bbox)


def test_labels_cover_every_box_color():
    assert set(live.LABELS) == set(live.COLORS)


def test_falls_back_to_english_without_korean_font(monkeypatch):
    monkeypatch.setattr(live, "_font", lambda size: None)
    img = np.zeros((300, 400, 3), np.uint8)
    out = live.draw(img, [_det()], {"pothole"})
    # 영어 라벨 바탕(유형 색)이 박스 위에 그려진다
    assert (out[85:99, 101:150] == live.COLORS["sign"]).all(axis=-1).any()
    assert not img.any()  # 원본은 그대로


@pytest.mark.skipif(not HAS_FONT, reason="한글 글꼴 없음 (CI·Linux)")
def test_korean_label_patch_sits_above_box_and_is_cached():
    patch = live._label_patch("표지판 0.31", live.COLORS["sign"], 16)
    assert patch is not None and patch.ndim == 3
    assert (patch == live.COLORS["sign"]).all(axis=-1).any()  # 바탕
    assert (patch < 60).all(axis=-1).any()  # 검은 글자
    assert live._label_patch("표지판 0.31", live.COLORS["sign"], 16) is patch

    img = np.zeros((300, 400, 3), np.uint8)
    out = live.draw(img, [_det()], set())
    h, w = patch.shape[:2]
    assert (out[100 - h:100, 100:100 + w] == patch).all()


@pytest.mark.skipif(not HAS_FONT, reason="한글 글꼴 없음 (CI·Linux)")
def test_label_at_frame_edge_is_clipped_not_crashing():
    img = np.zeros((120, 160, 3), np.uint8)
    out = live.draw(img, [_det("pothole", 0.9, (130, 0, 159, 60))], {"pothole"})
    assert out.shape == img.shape and out.any()


def test_paste_clips_outside_frame():
    out = np.zeros((10, 10, 3), np.uint8)
    live._paste(out, np.full((4, 6, 3), 7, np.uint8), 7, -2)
    assert (out[0:2, 7:10] == 7).all() and out[2:].sum() == 0 and out[:, :7].sum() == 0


def test_is_current_hides_old_or_unknown_devices():
    now = 1_000_000.0
    assert live.is_current({"updated_at": now - 3}, now)
    assert live.is_current({"updated_at": now - live.HIDE_AFTER_S}, now)
    assert not live.is_current({"updated_at": now - live.HIDE_AFTER_S - 1}, now)  # 지난 주행이 남긴 화면
    assert not live.is_current({}, now)
