"""품질 판별 기준 (B6)."""

import numpy as np

from edge.blackbox.quality import measure

H, W = 1080, 1920


def _textured(mean, amp, seed=0):
    rng = np.random.default_rng(seed)
    g = np.clip(rng.normal(mean, amp, (H, W)), 0, 255).astype(np.uint8)
    return np.dstack([g] * 3)


def test_normal_road_passes():
    q = measure(_textured(110, 40))
    assert q.ok and q.reason == ""


def test_dark_rejected():
    q = measure(_textured(15, 5))
    assert not q.ok and "어두움" in q.reason


def test_flat_hazy_rejected():
    q = measure(np.full((H, W, 3), 120, np.uint8))  # 질감도 대비도 없는 뿌연 화면
    assert not q.ok and q.reason == "흐림"


def test_overexposed_rejected():
    q = measure(np.full((H, W, 3), 255, np.uint8))
    assert not q.ok and "과노출" in q.reason


def test_low_sharpness_but_high_contrast_passes():
    # 가로등 아래 야간 노면처럼: 매끈하지만 밝고 어두운 곳이 크게 갈린다
    img = np.full((H, W), 40, np.uint8)
    img[:, W // 2:] = 140
    q = measure(np.dstack([img] * 3))
    assert q.sharpness < 200 and q.contrast >= 30 and q.ok
