import numpy as np

from daboyeo_ai.inference.lane_seg import decode_seg
from daboyeo_ai.inference.onnx_detector import letterbox


def _outputs(cands, nc=2, nm=2):
    """cands: [(cx, cy, w, h, cls, score, coef)] → out0 (1, 4+nc+nm, N)."""
    rows = []
    for cx, cy, w, h, c, s, coef in cands:
        sc = [0.0] * nc
        sc[c] = s
        rows.append([cx, cy, w, h, *sc, *coef])
    return np.asarray(rows, np.float32).T[None]


def test_decode_seg_left_half_and_padding():
    _, lb = letterbox(np.zeros((32, 64, 3), np.uint8), 64)  # 위아래 16 px 패딩
    assert (lb.pad_x, lb.pad_y, lb.ratio) == (0, 16, 1.0)
    protos = np.full((1, 2, 16, 16), -10, np.float32)
    protos[0, 0, :, :8] = 10   # 프로토타입 0 = 왼쪽 절반
    protos[0, 1, :, 8:] = 10   # 프로토타입 1 = 오른쪽 절반
    out0 = _outputs([
        (32, 32, 64, 64, 0, 0.9, (1, 0)),   # lane: 왼쪽 절반
        (32, 32, 64, 64, 1, 0.9, (0, 1)),   # marking: 오른쪽 절반
        (32, 32, 64, 64, 0, 0.1, (0, 1)),   # 점수 낮음 → 버림
    ])
    res = decode_seg(out0, protos, lb, 64, 64, 32, nc=2, conf=0.25, classes=(0, 1))
    lane, mark = res[0], res[1]
    assert lane.shape == (32, 64)
    assert lane[:, :30].min() == 255 and lane[:, 34:].max() == 0
    assert mark[:, 34:].min() == 255 and mark[:, :30].max() == 0


def test_decode_seg_box_crops_mask_and_empty():
    _, lb = letterbox(np.zeros((64, 64, 3), np.uint8), 64)
    protos = np.full((1, 2, 16, 16), 10, np.float32)          # 어디서나 1
    out0 = _outputs([(16, 16, 32, 32, 0, 0.9, (1, 0))])       # 왼쪽 위 32×32 상자
    m = decode_seg(out0, protos, lb, 64, 64, 64, nc=2)[0]
    assert m[:28, :28].min() == 255 and m[36:, :].max() == 0 and m[:, 36:].max() == 0
    none = decode_seg(_outputs([(16, 16, 32, 32, 0, 0.1, (1, 0))]), protos, lb, 64, 64, 64, nc=2)[0]
    assert none.max() == 0
