"""rdd_figs — 가짜 박스로 결과 분류·후보 선택·P/R·그림·zip → 산출물 흐름을 시험한다."""

import json
import zipfile

import cv2
import numpy as np

from daboyeo_ai.eval import rdd_figs
from daboyeo_ai.eval.rdd_figs import (
    contact_sheet,
    country_of,
    draw,
    filter_preds,
    outcome_of,
    pr_at,
    select_candidates,
)

CONF = {"pothole": 0.31, "crack": 0.42}
P = [10, 10, 60, 60]


def _im(name="rdd_Japan_000001", gts=(), preds=(), w=100, h=100, brightness=120.0):
    return {"name": name, "w": w, "h": h, "brightness": brightness,
            "gts": [{"cls": c, "box": b} for c, b in gts],
            "preds": [{"cls": c, "score": s, "box": b} for c, s, b in preds]}


def test_country_of():
    assert country_of("rdd_United_States_001808") == "United_States"
    assert country_of("rdd_China_MotorBike_000003") == "China_MotorBike"
    assert country_of("rdd_Japan_002032") == "Japan"


def test_filter_preds_uses_class_conf():
    preds = [{"cls": "pothole", "score": 0.31, "box": P}, {"cls": "crack", "score": 0.41, "box": P},
             {"cls": "pothole", "score": 0.30, "box": P}]
    assert [p["score"] for p in filter_preds(preds, CONF)] == [0.31]


def test_outcome_kinds():
    assert outcome_of(_im(gts=[("pothole", P)], preds=[("pothole", 0.8, P)]), CONF).kind == "hit"
    assert outcome_of(_im(gts=[("pothole", P)], preds=[("pothole", 0.2, P)]), CONF).kind == "miss"  # 기준 미만
    assert outcome_of(_im(preds=[("pothole", 0.9, P)]), CONF).kind == "false"
    o = outcome_of(_im(gts=[("pothole", P), ("pothole", [70, 70, 90, 90])], preds=[("pothole", 0.8, P)]), CONF)
    assert (o.kind, o.tp, o.fn) == ("partial", 1, 1)
    assert outcome_of(_im(gts=[("crack", P)], preds=[("crack", 0.9, P)]), CONF).kind == "none"
    o = outcome_of(_im(gts=[("pothole", P)], preds=[("pothole", 0.8, P)]), CONF)
    assert o.gt_area == 0.25 and o.best_score == 0.8 and o.country == "Japan"


def test_pr_at():
    images = [_im(gts=[("pothole", P)], preds=[("pothole", 0.8, P), ("pothole", 0.5, [70, 70, 90, 90])]),
              _im(gts=[("pothole", P)], preds=[("pothole", 0.1, P)])]
    r = pr_at(images, CONF)["pothole"]
    assert (r["tp"], r["fp"], r["n_gt"], r["P"], r["R"]) == (1, 1, 2, 0.5, 0.5)
    assert pr_at(images, CONF)["crack"]["P"] is None


def test_select_candidates_order_and_filters():
    outs = [outcome_of(im, CONF) for im in [
        _im("rdd_India_000001", [("pothole", P)], [("pothole", 0.9, P)]),
        _im("rdd_Japan_000002", [("pothole", P)], [("pothole", 0.7, P)]),
        _im("rdd_Japan_000003", [("pothole", P)], [("pothole", 0.9, P)], brightness=20),  # 밤 → 빠짐
        _im("rdd_Japan_000004", [("pothole", [0, 0, 2, 2])], [("pothole", 0.9, [0, 0, 2, 2])]),  # 너무 작음
        _im("rdd_Czech_000005", [("pothole", P)], []),
        _im("rdd_Norway_000006", [], [("pothole", 0.6, P)]),
    ]]
    got = [o.name for o in select_candidates(outs)]
    assert got == ["rdd_Japan_000002", "rdd_India_000001", "rdd_Czech_000005", "rdd_Norway_000006"]
    assert len(select_candidates(outs, n_hit=1, n_miss=0, n_false=0)) == 1


def test_draw_and_contact_sheet():
    img = np.zeros((100, 120, 3), np.uint8)
    fig = draw(img, _im(gts=[("pothole", P)], preds=[("pothole", 0.8, [20, 20, 70, 70])], w=120), CONF)
    assert fig.shape == img.shape and img.sum() == 0 and fig.sum() > 0
    b, g, r = (int(v) for v in fig[10, 35])  # 정답 위쪽 선 (안티에일리어싱이라 정확한 값 대신 초록이 우세한지)
    assert g > 150 and b < 100 and r < 100
    sheet = contact_sheet([(fig, "a"), (fig, "b")], cols=2, tw=60, th=50)
    assert sheet.shape == (50 + 28, 120, 3)


class _FakeDet:
    def raw(self, img):  # (cls_id, score, x1, y1, x2, y2)
        return [(0, 0.9, 10, 10, 60, 60), (2, 0.9, 0, 0, 5, 5)]


def test_main_end_to_end(tmp_path, monkeypatch):
    zp = tmp_path / "rdd2022.zip"
    img = np.full((100, 100, 3), 128, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("det/images/test/rdd_Japan_000001.jpg", buf.tobytes())
        z.writestr("det/labels/test/rdd_Japan_000001.txt", "0 0.35 0.35 0.5 0.5\n")
    card = tmp_path / "m.json"
    card.write_text(json.dumps({"class_conf_draft": {"pothole": 0.31, "crack": 0.42, "guardrail": 0.2, "sign": 0.2}}))
    import daboyeo_ai.inference as inf

    monkeypatch.setattr(inf, "OnnxDetector", lambda *a, **k: _FakeDet())
    out = tmp_path / "out"
    assert rdd_figs.main(["--zip", str(zp), "--model", str(tmp_path / "m.onnx"), "--card", str(card),
                          "--out", str(out)]) == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["class_conf"] == CONF
    assert s["pr_at_class_conf"]["pothole"] == {"conf": 0.31, "tp": 1, "fp": 0, "n_gt": 1, "P": 1.0, "R": 1.0}
    assert [c["kind"] for c in s["candidates"]] == ["hit"]
    assert (out / "01_hit_rdd_Japan_000001.png").exists() and (out / "contact_sheet.png").exists()
    # 캐시: 두 번째는 추론 없이
    monkeypatch.setattr(inf, "OnnxDetector", None)
    assert rdd_figs.main(["--zip", str(zp), "--model", str(tmp_path / "m.onnx"), "--card", str(card),
                          "--out", str(out), "--pick", "rdd_Japan_000001"]) == 0
