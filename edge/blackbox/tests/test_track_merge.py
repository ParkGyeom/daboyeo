"""edge/blackbox/track_merge.py: 연속 프레임 탐지를 결함 1건 = 관측 1건으로 합치는지."""

from dataclasses import dataclass
from itertools import pairwise

from edge.blackbox.track_merge import GAP_S, TrackMerger, follow_score, iou

FPS = 30.0


@dataclass
class D:
    type: str
    confidence: float
    bbox: tuple


def play(merger, frames, every=1):
    """frames: 프레임마다 탐지 목록. every 프레임마다만 탐지를 넣는다. 끝난 트랙을 모두 돌려준다."""
    out = []
    for k, dets in enumerate(frames):
        t = k / FPS
        out += merger.expire(t)
        if k % every == 0:
            out += merger.update(dets, t, payload=k)
    return out + merger.flush()


def test_same_pothole_10_frames_is_one_observation():
    confs = [0.4, 0.5, 0.45, 0.9, 0.6, 0.55, 0.5, 0.7, 0.3, 0.6]
    frames = [[D("pothole", c, (600 + k, 400 + 3 * k, 680 + 2 * k, 440 + 4 * k))] for k, c in enumerate(confs)]
    m = TrackMerger()
    out = play(m, frames)
    assert len(out) == 1
    tr = out[0]
    assert tr.hits == 10 and tr.best.confidence == 0.9 and tr.payload == 3  # 가장 신뢰도 높은 프레임
    assert m.detections == 10 and m.closed == 1


def test_gap_longer_than_t_makes_two_observations():
    box = (600, 400, 680, 440)
    gap = int((GAP_S + 0.2) * FPS)  # T 초보다 조금 더 안 보임
    frames = [[D("pothole", 0.6, box)]] * 5 + [[]] * gap + [[D("pothole", 0.6, box)]] * 5
    assert len(play(TrackMerger(), frames)) == 2
    # T 초 안에 다시 보이면 이어진다
    short = [[D("pothole", 0.6, box)]] * 5 + [[]] * int((GAP_S - 0.2) * FPS) + [[D("pothole", 0.6, box)]] * 5
    assert len(play(TrackMerger(), short)) == 1


def test_overlapping_different_classes_stay_separate():
    frames = [[D("pothole", 0.7, (600, 400, 680, 440)), D("crack", 0.5, (590, 395, 690, 450))]] * 10
    out = play(TrackMerger(), frames)
    assert sorted(tr.type for tr in out) == ["crack", "pothole"]
    assert all(tr.hits == 10 for tr in out)


def test_two_potholes_side_by_side_stay_separate():
    frames = [[D("pothole", 0.7, (500, 400, 560, 430)), D("pothole", 0.6, (620, 400, 680, 430))]] * 10
    assert len(play(TrackMerger(), frames)) == 2


def approaching_pothole(kmh=50.0, frames=60, f=537.0, cam_h=1.3, vp=(640, 340), x=0.8, w=0.6, length=0.4, z0=16.0):
    """핀홀 카메라에서 다가오는 포트홀의 박스 (땅 위 [z, z+length]). 화면 아래(보닛)로 나가면 빈 프레임."""
    out = []
    for k in range(frames):
        z = z0 - kmh / 3.6 * k / FPS
        y1, y2 = vp[1] + f * cam_h / (z + length), vp[1] + f * cam_h / z
        x1, x2 = vp[0] + f * (x - w / 2) / z, vp[0] + f * (x + w / 2) / z
        out.append([D("pothole", 0.5, (x1, y1 - 4, x2, y2 + 4))] if z > 0 and y2 < 620 else [])
    return out


def test_approaching_pothole_is_one_observation_although_iou_is_low():
    frames = approaching_pothole()
    every = 10  # 50 km/h 에서 탐지 사이 4.6 m — 포트홀 길이(0.4 m)보다 훨씬 멀리 간다
    shown = [fr[0].bbox for k, fr in enumerate(frames) if fr and k % every == 0]
    assert len(shown) >= 3
    assert max(iou(a, b) for a, b in pairwise(shown)) < 0.3  # IoU 만으로는 이어지지 않는다
    assert len(play(TrackMerger(gate=0), frames, every)) == len(shown)  # IoU 만: 탐지마다 1건
    assert len(play(TrackMerger(), frames, every)) == 1


def test_long_crack_is_cut_by_max_age():
    frames = [[D("crack", 0.5, (500, 450, 700, 600))]] * int(7 * FPS)  # 7초 동안 계속 보인다
    assert len(play(TrackMerger(max_age_s=3.0), frames, every=3)) == 3


def test_time_going_backwards_closes_tracks():
    m = TrackMerger()
    m.update([D("pothole", 0.5, (0, 0, 10, 10))], 100.0)
    assert len(m.update([D("pothole", 0.5, (0, 0, 10, 10))], 5.0)) == 1
    assert len(m.flush()) == 1


def test_follow_score_prefers_iou_and_rejects_upward_jump():
    prev = (600, 400, 680, 440)
    assert follow_score(prev, prev) > 1.0
    assert 0 < follow_score(prev, (610, 470, 710, 530)) <= 1.0  # 아래로 내려가며 커짐
    assert follow_score(prev, (600, 200, 680, 240)) is None  # 위로 크게
    assert follow_score(prev, (1100, 400, 1180, 440)) is None  # 옆으로 멀리
