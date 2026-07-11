"""Tests unitarios del pipeline de impacto (sin torch)."""
import pytest
import numpy as np

from src.pipeline.impact_utils import (
    merge_pose_keypoints,
    try_detect_stroke,
)
from src.schema import PlayerDetection


def _player(track_id: int, x1: float, kp_fill: float = 100.0) -> PlayerDetection:
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 0] = kp_fill
    kp[:, 1] = kp_fill
    kp[:, 2] = 0.95
    return PlayerDetection(
        x=(x1 + 110) / 2,
        y=220.0,
        conf=0.9,
        id=track_id,
        xyxy=(x1, 200.0, x1 + 100.0, 240.0),
        keypoints=kp.copy(),
    )


def test_merge_pose_keypoints_prefers_high_iou():
    tracked = [_player(1, 50.0, kp_fill=10.0)]
    pose_same_box = PlayerDetection(
        x=100,
        y=220,
        conf=0.88,
        id=None,
        xyxy=(50.0, 200.0, 150.0, 240.0),
        keypoints=np.full((17, 3), 5.0, dtype=np.float32),
    )
    pose_far = PlayerDetection(
        x=400,
        y=220,
        conf=0.88,
        id=None,
        xyxy=(350.0, 200.0, 450.0, 240.0),
        keypoints=np.full((17, 3), 99.0, dtype=np.float32),
    )
    out = merge_pose_keypoints(tracked, [pose_far, pose_same_box], iou_min=0.15)
    assert len(out) == 1
    assert out[0].id == 1
    assert np.allclose(out[0].keypoints, pose_same_box.keypoints)


def test_try_detect_stroke_positive_near_wrist():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[9, :] = [400.0, 300.0, 0.9]  # left wrist
    kp[5, :] = [380.0, 280.0, 0.9]
    kp[6, :] = [420.0, 280.0, 0.9]
    kp[0, :] = [400.0, 260.0, 0.9]
    pl = PlayerDetection(
        x=400,
        y=350,
        conf=0.9,
        id=7,
        xyxy=(360.0, 260.0, 440.0, 360.0),
        keypoints=kp,
    )
    stroke = try_detect_stroke(
        pl,
        (402.0, 302.0),
        threshold_px=80,
        wrist_conf_min=0.2,
        ball_conf=0.85,
    )
    assert stroke is not None
    assert stroke["side"] in ("forehand", "backhand")
    assert stroke["vertical_zone"] in ("high", "mid", "low")
    assert stroke["confidence_score"] == pytest.approx(min(0.9, 0.85))


def test_try_detect_stroke_negative_far_ball():
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[10, :] = [100.0, 100.0, 0.95]
    pl = PlayerDetection(
        x=50,
        y=200,
        conf=0.9,
        id=1,
        xyxy=(0.0, 0.0, 100.0, 250.0),
        keypoints=kp,
    )
    stroke = try_detect_stroke(
        pl,
        (600.0, 400.0),
        threshold_px=40,
        wrist_conf_min=0.2,
        ball_conf=0.9,
    )
    assert stroke is None
