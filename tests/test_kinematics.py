"""Tests del motor de cinemática 2D (`analytics.kinematics`)."""
import math

import pytest

import numpy as np

from analytics.kinematics import (
    calculate_angle,
    calculate_velocity,
    classify_side,
    classify_vertical_zone,
    detect_impact_candidate,
    get_distance,
)
from schema import Detection


def test_calculate_angle_right_angle():
    # Vértice en origen: (-1,0) - (0,0) - (0,1) => 90°
    assert calculate_angle((-1.0, 0.0), (0.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)


def test_calculate_angle_straight_line_180():
    # Colineales, ángulo en el punto medio = 180°
    assert calculate_angle((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)) == pytest.approx(180.0)


def test_calculate_angle_with_detection_objects():
    p1 = Detection(x=0.0, y=0.0, conf=1.0)
    p2 = Detection(x=1.0, y=0.0, conf=1.0)
    p3 = Detection(x=2.0, y=0.0, conf=1.0)
    assert calculate_angle(p1, p2, p3) == pytest.approx(180.0)


def test_calculate_angle_degenerate_returns_nan():
    # p1 = p2 => vector nulo en el vértice
    out = calculate_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isnan(out)


def test_get_distance_known():
    assert get_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert get_distance((1.0, 2.0), (1.0, 2.0)) == pytest.approx(0.0)


def test_get_distance_detection_mixed():
    a = Detection(x=0.0, y=0.0, conf=1.0)
    assert get_distance(a, (3.0, 4.0)) == pytest.approx(5.0)


def test_detect_impact_candidate():
    assert detect_impact_candidate((0.0, 0.0), (0.0, 0.0), 1.0) is True
    assert detect_impact_candidate((0.0, 0.0), (1.0, 0.0), 1.0) is True
    assert detect_impact_candidate((0.0, 0.0), (2.0, 0.0), 1.0) is False


def test_detect_impact_candidate_negative_threshold():
    with pytest.raises(ValueError, match="threshold"):
        detect_impact_candidate((0.0, 0.0), (0.0, 0.0), -0.1)


def test_classify_side_forehand_backhand():
    torso = (100.0, 200.0)
    assert classify_side((150.0, 50.0), torso) == "forehand"
    assert classify_side((50.0, 50.0), torso) == "backhand"
    assert classify_side((100.0, 50.0), torso) == "backhand"  # igualdad => no forehand


def test_classify_side_with_detection():
    ball = Detection(x=200.0, y=0.0, conf=1.0)
    torso = Detection(x=100.0, y=0.0, conf=1.0)
    assert classify_side(ball, torso) == "forehand"


def test_vec2_raises_on_short_sequence():
    with pytest.raises(ValueError):
        calculate_angle((0.0,), (0.0, 0.0), (1.0, 0.0))


def test_calculate_velocity_dt1_equals_displacement():
    v = calculate_velocity((0.0, 0.0), (3.0, 4.0), dt=1.0)
    assert np.allclose(v, np.array([3.0, 4.0]))


def test_calculate_velocity_scales_with_dt():
    v = calculate_velocity((0.0, 0.0), (4.0, 0.0), dt=2.0)
    assert np.allclose(v, np.array([2.0, 0.0]))


def test_calculate_velocity_nonpositive_dt():
    with pytest.raises(ValueError, match="dt"):
        calculate_velocity((0.0, 0.0), (1.0, 0.0), dt=0.0)


def test_classify_vertical_zone_high_mid_low():
    # y hacia abajo: cabeza arriba, hombro abajo; cintura estimada = ys + 2*|ys-yh|
    head = (100.0, 50.0)
    shoulder = (100.0, 150.0)
    assert classify_vertical_zone((100.0, 100.0), shoulder, head) == "high"
    assert classify_vertical_zone((100.0, 200.0), shoulder, head) == "mid"  # entre 150 y 350
    assert classify_vertical_zone((100.0, 400.0), shoulder, head) == "low"


def test_classify_vertical_zone_degenerate_head_shoulder_same_y():
    """Sin segmento cabeza-hombro solo se separa alto del resto (mid)."""
    h = s = (100.0, 150.0)
    assert classify_vertical_zone((100.0, 100.0), h, s) == "high"
    assert classify_vertical_zone((100.0, 200.0), h, s) == "mid"
