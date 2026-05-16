"""
Tests unitarios de BallTracker: mocks de detecciones, sin video ni modelos.
"""
import pytest

import config
from schema import Detection, BallInfo
from trackers.ball_tracker import BallTracker


def _det(x, y, conf=0.99):
    return [Detection(x=float(x), y=float(y), conf=conf, id=None)]


@pytest.fixture
def restore_moving_average_window():
    prev = config.BALL_MOVING_AVERAGE_WINDOW
    yield
    config.BALL_MOVING_AVERAGE_WINDOW = prev


def test_update_accepts_mock_detections_above_confidence():
    tr = BallTracker()
    dets = _det(100.0, 200.0, conf=config.BALL_CONFIDENCE + 0.01)
    info: BallInfo = tr.update(1, dets)
    assert info.position == (100.0, 200.0)
    assert tr.get_last_position() == (100.0, 200.0)


def test_update_rejects_below_confidence():
    tr = BallTracker()
    dets = _det(1.0, 2.0, conf=config.BALL_CONFIDENCE - 0.05)
    info = tr.update(1, dets)
    assert info.position is None


def test_gap_five_frames_linear_interpolation():
    """
    Hueco de 5 frames entre dos detecciones conocidas (frame 0 y frame 6).
    Tras interpolate + ffill/bfill, los puntos intermedios siguen trayectoria lineal.
    """
    tr = BallTracker()
    # 7 frames: índices 0..6; huecos en 1..5
    seq = []
    for i in range(7):
        if i == 0:
            seq.append(_det(0.0, 0.0))
        elif i == 6:
            seq.append(_det(10.0, 10.0))
        else:
            seq.append([])

    out = tr.interpolate_ball_positions(seq)
    assert len(out) == 7
    # Frame 3: 3/6 del segmento (0,0)->(10,10)
    mid = out[3][0]
    assert abs(mid.x - 5.0) < 1e-6
    assert abs(mid.y - 5.0) < 1e-6
    # Extremos preservan detección original (mismas coords)
    assert abs(out[0][0].x - 0.0) < 1e-6
    assert abs(out[6][0].x - 10.0) < 1e-6


def test_moving_average_smooths_spike(restore_moving_average_window):
    """
    Con BALL_MOVING_AVERAGE_WINDOW>=2, la media móvil (rolling) atenúa picos aislados
    sobre la trayectoria ya interpolada/rellenada.
    """
    config.BALL_MOVING_AVERAGE_WINDOW = 3
    tr = BallTracker()
    seq = []
    for i in range(7):
        if i == 3:
            seq.append(_det(100.0, 100.0))
        else:
            seq.append(_det(float(i), float(i)))

    out = tr.interpolate_ball_positions(seq)
    p = out[3][0]
    # Media de vecinos (2,2), (100,100), (4,4) → ~35.3
    assert 30.0 < p.x < 45.0
    assert 30.0 < p.y < 45.0
