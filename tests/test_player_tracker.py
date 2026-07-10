"""
PlayerTracker: mocks de cv2 (perspectiveTransform), sin geometry_utils en lógica de píxeles.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from schema import PlayerDetection, PlayersInfo
from src.trackers.player_tracker import PlayerTracker


def _player(track_id, x, y, conf=0.9, keypoints=None):
    return PlayerDetection(
        x=x,
        y=y,
        conf=conf,
        id=track_id,
        xyxy=(x - 10, y - 50, x + 10, y),
        keypoints=keypoints,
    )


@pytest.fixture
def small_interp_window(monkeypatch):
    monkeypatch.setattr(config, "PERSON_MAX_INTERPOLATION_FRAMES", 2)
    monkeypatch.setattr(config, "PERSON_TRACK_HISTORY_MAX", 10)
    yield


def test_track_lifecycle_create_update_remove(small_interp_window):
    """Crear track, actualizar, perder detección hasta superar missing_frames y eliminar."""
    with patch("src.trackers.player_tracker.cv2.perspectiveTransform") as pt_mock:
        pt_mock.return_value = np.array([[[1.0, 2.0]]], dtype=np.float32)

        tr = PlayerTracker()
        H = np.eye(3, dtype=np.float64)
        tr.set_homography(H)

        tr.update([_player(1, 100.0, 200.0)], inv_homography=H, frame_number=1)
        assert 1 in tr.tracks
        assert tr.tracks[1]["missing_frames"] == 0

        tr.update([_player(1, 110.0, 200.0)], inv_homography=H, frame_number=2)
        assert tr.tracks[1]["last_position_px"] == (110.0, 200.0)

        # Sin detección del track 1: missing_frames sube
        tr.update([], inv_homography=H, frame_number=3)
        assert tr.tracks[1]["missing_frames"] == 1

        tr.update([], inv_homography=H, frame_number=4)
        assert tr.tracks[1]["missing_frames"] == 2

        # missing_frames > max (2) → eliminación
        tr.update([], inv_homography=H, frame_number=5)
        assert 1 not in tr.tracks


def test_interpolation_generates_missing_positions(small_interp_window):
    """Con al menos 2 puntos en historial, la interpolación predice px desplazados."""
    with patch("src.trackers.player_tracker.cv2.perspectiveTransform") as pt_mock:
        pt_mock.return_value = np.array([[[0.0, 0.0]]], dtype=np.float32)

        tr = PlayerTracker()
        H = np.eye(3)
        dets = [
            _player(7, 0.0, 0.0),
            _player(7, 10.0, 0.0),
        ]
        tr.update([dets[0]], inv_homography=H, frame_number=1)
        tr.update([dets[1]], inv_homography=H, frame_number=2)

        info: PlayersInfo = tr.update([], inv_homography=H, frame_number=3)
        assert 7 in info.active_tracks
        assert info.active_tracks[7]["is_interpolated"] is True
        # vel (10,0), missing_frames 1 → last (10,0) + (10,0)*1 = (20,0)
        px = info.active_tracks[7]["px"]
        assert abs(px[0] - 20.0) < 1e-6
        assert abs(px[1] - 0.0) < 1e-6


def test_keypoints_high_confidence_vs_zero(small_interp_window):
    """Solo keypoints con confianza > umbral reciben el offset; conf 0 no se mueve."""
    with patch("src.trackers.player_tracker.cv2.perspectiveTransform") as pt_mock:
        pt_mock.return_value = np.array([[[0.0, 0.0]]], dtype=np.float32)

        tr = PlayerTracker()
        H = np.eye(3)
        kps = np.array(
            [
                [0.0, 0.0, 1.0],
                [10.0, 0.0, 0.0],
                [5.0, 5.0, 0.5],
            ],
            dtype=np.float32,
        )
        tr.update([_player(2, 100.0, 200.0, keypoints=kps)], inv_homography=H, frame_number=1)
        tr.update([_player(2, 110.0, 200.0, keypoints=kps.copy())], inv_homography=H, frame_number=2)

        info = tr.update([], inv_homography=H, frame_number=3)
        est = info.active_tracks[2]["keypoints"]
        thr = config.PERSON_KEYPOINT_VISIBILITY_THRESHOLD
        # Fila 0: visible (> thr) → desplazada +10 en x
        assert est[0][0] > kps[0][0]
        # Fila 1: conf 0 no > thr típicamente si thr es 0.0 — en código es > thr, 0 > 0 es False
        assert est[1][0] == kps[1][0]
        # Fila 2: 0.5 > 0 → desplazada
        assert est[2][0] > kps[2][0]


def test_set_homography_singular_returns_none():
    tr = PlayerTracker()
    bad = np.zeros((3, 3), dtype=float)
    tr.set_homography(bad)
    assert tr.homography is None


def test_get_track_history_and_get_all_tracks():
    tr = PlayerTracker()
    with patch("src.trackers.player_tracker.cv2.perspectiveTransform", return_value=np.array([[[0.0, 0.0]]])):
        tr.update([_player(3, 1.0, 2.0)], frame_number=1)
    hpx, hc, hk = tr.get_track_history(3)
    assert len(hpx) >= 1
    assert tr.get_all_tracks()[3]["last_position_px"] == (1.0, 2.0)
    hx, hy, hz = tr.get_track_history(999)
    assert hx is None and hy is None and hz is None


def test_skips_non_player_or_none_id():
    tr = PlayerTracker()
    bad = MagicMock()
    info = tr.update([bad], frame_number=1)
    assert info.active_tracks == {}
