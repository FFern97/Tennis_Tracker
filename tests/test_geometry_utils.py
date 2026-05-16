"""Cobertura de geometry_utils.get_trans_matrix y módulo global."""
import numpy as np
import pytest
from unittest.mock import patch

pytest.importorskip("scipy")


def test_get_trans_matrix_with_identity_homography():
    """Ejecuta el bucle de configuraciones con homografía identidad simulada."""
    import geometry_utils as gu

    pts = [(float(x), float(y)) for (x, y) in gu.court_ref.key_points]

    eye = np.eye(3, dtype=np.float32)

    def fake_perspective(pts_in, _m):
        return np.asarray(pts_in, dtype=np.float32)

    with patch.object(gu.cv2, "findHomography", return_value=(eye, None)):
        with patch.object(gu.cv2, "perspectiveTransform", side_effect=fake_perspective):
            M = gu.get_trans_matrix(pts)

    assert M is not None
    assert M.shape == (3, 3)


def test_get_trans_matrix_returns_none_when_all_configs_skip():
    import geometry_utils as gu

    pts = [(None, None)] * 14
    assert gu.get_trans_matrix(pts) is None
