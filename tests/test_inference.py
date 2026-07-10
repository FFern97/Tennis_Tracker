"""
YoloDetector: mocks de resultados YOLO; stub de torch/ultralytics si no hay PyTorch real.
"""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


def _bootstrap_torch_ultralytics_stubs():
    """Permite importar inference sin torch real (CI / entornos mínimos)."""
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod._tennis_stub = True
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch_mod.device = lambda *a, **k: "cpu"
        torch_mod.tensor = lambda data, **kw: np.asarray(data, dtype=np.float32)
        sys.modules["torch"] = torch_mod
    if "ultralytics" not in sys.modules:
        ultra = types.ModuleType("ultralytics")

        def _make_yolo(*_a, **_k):
            return MagicMock()

        ultra.YOLO = _make_yolo
        sys.modules["ultralytics"] = ultra


_bootstrap_torch_ultralytics_stubs()

from src.vision_tracking.inference import YoloDetector  # noqa: E402


def _bare_detector():
    d = YoloDetector.__new__(YoloDetector)
    d._person_model = MagicMock()
    d._ball_model = MagicMock()
    d._device = "cpu"
    return d


def _ball_result_with_boxes(box_mocks):
    r0 = MagicMock()
    r0.boxes = box_mocks
    return [r0]


def _person_result_with_boxes(box_list, keypoints_data=None):
    r0 = MagicMock()
    r0.boxes = box_list
    kp = MagicMock()
    kp.data = keypoints_data
    r0.keypoints = kp if keypoints_data is not None else None
    return [r0]


def test_parse_ball_results_empty_and_none_boxes():
    d = _bare_detector()
    assert d._parse_ball_results([]) == []
    assert d._parse_ball_results(None) == []
    assert d._parse_ball_results(_ball_result_with_boxes(None)) == []
    assert d._parse_ball_results(_ball_result_with_boxes([])) == []


def test_parse_ball_results_skips_nonpositive_conf_and_multiple():
    d = _bare_detector()
    low = MagicMock()
    low.conf.item.return_value = 0.0
    low.xyxy = [np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)]

    hi = MagicMock()
    hi.conf.item.return_value = 0.88
    hi.xyxy = [np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)]

    out = d._parse_ball_results(_ball_result_with_boxes([low, hi]))
    assert len(out) == 1
    assert abs(out[0].x - 20.0) < 1e-4
    assert abs(out[0].y - 30.0) < 1e-4


def test_parse_person_results_empty_variants():
    d = _bare_detector()
    assert d._parse_person_results([]) == []
    assert d._parse_person_results(None) == []

    r0 = MagicMock()
    r0.boxes = None
    assert d._parse_person_results([r0]) == []


def test_parse_person_results_multiple_and_keypoints_tensor_path():
    d = _bare_detector()
    b0 = MagicMock()
    b0.id = MagicMock()
    b0.id.item.return_value = 42
    b0.conf.item.return_value = 0.91
    b0.xyxy = [np.array([0.0, 0.0, 20.0, 100.0], dtype=np.float32)]

    b1 = MagicMock()
    b1.id = None
    b1.conf.item.return_value = 0.8
    b1.xyxy = [np.array([5.0, 5.0, 15.0, 25.0], dtype=np.float32)]

    kdata = np.zeros((2, 4, 3), dtype=np.float32)
    kdata[0, 0, :] = (1.0, 2.0, 0.9)
    kdata[1, 0, :] = (3.0, 4.0, 0.8)

    out = d._parse_person_results(_person_result_with_boxes([b0, b1], keypoints_data=kdata))
    assert len(out) == 2
    assert out[0].id == 42
    assert out[1].id is None


def test_parse_person_keypoints_numpy_branch():
    d = _bare_detector()
    b0 = MagicMock()
    b0.id = MagicMock()
    b0.id.item.return_value = 1
    b0.conf.item.return_value = 0.9
    b0.xyxy = [np.array([0.0, 0.0, 10.0, 50.0], dtype=np.float32)]

    kp_row = np.zeros((2, 3), dtype=np.float32)
    kdata = [kp_row]

    out = d._parse_person_results(_person_result_with_boxes([b0], keypoints_data=kdata))
    assert len(out) == 1
    assert out[0].keypoints is not None


def test_detect_localized_none_last_pos():
    d = _bare_detector()
    assert d.detect_localized(np.zeros((100, 100, 3), dtype=np.uint8), None) == []


def test_detect_localized_roi_too_small():
    d = _bare_detector()
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    assert d.detect_localized(frame, (20.0, 20.0)) == []


def test_detect_localized_with_mock_predict():
    d = _bare_detector()
    box = MagicMock()
    box.conf.item.return_value = 0.5
    box.xyxy = [np.array([5.0, 5.0, 15.0, 15.0], dtype=np.float32)]
    d._ball_model.predict.return_value = _ball_result_with_boxes([box])

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    out = d.detect_localized(frame, (100.0, 100.0))
    assert len(out) == 1


def test_detect_calls_models_and_returns_frame_data(monkeypatch):
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    ball_box = MagicMock()
    ball_box.conf.item.return_value = 0.7
    ball_box.xyxy = [np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float32)]

    pb = MagicMock()
    pb.id = MagicMock()
    pb.id.item.return_value = 9
    pb.conf.item.return_value = 0.85
    pb.xyxy = [np.array([10.0, 10.0, 20.0, 40.0], dtype=np.float32)]

    m_person = MagicMock()
    m_ball = MagicMock()
    m_person.track.return_value = _person_result_with_boxes(
        [pb], keypoints_data=np.zeros((1, 2, 3), dtype=np.float32)
    )
    m_ball.predict.return_value = _ball_result_with_boxes([ball_box])

    import src.vision_tracking.inference as inference_mod

    calls = {"n": 0}

    def _yo_side(*_a, **_k):
        calls["n"] += 1
        return m_person if calls["n"] == 1 else m_ball

    monkeypatch.setattr(inference_mod, "YOLO", _yo_side)
    det = YoloDetector()
    fd = det.detect(frame)

    assert len(fd.ball) >= 1
    assert len(fd.players) >= 1
    assert fd.players[0].id == 9
