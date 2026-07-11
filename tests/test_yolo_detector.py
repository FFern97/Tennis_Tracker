"""Tests del detector YOLO-Pose (keypoints COCO)."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.detectors.yolo_pose_detector import (
    COCO_POSE_KEYPOINTS,
    YoloPoseDetector,
)


def _make_pose_result(num_persons: int = 1):
    """Construye un objeto tipo Results de Ultralytics con cajas y keypoints."""
    boxes_list = []
    kpts_rows = []
    for p in range(num_persons):
        box = MagicMock()
        box.cls = MagicMock()
        box.cls.item.return_value = 0
        box.conf = MagicMock()
        box.conf.item.return_value = 0.9
        box.xyxy = MagicMock()
        box.xyxy.__getitem__.return_value = np.array([10.0, 20.0, 110.0, 220.0], dtype=np.float32)
        box.id = None
        boxes_list.append(box)

        coords = np.arange(p * 300, p * 300 + COCO_POSE_KEYPOINTS * 3, dtype=np.float32).reshape(
            COCO_POSE_KEYPOINTS, 3
        )
        kpts_rows.append(coords)

    result = MagicMock()
    result.boxes = boxes_list

    kp = MagicMock()
    kp.data = np.stack(kpts_rows, axis=0)
    result.keypoints = kp
    return result


def test_detect_single_frame_player_has_seventeen_keypoints():
    """Imagen de prueba (mock): cada jugador debe tener 17 keypoints × (x, y, conf)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_results = [_make_pose_result(1)]

    model_mock = MagicMock()
    model_mock.predict.return_value = fake_results
    det = YoloPoseDetector(model=model_mock)
    out = det.detect(frame)

    assert out.ball == []
    assert len(out.players) == 1
    kp = out.players[0].keypoints
    assert kp is not None
    assert kp.shape == (COCO_POSE_KEYPOINTS, 3)


def test_detect_list_returns_one_framedata_per_image():
    frames = [
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.ones((240, 320, 3), dtype=np.uint8) * 255,
    ]
    fake_results = [_make_pose_result(1), _make_pose_result(1)]

    model_mock = MagicMock()
    model_mock.predict.return_value = fake_results
    det = YoloPoseDetector(model=model_mock)
    batch_out = det.detect(frames)

    assert isinstance(batch_out, list)
    assert len(batch_out) == 2
    for fd in batch_out:
        assert len(fd.players) == 1
        assert fd.players[0].keypoints.shape == (COCO_POSE_KEYPOINTS, 3)


def test_detect_filters_non_person_class():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    box_person = MagicMock()
    box_person.cls = MagicMock()
    box_person.cls.item.return_value = 0
    box_person.conf = MagicMock()
    box_person.conf.item.return_value = 0.8
    box_person.xyxy = MagicMock()
    box_person.xyxy.__getitem__.return_value = np.array([0.0, 0.0, 50.0, 100.0], dtype=np.float32)
    box_person.id = None

    box_other = MagicMock()
    box_other.cls = MagicMock()
    box_other.cls.item.return_value = 1
    box_other.conf = MagicMock()
    box_other.conf.item.return_value = 0.99
    box_other.xyxy = MagicMock()
    box_other.xyxy.__getitem__.return_value = np.array([5.0, 5.0, 20.0, 40.0], dtype=np.float32)
    box_other.id = None

    kpts = np.zeros((2, COCO_POSE_KEYPOINTS, 3), dtype=np.float32)
    result = MagicMock()
    result.boxes = [box_person, box_other]
    kp = MagicMock()
    kp.data = kpts
    result.keypoints = kp

    model_mock = MagicMock()
    model_mock.predict.return_value = [result]
    det = YoloPoseDetector(model=model_mock)
    out = det.detect(frame)

    assert len(out.players) == 1


def test_export_to_onnx_delegates_to_ultralytics_export():
    model_mock = MagicMock()
    model_mock.export.return_value = "/tmp/model.onnx"
    det = YoloPoseDetector(model=model_mock)
    path = det.export_to_onnx(simplify=True)

    model_mock.export.assert_called_once()
    _args, kwargs = model_mock.export.call_args
    assert kwargs.get("format") == "onnx"
    assert path == "/tmp/model.onnx"


def test_detect_rejects_non_ndarray_list_elements():
    det = YoloPoseDetector(model=MagicMock())
    with pytest.raises(TypeError):
        det.detect([np.zeros((10, 10, 3), dtype=np.uint8), "no-array"])
