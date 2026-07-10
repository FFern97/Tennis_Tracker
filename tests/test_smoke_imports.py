"""Imports opcionales para cobertura de geometry_utils e inference (requiere scipy y torch)."""
import pytest


def test_geometry_utils_court_reference():
    pytest.importorskip("scipy")
    from src.vision_tracking import geometry_utils

    ref = geometry_utils.CourtReference()
    assert len(ref.key_points) > 0


def test_yolo_detector_subclass_of_base_detector():
    pytest.importorskip("torch")
    from src.core.interfaces import BaseDetector
    from src.vision_tracking.inference import YoloDetector

    assert issubclass(YoloDetector, BaseDetector)
