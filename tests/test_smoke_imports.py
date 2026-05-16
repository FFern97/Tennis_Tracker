"""Imports opcionales para cobertura de geometry_utils e inference (requiere scipy y torch)."""
import pytest


def test_geometry_utils_court_reference():
    pytest.importorskip("scipy")
    import geometry_utils

    ref = geometry_utils.CourtReference()
    assert len(ref.key_points) > 0


def test_yolo_detector_subclass_of_base_detector():
    pytest.importorskip("torch")
    from core.interfaces import BaseDetector
    from inference import YoloDetector

    assert issubclass(YoloDetector, BaseDetector)
