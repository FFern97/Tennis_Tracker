"""Cobertura de core/interfaces: ABC y comportamiento por defecto."""
import pytest

from core.interfaces import BaseDetector, BaseTracker


def test_base_detector_incomplete_raises_type_error():
    """Subclase sin implementar detect() no puede instanciarse (ABC)."""

    class Incomplete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_base_detector_detect_localized_default_empty():
    """Implementación mínima: detect_localized por defecto devuelve []."""

    class Minimal(BaseDetector):
        def detect(self, frame):
            return "ok"

    m = Minimal()
    assert m.detect_localized(None, (1.0, 2.0)) == []


def test_base_tracker_incomplete_raises_type_error():
    class Incomplete(BaseTracker):
        pass

    with pytest.raises(TypeError):
        Incomplete()
