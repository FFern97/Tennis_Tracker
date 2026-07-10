"""
Interfaces (DIP): el orquestador depende de abstracciones, no de implementaciones concretas.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional


class BaseDetector(ABC):
    """
    Contrato de detección sobre un frame de imagen.
    """

    @abstractmethod
    def detect(self, frame: Any):
        """
        Detección global sobre el frame completo.

        Args:
            frame: Imagen BGR (numpy array).

        Returns:
            FrameData (schema) con listas ball y players.
        """
        raise NotImplementedError

    def detect_localized(self, frame: Any, last_pos: Optional[tuple]) -> List:
        """
        Detección opcional en ROI alrededor de last_pos.
        Implementación por defecto: sin detecciones localizadas.
        """
        return []


class BaseTracker(ABC):
    """
    Contrato genérico de tracking por frame.
    Cada subclase define la firma concreta de update (pelota vs jugadores).
    """

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> Any:
        """
        Actualiza el estado del tracker. Parámetros y retorno dependen de la subclase
        (p. ej. BallTracker: frame_number + detecciones de pelota -> BallInfo).
        """
        raise NotImplementedError
