"""
Esquemas de datos que viajan entre módulos del sistema de tracking.
Usar dataclasses para tipado claro y legibilidad.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Any
import numpy as np


@dataclass
class Detection:
    """
    Una única detección (pelota o centro/pies de jugador).
    Coordenadas en píxeles del frame.
    """
    x: float
    y: float
    conf: float
    id: Optional[int] = None


@dataclass
class PlayerDetection(Detection):
    """
    Detección de jugador con caja y keypoints de pose.
    Hereda x, y (centro horizontal, base del bbox = pies), conf, id.
    """
    xyxy: tuple = (0.0, 0.0, 0.0, 0.0)  # x1, y1, x2, y2
    keypoints: Optional[np.ndarray] = None  # shape (N, 3) -> x, y, confidence


@dataclass
class FrameData:
    """
    Datos de detección para un frame.
    Contiene las salidas estructuradas del modelo de inferencia.
    """
    ball: List[Detection] = field(default_factory=list)
    players: List[PlayerDetection] = field(default_factory=list)


@dataclass
class BallInfo:
    """
    Salida del BallTracker para un frame.
    Solo contiene lo esencial para dibujo y tracking.
    """
    position: Optional[tuple] = None  # (x, y) en píxeles
    is_interpolated: bool = False
    trajectory_history: List[tuple] = field(default_factory=list)


@dataclass
class PlayersInfo:
    """
    Salida del PersonTracker para un frame.
    """
    active_tracks: dict = field(default_factory=dict)  # track_id -> {px, court, is_interpolated, keypoints}
    all_positions: List[dict] = field(default_factory=list)
