"""
Motor de cinemática 2D: entra coordenadas, salen magnitudes geométricas.
Independiente del origen de las detecciones (solo `numpy` + tipos de `schema`).
"""
from __future__ import annotations

from typing import Literal, Union

import numpy as np

from src.schema import Detection

PointLike = Union[tuple[float, float], Detection]

_EPS_LEN = 1e-12

# Sin keypoint de cadera: cintura estimada por debajo del hombro,
# proporcional al segmento cabeza-hombro (coords. imagen, y hacia abajo).
_WAIST_FACTOR = 2.0


def _vec2(p: PointLike) -> np.ndarray:
    """Convierte un punto a ndarray (2,) float64."""
    if isinstance(p, Detection):
        return np.array([float(p.x), float(p.y)], dtype=np.float64)
    a = np.asarray(p, dtype=np.float64).reshape(-1)
    if a.size < 2:
        raise ValueError("Se esperan al menos dos coordenadas (x, y).")
    return a[:2].astype(np.float64, copy=False)


def calculate_angle(p1: PointLike, p2: PointLike, p3: PointLike) -> float:
    """
    Ángulo en grados en el vértice `p2` formado por los segmentos (p1-p2) y (p3-p2).
    Usa el producto escalar: cos(theta) = (u·v) / (|u||v|) con u = p1-p2, v = p3-p2.
    Si algún vector es degenerado, devuelve NaN.
    """
    v1 = _vec2(p1) - _vec2(p2)
    v2 = _vec2(p3) - _vec2(p2)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < _EPS_LEN or n2 < _EPS_LEN:
        return float(np.nan)
    cos_t = float(np.dot(v1, v2) / (n1 * n2))
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(np.degrees(np.arccos(cos_t)))


def get_distance(p1: PointLike, p2: PointLike) -> float:
    """Distancia euclidiana entre dos puntos 2D."""
    d = _vec2(p1) - _vec2(p2)
    return float(np.linalg.norm(d))


def calculate_velocity(
    p_prev: PointLike,
    p_curr: PointLike,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Vector velocidad media en el intervalo ``dt``: ``(p_curr - p_prev) / dt``.
    Con ``dt=1`` coincide con el desplazamiento entre frames (útil para derivar
    cambios de dirección en el impacto). Unidades: coords / unidad de ``dt``.
    """
    if dt <= 0:
        raise ValueError("dt debe ser > 0.")
    return (_vec2(p_curr) - _vec2(p_prev)) / float(dt)


def classify_vertical_zone(
    ball_pos: PointLike,
    shoulder_pos: PointLike,
    head_pos: PointLike,
) -> Literal["high", "mid", "low"]:
    """
    Zona vertical heurística (coordenadas de imagen, **y creciente hacia abajo**):

    - ``high``: la pelota está por encima de la línea del hombro.
    - ``mid``: entre hombro y cintura estimada (aprox. ``hombro + factor × |hombro − cabeza|``).
    - ``low``: por debajo de esa cintura estimada.

    Sin keypoint de cadera: la cintura se aproxima solo con cabeza y hombro.
    Si ese segmento es degenerado, solo se distingue ``high`` del resto (``mid``).
    """
    yb = float(_vec2(ball_pos)[1])
    ys = float(_vec2(shoulder_pos)[1])
    yh = float(_vec2(head_pos)[1])

    seg = abs(ys - yh)
    if seg < _EPS_LEN:
        return "high" if yb < ys else "mid"

    # Cintura estimada por debajo del hombro (típicamente ys < yh no ocurre; importa |seg|)
    waist_y = ys + _WAIST_FACTOR * seg

    if yb < ys:
        return "high"
    if yb > waist_y:
        return "low"
    return "mid"


def detect_impact_candidate(
    ball_pos: PointLike,
    wrist_pos: PointLike,
    threshold: float,
) -> bool:
    """
    True si la pelota está a distancia <= `threshold` de la muñeca (misma unidad que las coords).
    """
    if threshold < 0:
        raise ValueError("threshold debe ser >= 0.")
    return get_distance(ball_pos, wrist_pos) <= threshold


def classify_side(
    ball_pos: PointLike,
    torso_center_pos: PointLike,
) -> Literal["forehand", "backhand"]:
    """
    Heurística lateral (jugador diestro): en coordenadas de imagen con x creciente a la derecha,
    si la pelota está estrictamente a la derecha del centro del torso → ``'forehand'``;
    en caso contrario → ``'backhand'``.
    """
    bx = float(_vec2(ball_pos)[0])
    tx = float(_vec2(torso_center_pos)[0])
    return "forehand" if bx > tx else "backhand"
