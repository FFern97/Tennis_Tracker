"""
Punto de entrada único para dibujar frame: court, ball, players y minimap.
"""
import cv2
import numpy as np

from schema import BallInfo, PlayersInfo
from visualization_utils import (
    draw_keypoints,
    draw_players_from_info,
    draw_minimap,
    save_final_heatmap,
)


def render(
    frame,
    ball_info: BallInfo,
    players_info: PlayersInfo,
    court_keypoints=None,
    homography_matrix=None,
    inv_homography=None,
    show_minimap=True,
):
    """
    Dibuja en el frame: puntos de cancha, pelota (posición/gap), jugadores y minimap.
    Diseñado para uso en el loop principal: visualization.render(frame, ball_info, players_info).
    """
    out = frame.copy()

    # Puntos clave de la cancha
    if court_keypoints is not None:
        out = draw_keypoints(out, court_keypoints)

    # Pelota: normal si detectada, amarillo con 'GAP' si interpolada
    if ball_info.position is not None:
        cx, cy = int(ball_info.position[0]), int(ball_info.position[1])
        if ball_info.is_interpolated:
            # Posición interpolada: dibujar en amarillo con texto 'GAP'
            color = (0, 255, 255)  # Amarillo en BGR
            cv2.circle(out, (cx, cy), 8, color, 2)
            cv2.putText(out, "GAP", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # Posición detectada: dibujar normalmente
            cv2.circle(out, (cx, cy), 8, (0, 255, 255), 2)

    # Jugadores (posición + esqueleto desde trackers)
    out = draw_players_from_info(out, players_info)

    # Mini-mapa (opcional)
    if show_minimap and homography_matrix is not None and inv_homography is not None:
        player_positions_px = None
        if players_info and getattr(players_info, "all_positions", None):
            player_positions_px = [p.get("px") for p in players_info.all_positions if p.get("px")]
        out = draw_minimap(
            out,
            homography_matrix,
            person_results=None,
            ball_results=None,
            minimap_size=200,
            position="top-right",
            inv_homography=inv_homography,
            ball_position=ball_info.position,
            trajectory_history=ball_info.trajectory_history if ball_info.trajectory_history else None,
            player_positions_px=player_positions_px,
            is_ball_interpolated=ball_info.is_interpolated,
        )

    return out


def save_heatmap(homography_matrix, inv_homography, video_out_path):
    """Wrapper para guardar el heatmap final (sin bounce_history)."""
    # Heatmap sin piques - solo minimap básico
    save_final_heatmap(homography_matrix, inv_homography, [], video_out_path)
