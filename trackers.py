"""
Trackers de pelota y jugadores. Usan esquemas de schema.py.
BallTracker: Pure Vision - sin extrapolación, solo detección directa.
"""
import cv2
import numpy as np

from schema import Detection, PlayerDetection, BallInfo, PlayersInfo
import config

TRAJECTORY_HISTORY_SIZE = 10


class BallTracker:
    """
    Rastrea la pelota con enfoque Pure Vision.
    Sin extrapolación: si no hay detección, la posición es None.
    Mantiene last_position únicamente para referencia al ROI de inferencia localizada.
    """

    def __init__(self):
        """Inicializa el tracker sin lógica de extrapolación."""
        self.last_position = None
        self.last_frame = None
        self.trajectory_history = []

    def update(
        self,
        frame_number: int,
        ball_detections: list,
        inv_homography=None,
        frame_height=None,
    ):
        """
        Pure Vision: solo procesa detecciones directas.
        Si ball_detections está vacío, la posición es None.
        """
        detected_position = None
        for d in ball_detections:
            if isinstance(d, Detection) and d.conf > config.BALL_CONFIDENCE:
                detected_position = (d.x, d.y)
                break

        if detected_position is not None:
            # Detección encontrada: actualizar posición
            real_x, real_y = detected_position
            position = (real_x, real_y)
            
            self.last_position = (real_x, real_y)
            self.last_frame = frame_number
            self._append_trajectory(frame_number, position)
            
            return BallInfo(
                position=position,
                is_interpolated=False,
                trajectory_history=self.get_trajectory_history(),
            )
        else:
            # Sin detección: posición es None
            # Mantener last_position para referencia al ROI (no se actualiza)
            return BallInfo(
                position=None,
                is_interpolated=False,
                trajectory_history=self.get_trajectory_history(),
            )

    def _append_trajectory(self, frame_number: int, position: tuple):
        self.trajectory_history.append((frame_number, position[0], position[1]))
        while len(self.trajectory_history) > TRAJECTORY_HISTORY_SIZE:
            self.trajectory_history.pop(0)

    def get_last_position(self):
        """Última posición conocida (x, y) o None."""
        return self.last_position

    def get_trajectory_history(self):
        """Copia del historial de trayectoria para estela (últimos 10)."""
        return self.trajectory_history.copy()


class PersonTracker:
    """
    Rastrea jugadores con tracking directo basado en track_id de YOLO.
    Interpolación simple en píxeles cuando se pierde un track.
    """

    def __init__(self, max_interpolation_frames=None):
        self.max_interpolation_frames = (
            max_interpolation_frames
            if max_interpolation_frames is not None
            else config.PERSON_MAX_INTERPOLATION_FRAMES
        )
        self.tracks = {}
        self.inv_homography = None  # Frame -> Cancha (para coordenadas de cancha opcionales)
        self.homography = None  # Cancha -> Frame

    def set_homography(self, inv_homography):
        """
        Establece la matriz de homografía inversa (Frame -> Cancha).
        Opcional: se usa para almacenar coordenadas de cancha, pero la interpolación es en píxeles.
        """
        self.inv_homography = inv_homography
        if inv_homography is not None:
            try:
                self.homography = np.linalg.inv(inv_homography)
            except np.linalg.LinAlgError:
                self.homography = None

    def update(
        self,
        player_detections: list,
        inv_homography=None,
        frame_number=None,
    ):
        """
        Actualiza el tracker con detecciones de jugadores (esquema PlayerDetection).
        Tracking directo por track_id de YOLO. Interpolación simple en píxeles cuando se pierde un track.

        Args:
            player_detections: Lista de PlayerDetection (id, x, y, conf, keypoints, xyxy).
            inv_homography: Matriz 3x3 inversa (Frame -> Cancha); opcional.
            frame_number: Número de frame; opcional.

        Returns:
            PlayersInfo: active_tracks y all_positions.
        """
        if inv_homography is not None:
            self.set_homography(inv_homography)

        # Detectar personas activas en este frame
        active_track_ids = set()
        for d in player_detections:
            if not isinstance(d, PlayerDetection) or d.id is None:
                continue

            track_id = d.id
            active_track_ids.add(track_id)

            # Obtener posición en píxeles (centro horizontal, base del bounding box)
            position_px = (d.x, d.y)

            # Transformar a coordenadas de cancha si tenemos homografía (opcional)
            position_court = None
            if self.inv_homography is not None:
                pt = np.array([[[d.x, d.y]]], dtype=np.float32)
                out = cv2.perspectiveTransform(pt, self.inv_homography)
                if out is not None and len(out) > 0:
                    position_court = (float(out[0][0][0]), float(out[0][0][1]))

            # Obtener keypoints
            current_keypoints = d.keypoints.copy() if d.keypoints is not None else None

            # Actualizar o crear track
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "last_frame": frame_number or 0,
                    "last_position_px": position_px,
                    "last_position_court": position_court,
                    "last_keypoints": current_keypoints,
                    "history_px": [],
                    "history_court": [],
                    "history_keypoints": [],
                    "missing_frames": 0,
                }
            else:
                # Actualizar track existente
                self.tracks[track_id]["last_frame"] = frame_number or 0
                self.tracks[track_id]["last_position_px"] = position_px
                self.tracks[track_id]["last_position_court"] = position_court
                self.tracks[track_id]["last_keypoints"] = current_keypoints
                self.tracks[track_id]["missing_frames"] = 0

            # Agregar al historial
            self.tracks[track_id]["history_px"].append(position_px)
            self.tracks[track_id]["history_court"].append(position_court)
            if current_keypoints is not None:
                self.tracks[track_id]["history_keypoints"].append(
                    current_keypoints.copy()
                )

            # Mantener solo los últimos 30 frames en el historial
            while len(self.tracks[track_id]["history_px"]) > 30:
                self.tracks[track_id]["history_px"].pop(0)
                self.tracks[track_id]["history_court"].pop(0)
                if self.tracks[track_id]["history_keypoints"]:
                    self.tracks[track_id]["history_keypoints"].pop(0)

        # Procesar tracks que no aparecieron en este frame (interpolación)
        active_tracks_data = {}
        all_positions = []

        for track_id, track_data in self.tracks.items():
            if track_id in active_track_ids:
                # Track activo: usar posición detectada
                active_tracks_data[track_id] = {
                    "px": track_data["last_position_px"],
                    "court": track_data["last_position_court"],
                    "is_interpolated": False,
                    "keypoints": track_data["last_keypoints"],
                }
                all_positions.append(
                    {
                        "track_id": track_id,
                        "px": track_data["last_position_px"],
                        "court": track_data["last_position_court"],
                        "is_interpolated": False,
                    }
                )
            else:
                # Track perdido: incrementar contador
                track_data["missing_frames"] += 1

                # Si no excede el límite, interpolar en píxeles (simple)
                if track_data["missing_frames"] <= self.max_interpolation_frames:
                    if len(track_data["history_px"]) >= 2:
                        # Calcular velocidad promedio de los últimos 2 frames (en píxeles)
                        last_pos = track_data["history_px"][-1]
                        prev_pos = track_data["history_px"][-2]
                        vel_x = last_pos[0] - prev_pos[0]
                        vel_y = last_pos[1] - prev_pos[1]

                        # Calcular posición interpolada en píxeles
                        interpolated_px = (
                            last_pos[0] + vel_x * track_data["missing_frames"],
                            last_pos[1] + vel_y * track_data["missing_frames"],
                        )

                        # Transformar a coordenadas de cancha si tenemos homografía (Píxel → Cancha)
                        interpolated_court = None
                        if self.inv_homography is not None and interpolated_px is not None:
                            pt_px = np.array(
                                [[[interpolated_px[0], interpolated_px[1]]]], dtype=np.float32
                            )
                            pt_court = cv2.perspectiveTransform(pt_px, self.inv_homography)
                            if pt_court is not None and len(pt_court) > 0:
                                interpolated_court = (
                                    float(pt_court[0][0][0]),
                                    float(pt_court[0][0][1]),
                                )

                        # Estimar keypoints aplicando el mismo desplazamiento
                        estimated_keypoints = None
                        if (
                            track_data["last_keypoints"] is not None
                            and interpolated_px is not None
                        ):
                            # Calcular offset en píxeles
                            last_px = track_data["last_position_px"]
                            offset_x = interpolated_px[0] - last_px[0]
                            offset_y = interpolated_px[1] - last_px[1]

                            # Aplicar offset a todos los keypoints
                            estimated_keypoints = track_data["last_keypoints"].copy()
                            for kp_idx in range(len(estimated_keypoints)):
                                if estimated_keypoints[kp_idx][2] > 0:  # Visible
                                    estimated_keypoints[kp_idx][0] += offset_x
                                    estimated_keypoints[kp_idx][1] += offset_y

                        active_tracks_data[track_id] = {
                            "px": interpolated_px,
                            "court": interpolated_court,
                            "is_interpolated": True,
                            "keypoints": estimated_keypoints,
                        }
                        all_positions.append(
                            {
                                "track_id": track_id,
                                "px": interpolated_px,
                                "court": interpolated_court,
                                "is_interpolated": True,
                            }
                        )

        # Limpiar tracks que llevan demasiado tiempo perdidos
        tracks_to_remove = [
            tid
            for tid, data in self.tracks.items()
            if data["missing_frames"] > self.max_interpolation_frames
        ]
        for tid in tracks_to_remove:
            del self.tracks[tid]

        return PlayersInfo(
            active_tracks=active_tracks_data,
            all_positions=all_positions,
        )

    def get_track_history(self, track_id):
        """Historial (history_px, history_court, history_keypoints) para un track_id."""
        if track_id in self.tracks:
            t = self.tracks[track_id]
            return (
                t["history_px"].copy(),
                t["history_court"].copy(),
                t["history_keypoints"].copy(),
            )
        return (None, None, None)

    def get_all_tracks(self):
        """Copia del diccionario de tracks."""
        return self.tracks.copy()
