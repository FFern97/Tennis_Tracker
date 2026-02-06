"""
Trackers de pelota y jugadores. Usan esquemas de schema.py.
Lógica simplificada: prioridad absoluta a YOLO, interpolación lineal básica.
"""
import cv2
import numpy as np

from schema import Detection, PlayerDetection, BallInfo, PlayersInfo
import config


class BallTracker:
    """
    Rastrea la pelota con interpolación conservadora (gap filling) y suavizado por media móvil.
    Prioridad absoluta a detecciones YOLO con conf > config.BALL_CONFIDENCE.
    Interpolación lineal simple de A a B cuando hay gap.
    """

    def __init__(self, buffer_size=10, smoothing_window=3, max_interpolation_frames=10):
        """
        Args:
            buffer_size: Tamaño del buffer para gap filling (esperar detección B).
            smoothing_window: Ventana para moving average (últimas N posiciones).
            max_interpolation_frames: Máximo de frames en un gap para aplicar gap filling bidireccional.
        """
        self.buffer_size = buffer_size
        self.smoothing_window = smoothing_window
        self.max_interpolation_frames = max_interpolation_frames
        self.detection_history = []
        self.trajectory_history = []
        self.gap_buffer = []
        self.last_detected_position = None
        self.last_detected_frame = None

    def update(
        self,
        frame_number: int,
        ball_detections: list,
        inv_homography=None,
        frame_height=None,
    ):
        """
        Actualiza el tracker con las detecciones de pelota (esquema Detection).
        Prioridad absoluta a YOLO: usa la primera detección con conf > config.BALL_CONFIDENCE.

        Args:
            frame_number: Número del frame actual.
            ball_detections: Lista de Detection (schema). Se usa la primera con conf > config.BALL_CONFIDENCE.
            inv_homography: No usado; se mantiene por compatibilidad.
            frame_height: No usado; se mantiene por compatibilidad.

        Returns:
            BallInfo: position, is_interpolated, trajectory_history.
        """
        # PRIORIDAD ABSOLUTA A YOLO: usar detección si conf > config.BALL_CONFIDENCE
        detected_position = None
        for d in ball_detections:
            if isinstance(d, Detection) and d.conf > config.BALL_CONFIDENCE:
                detected_position = (d.x, d.y)
                break

        if detected_position is not None:
            # Si había un gap en el buffer, hacer gap filling bidireccional
            if len(self.gap_buffer) > 0 and self.last_detected_position is not None:
                self._fill_gap_backward(frame_number, detected_position)

            # Aplicar suavizado con moving average
            smoothed_position = self._apply_smoothing(detected_position)

            # Agregar al historial
            self.detection_history.append(
                (frame_number, smoothed_position[0], smoothed_position[1], False)
            )
            self.trajectory_history.append(
                (frame_number, smoothed_position[0], smoothed_position[1])
            )

            # Mantener límites del historial
            if len(self.detection_history) > self.buffer_size:
                self.detection_history.pop(0)
            if len(self.trajectory_history) > 10:
                self.trajectory_history.pop(0)

            # Actualizar última detección y limpiar buffer
            self.last_detected_position = smoothed_position
            self.last_detected_frame = frame_number
            self.gap_buffer = []

            return BallInfo(
                position=smoothed_position,
                is_interpolated=False,
                trajectory_history=self.get_trajectory_history(),
            )

        # Si no hay detección, agregar al buffer de gap (NO interpolar hacia adelante)
        if self.last_detected_position is not None:
            self.gap_buffer.append(frame_number)

            # Si el buffer es muy grande, descartar
            if len(self.gap_buffer) > self.buffer_size:
                print(
                    f"[BallTracker] Frame {frame_number}: Buffer de gap excedido, descartando gap"
                )
                self.gap_buffer = []
                self.last_detected_position = None
                return BallInfo(
                    position=None,
                    is_interpolated=False,
                    trajectory_history=self.get_trajectory_history(),
                )

        # No hay detección y no hay historial suficiente
        return BallInfo(
            position=None,
            is_interpolated=False,
            trajectory_history=self.get_trajectory_history(),
        )

    def _apply_smoothing(self, current_position):
        """
        Media móvil sobre las últimas N posiciones detectadas (no interpoladas).
        Fórmula: avg_x = mean(x_i), avg_y = mean(y_i) para i en ventana.
        """
        if len(self.detection_history) < self.smoothing_window:
            return current_position

        # Obtener últimas N posiciones detectadas (no interpoladas)
        recent_positions = []
        for entry in self.detection_history[-self.smoothing_window :]:
            _, x, y, is_interp = entry
            if not is_interp:
                recent_positions.append((x, y))

        # Si no hay suficientes posiciones detectadas, retornar actual sin suavizar
        if len(recent_positions) < 2:
            return current_position

        # Agregar posición actual
        recent_positions.append(current_position)

        # Calcular promedio
        avg_x = np.mean([p[0] for p in recent_positions])
        avg_y = np.mean([p[1] for p in recent_positions])

        return (avg_x, avg_y)

    def _fill_gap_backward(self, end_frame, end_position):
        """
        Rellena un gap hacia atrás cuando tenemos detección A (última) y detección B (actual).
        Interpolación lineal simple: p(t) = start + (end - start) * t, t in [0,1].
        """
        if len(self.gap_buffer) == 0 or self.last_detected_position is None:
            return

        start_frame = self.last_detected_frame
        start_position = self.last_detected_position
        start_x, start_y = start_position
        end_x, end_y = end_position

        gap_size = end_frame - start_frame
        if gap_size <= 1:
            return

        # Limitar el gap al máximo permitido
        if gap_size > self.max_interpolation_frames:
            print(
                f"[BallTracker] Frame {end_frame}: Gap de {gap_size} frames excede el máximo ({self.max_interpolation_frames}), limitando"
            )
            gap_size = self.max_interpolation_frames
            start_frame = end_frame - gap_size

        print(f"[BallTracker] Frame {end_frame}: Interpolando gap de {gap_size} frames")

        # Interpolación lineal simple entre start y end
        for i in range(1, gap_size):
            frame_num = start_frame + i
            t = i / gap_size  # Parámetro [0, 1]

            # Interpolación lineal
            x = start_x + (end_x - start_x) * t
            y = start_y + (end_y - start_y) * t

            # Insertar en el historial en orden cronológico
            inserted = False
            for idx, entry in enumerate(self.detection_history):
                if entry[0] > frame_num:
                    self.detection_history.insert(idx, (frame_num, x, y, True))
                    inserted = True
                    break
            if not inserted:
                self.detection_history.append((frame_num, x, y, True))

            # Agregar a trajectory_history
            self.trajectory_history.append((frame_num, x, y))

            # Mantener límites
            if len(self.detection_history) > self.buffer_size:
                self.detection_history.pop(0)
            if len(self.trajectory_history) > 10:
                self.trajectory_history.pop(0)

    def get_last_position(self):
        """Última posición conocida (x, y) o None."""
        if self.detection_history:
            _, x, y, _ = self.detection_history[-1]
            return (x, y)
        return None

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
