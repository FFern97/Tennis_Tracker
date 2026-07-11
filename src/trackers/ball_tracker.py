"""
Tracker de pelota: Pure Vision, estela y suavizado Pandas sobre secuencias de detecciones.
"""
import numpy as np
import pandas as pd

from src.core.interfaces import BaseTracker
from src.schema import Detection, BallInfo
import config


class BallTracker(BaseTracker):
    """
    Rastrea la pelota con enfoque Pure Vision.
    Sin extrapolación: si no hay detección, la posición es None.
    Mantiene last_position para referencia al ROI de inferencia localizada.
    """

    def __init__(self):
        self._ball_confidence = config.BALL_CONFIDENCE
        self._trajectory_history_size = config.TRAJECTORY_HISTORY_SIZE
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
            if isinstance(d, Detection) and d.conf > self._ball_confidence:
                detected_position = (d.x, d.y)
                break

        if detected_position is not None:
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
            return BallInfo(
                position=None,
                is_interpolated=False,
                trajectory_history=self.get_trajectory_history(),
            )

    def _append_trajectory(self, frame_number: int, position: tuple):
        self.trajectory_history.append((frame_number, position[0], position[1]))
        while len(self.trajectory_history) > self._trajectory_history_size:
            self.trajectory_history.pop(0)

    def get_last_position(self):
        """Última posición conocida (x, y) o None."""
        return self.last_position

    def get_trajectory_history(self):
        """Copia del historial de trayectoria para estela."""
        return self.trajectory_history.copy()

    def interpolate_ball_positions(self, ball_detections):
        """
        Suaviza la trayectoria de la pelota usando interpolación lineal con Pandas.
        """
        if not ball_detections:
            return ball_detections

        n_frames = len(ball_detections)
        x_arr = np.full(n_frames, np.nan, dtype=float)
        y_arr = np.full(n_frames, np.nan, dtype=float)
        conf_arr = np.full(n_frames, np.nan, dtype=float)

        for i, det_list in enumerate(ball_detections):
            if not det_list:
                continue
            for d in det_list:
                if isinstance(d, Detection):
                    x_arr[i] = d.x
                    y_arr[i] = d.y
                    conf_arr[i] = d.conf
                    break

        df = pd.DataFrame({"x": x_arr, "y": y_arr})
        df = df.interpolate(method="linear")
        df = df.ffill().bfill()

        w = getattr(config, "BALL_MOVING_AVERAGE_WINDOW", 0) or 0
        if w >= 2:
            df["x"] = df["x"].rolling(window=int(w), center=True, min_periods=1).mean()
            df["y"] = df["y"].rolling(window=int(w), center=True, min_periods=1).mean()

        result = []
        for i in range(n_frames):
            x_val = df["x"].iloc[i]
            y_val = df["y"].iloc[i]
            if np.isnan(x_val) or np.isnan(y_val):
                result.append([])
                continue
            conf = conf_arr[i] if not np.isnan(conf_arr[i]) else 1.0
            result.append([Detection(x=float(x_val), y=float(y_val), conf=float(conf), id=None)])

        return result
