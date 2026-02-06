"""
Motor de inferencia: carga modelos YOLO y expone predict(frame) con detecciones estructuradas.
"""
import torch
from ultralytics import YOLO

from config import (
    PERSON_MODEL_VARIANT,
    BALL_MODEL_PATH,
    PERSON_CONFIDENCE,
    BALL_CONFIDENCE,
    PERSON_CLASS_ID,
    BALL_CLASS_ID,
    PERSON_IMGSZ,
)
from schema import FrameData, Detection, PlayerDetection


class TennisInference:
    """
    Encapsula la carga de modelos YOLO-pose y YOLO-ball.
    Expone predict(frame) que devuelve detecciones estructuradas (FrameData).
    """

    def __init__(self):
        self._person_model = YOLO(PERSON_MODEL_VARIANT)
        self._ball_model = YOLO(BALL_MODEL_PATH)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TennisInference: modelos cargados. Dispositivo: {self._device}")

    def predict(self, frame):
        """
        Ejecuta detección de pelota y jugadores sobre el frame.
        
        Args:
            frame: Imagen BGR (numpy array).
        
        Returns:
            FrameData con listas ball y players pobladas según las detecciones.
        """
        # Detección de jugadores (tracking con persistencia para IDs)
        person_results = self._person_model.track(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=PERSON_CONFIDENCE,
            persist=True,
            imgsz=PERSON_IMGSZ,
            verbose=False,
        )

        # Detección de pelota
        ball_results = self._ball_model.predict(
            source=frame,
            classes=[BALL_CLASS_ID],
            conf=BALL_CONFIDENCE,
            verbose=False,
        )

        ball_detections = self._parse_ball_results(ball_results)
        player_detections = self._parse_person_results(person_results)

        return FrameData(ball=ball_detections, players=player_detections)

    def _parse_ball_results(self, ball_results):
        """Convierte salida YOLO de pelota a lista de Detection."""
        out = []
        if not ball_results or not ball_results[0].boxes:
            return out
        for box in ball_results[0].boxes:
            conf = float(box.conf.item())
            if conf <= 0.0:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            out.append(Detection(x=x, y=y, conf=conf, id=None))
        return out

    def _parse_person_results(self, person_results):
        """Convierte salida YOLO-pose a lista de PlayerDetection."""
        out = []
        if not person_results or len(person_results) == 0 or person_results[0].boxes is None:
            return out
        result = person_results[0]
        boxes = result.boxes
        keypoints_data = result.keypoints.data if result.keypoints is not None else None

        for i, box in enumerate(boxes):
            track_id = int(box.id.item()) if box.id is not None else None
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf.item())
            center_x = (x1 + x2) / 2.0
            feet_y = y2
            kp = None
            if keypoints_data is not None and i < len(keypoints_data):
                kp_t = keypoints_data[i]
                if hasattr(kp_t, "cpu"):
                    kp = kp_t.cpu().numpy().copy()
                else:
                    kp = kp_t.copy()
            out.append(
                PlayerDetection(
                    x=center_x,
                    y=feet_y,
                    conf=conf,
                    id=track_id,
                    xyxy=(x1, y1, x2, y2),
                    keypoints=kp,
                )
            )
        return out
