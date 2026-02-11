"""
Motor de inferencia: carga modelos YOLO y expone predict(frame) con detecciones estructuradas.
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

from config import (
    PERSON_MODEL_VARIANT,
    BALL_MODEL_PATH,
    PERSON_CONFIDENCE,
    BALL_CONFIDENCE,
    PERSON_CLASS_ID,
    BALL_CLASS_ID,
    PERSON_IMGSZ,
    BALL_ROI_SIZE,
)
from schema import FrameData, Detection, PlayerDetection


class TennisInference:
    """
    Encapsula la carga de modelos YOLO-pose y YOLO-ball.
    Expone predict_global(frame) para detección global y predict_localized(frame, last_pos) para detección localizada.
    """

    def __init__(self):
        self._person_model = YOLO(PERSON_MODEL_VARIANT)
        self._ball_model = YOLO(BALL_MODEL_PATH)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sahi_model = None  # Inicialización lazy de SAHI
        print(f"TennisInference: modelos cargados. Dispositivo: {self._device}")

    def predict_global(self, frame):
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

    def predict_localized(self, frame, last_pos):
        """
        Ejecuta detección localizada de pelota en un ROI de 320x320 alrededor de last_pos.
        Primero intenta YOLO normal en el ROI, luego SAHI si YOLO falla.
        
        Args:
            frame: Imagen BGR completa (numpy array).
            last_pos: Tupla (x, y) con la última posición conocida de la pelota.
        
        Returns:
            Lista de Detection con detecciones de pelota (coordenadas ajustadas al frame completo).
        """
        if last_pos is None:
            return []
        
        x_center, y_center = last_pos
        roi_half = BALL_ROI_SIZE // 2
        
        # Calcular ROI asegurando que esté dentro de los límites del frame
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, int(x_center - roi_half))
        y1 = max(0, int(y_center - roi_half))
        x2 = min(frame_width, int(x_center + roi_half))
        y2 = min(frame_height, int(y_center + roi_half))
        
        # Si el ROI es muy pequeño, retornar vacío
        if (x2 - x1) < 50 or (y2 - y1) < 50:
            return []
        
        # Recortar ROI
        roi = frame[y1:y2, x1:x2]
        
        # Paso 1: Intentar YOLO normal en el ROI
        ball_results = self._ball_model.predict(
            source=roi,
            classes=[BALL_CLASS_ID],
            conf=BALL_CONFIDENCE,
            verbose=False,
        )
        
        ball_detections = self._parse_ball_results(ball_results)
        
        # Si YOLO encontró detecciones, ajustar coordenadas al frame completo y retornar
        if ball_detections:
            for det in ball_detections:
                det.x += x1
                det.y += y1
            return ball_detections
        
        # Paso 2: Si YOLO falló, intentar SAHI en el ROI
        # Inicialización lazy del modelo SAHI
        if self._sahi_model is None:
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=BALL_MODEL_PATH,
                confidence_threshold=BALL_CONFIDENCE,
                device=self._device
            )
        
        # Aplicar SAHI slicing solo en el ROI
        # Usar slice de 160 para procesar 4 sub-cuadrantes dentro del ROI de 320, aumentando resolución efectiva
        sahi_result = get_sliced_prediction(
            roi,
            self._sahi_model,
            slice_height=160,
            slice_width=160,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=False
        )
        
        # Parsear resultados de SAHI y ajustar coordenadas al frame completo
        sahi_detections = []
        if sahi_result and hasattr(sahi_result, 'object_prediction_list'):
            for obj_pred in sahi_result.object_prediction_list:
                if obj_pred.score.value >= BALL_CONFIDENCE:
                    # Obtener bbox en formato (x1, y1, x2, y2)
                    bbox = obj_pred.bbox
                    x1_sahi = bbox.minx
                    y1_sahi = bbox.miny
                    x2_sahi = bbox.maxx
                    y2_sahi = bbox.maxy
                    
                    # Calcular centro
                    center_x = (x1_sahi + x2_sahi) / 2.0
                    center_y = (y1_sahi + y2_sahi) / 2.0
                    
                    # Ajustar coordenadas al frame completo
                    center_x += x1
                    center_y += y1
                    
                    sahi_detections.append(
                        Detection(
                            x=center_x,
                            y=center_y,
                            conf=obj_pred.score.value,
                            id=None
                        )
                    )
        
        return sahi_detections

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
