"""
Detector YOLO-Pose (Pilar A): extrae esqueletos COCO (17 keypoints) para el motor cinemático.
"""
from __future__ import annotations

from typing import Any, List, Union

import numpy as np

from src.core.interfaces import BaseDetector
from src.schema import FrameData, PlayerDetection

DEFAULT_MODEL = "yolov8n-pose.pt"
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
PERSON_CLASS_ID = 0
COCO_POSE_KEYPOINTS = 17


FrameLike = np.ndarray
FrameInput = Union[FrameLike, List[FrameLike]]
DetectOutput = Union[FrameData, List[FrameData]]


class YoloPoseDetector(BaseDetector):
    """
    Pose estimation con Ultralytics; solo clase persona (COCO class_id == 0).

    ``detect`` acepta un frame BGR (numpy) o una lista de frames y devuelve
    ``FrameData`` con ``players`` (``PlayerDetection`` con 17 keypoints x,y,conf)
    o una lista de ``FrameData`` en el mismo orden que los frames de entrada.

    Args:
        model: Instancia opcional ya construida (p. ej. ``YOLO(...)``) o mock en tests.
               Si es ``None``, se carga ``model_path`` vía Ultralytics.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = DEFAULT_CONF,
        imgsz: int = DEFAULT_IMGSZ,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        self._conf = float(conf)
        self._imgsz = int(imgsz)

        if model is not None:
            self._model = model
            self._device = device or "cpu"
        else:
            import torch
            from ultralytics import YOLO

            self._model = YOLO(model_path)
            self._device = device or ("0" if torch.cuda.is_available() else "cpu")

    def detect(self, frame: FrameInput) -> DetectOutput:
        if isinstance(frame, np.ndarray):
            results = self._predict_batch([frame])
            return self._result_to_framedata(results[0])
        if isinstance(frame, list):
            if not frame:
                return []
            if not all(isinstance(f, np.ndarray) for f in frame):
                raise TypeError("Lista de frames: cada elemento debe ser np.ndarray BGR.")
            results = self._predict_batch(frame)
            return [self._result_to_framedata(r) for r in results]
        raise TypeError("frame debe ser np.ndarray (BGR) o lista de np.ndarray.")

    def export_to_onnx(self, *, imgsz: int | None = None, **kwargs: Any) -> str:
        """
        Exporta los pesos cargados a ONNX (útil para inferencia rápida en CPU).

        Los kwargs se reenvían a ``YOLO.export`` (p. ej. ``simplify=True``, ``half=False``).
        """
        export_imgsz = imgsz if imgsz is not None else self._imgsz
        out = self._model.export(format="onnx", imgsz=export_imgsz, **kwargs)
        return str(out)

    def _predict_batch(self, frames: List[np.ndarray]):
        return self._model.predict(
            source=frames,
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            imgsz=self._imgsz,
            verbose=False,
            device=self._device,
        )

    def _result_to_framedata(self, result) -> FrameData:
        players: List[PlayerDetection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return FrameData(ball=[], players=players)

        keypoints_tensor = result.keypoints.data if result.keypoints is not None else None

        for i, box in enumerate(result.boxes):
            if int(box.cls.item()) != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf.item())
            center_x = (x1 + x2) / 2.0
            feet_y = y2
            track_id = int(box.id.item()) if box.id is not None else None

            kp = self._extract_keypoints(keypoints_tensor, i)
            players.append(
                PlayerDetection(
                    x=center_x,
                    y=feet_y,
                    conf=conf,
                    id=track_id,
                    xyxy=(x1, y1, x2, y2),
                    keypoints=kp,
                )
            )

        return FrameData(ball=[], players=players)

    def _extract_keypoints(self, keypoints_tensor, index: int) -> np.ndarray | None:
        if keypoints_tensor is None or index >= len(keypoints_tensor):
            return None
        row = keypoints_tensor[index]
        kp = row.cpu().numpy().copy() if hasattr(row, "cpu") else np.asarray(row, dtype=np.float64).copy()
        kp = np.asarray(kp, dtype=np.float32).reshape(-1)
        need = COCO_POSE_KEYPOINTS * 3
        if kp.size < need:
            return None
        return kp[:need].reshape(COCO_POSE_KEYPOINTS, 3)
