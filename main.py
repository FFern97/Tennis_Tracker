"""
Punto de entrada del sistema de tracking de tenis.
Loop principal legible: inferencia -> trackers -> visualización.
"""
import os
import cv2
import numpy as np
import torch

import config
from tracknet import BallTrackerNet
from court_detector import CourtDetector
from inference import TennisInference
from trackers import BallTracker, PersonTracker
from schema import FrameData, BallInfo, PlayersInfo
from visualization import render


def _ensure_scipy():
    try:
        from scipy.spatial import distance  # noqa: F401
    except ImportError:
        print("Error: pip install scipy")
        raise SystemExit(1)


def _next_output_path():
    counter = 1
    while True:
        name = f"{config.VIDEO_OUT_BASENAME}{counter}{config.VIDEO_OUT_EXTENSION}"
        path = os.path.join(config.VIDEO_OUT_FOLDER, name)
        if not os.path.exists(path):
            return path
        counter += 1


def _load_court_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(config.KEYPOINT_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def _setup_court_and_homography(court_detector, frame, frame_width, frame_height, frame_count):
    """Detecta cancha y devuelve (stored_keypoints, homography_matrix, inv_homography) cuando aplica."""
    averaged = court_detector.detect_keypoints(frame, frame_width, frame_height)
    stored_keypoints = None
    homography_matrix = None
    inv_homography = None

    if averaged is not None and frame_count == config.N_FRAMES_TO_AVERAGE:
        keypoints_list = []
        for kp in averaged:
            if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                keypoints_list.append((kp[0], kp[1]))
            elif isinstance(kp, np.ndarray) and len(kp) >= 2:
                keypoints_list.append((kp[0], kp[1]))
            else:
                keypoints_list.append((None, None))
        refined = court_detector.refine_keypoints(frame, keypoints_list)
        refined_list = []
        for kp in refined:
            x, y = kp[0], kp[1]
            if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
                refined_list.append((None, None))
            else:
                refined_list.append((float(x), float(y)))
        final_keypoints = court_detector.apply_homography(refined_list)
        court_detector.stored_keypoints = final_keypoints
        stored_keypoints = final_keypoints
        homography_matrix = court_detector.get_homography_matrix()
        if homography_matrix is not None:
            try:
                inv_homography = np.linalg.inv(homography_matrix)
            except np.linalg.LinAlgError:
                inv_homography = None
    else:
        stored_keypoints = court_detector.get_keypoints() if court_detector.get_keypoints() is not None else averaged

    return stored_keypoints, homography_matrix, inv_homography


def main():
    _ensure_scipy()
    # Crear carpetas necesarias si no existen (bootstrap para clonación)
    for dir_path in (
        config.VIDEO_OUT_FOLDER,
        os.path.dirname(config.VIDEO_IN_PATH),
        os.path.dirname(config.BALL_MODEL_PATH),
    ):
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    video_out_path = _next_output_path()

    # Modelos
    engine = TennisInference()
    keypoint_model, device = _load_court_model()
    court_detector = CourtDetector(
        keypoint_model=keypoint_model,
        device=device,
        keypoint_input_width=config.KEYPOINT_INPUT_WIDTH,
        keypoint_input_height=config.KEYPOINT_INPUT_HEIGHT,
        n_frames_to_average=config.N_FRAMES_TO_AVERAGE,
    )

    # Trackers
    ball_tracker = BallTracker()
    person_tracker = PersonTracker(max_interpolation_frames=config.PERSON_MAX_INTERPOLATION_FRAMES)

    # Video
    cap = cv2.VideoCapture(config.VIDEO_IN_PATH)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {config.VIDEO_IN_PATH}")
        raise SystemExit(1)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    print(f"Salida: {video_out_path}")

    inv_homography = None
    homography_matrix = None
    stored_keypoints = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}...")

        # Jerarquía de detección: Global -> Localizado (ROI/SAHI) -> Pure Vision (sin extrapolación)
        # Paso 1: Intentar detección global
        detections: FrameData = engine.predict_global(frame)
        
        # Paso 2: Si la detección global falla y existe una última posición, intentar detección localizada
        if not detections.ball:
            last_position = ball_tracker.get_last_position()
            if last_position is not None:
                localized_ball_detections = engine.predict_localized(frame, last_position)
                if localized_ball_detections:
                    detections.ball = localized_ball_detections
        
        # Paso 3: Si ambas detecciones fallan, ball_tracker retornará posición None (Pure Vision)

        # Homografía de cancha (solo actualiza cuando se completa el promediado)
        sk, hm, inv = _setup_court_and_homography(
            court_detector, frame, frame_width, frame_height, frame_count
        )
        if sk is not None:
            stored_keypoints = sk
        if hm is not None:
            homography_matrix = hm
        if inv is not None:
            inv_homography = inv
            person_tracker.set_homography(inv_homography)

        # Actualizar trackers
        ball_info: BallInfo = ball_tracker.update(
            frame_count,
            detections.ball,
            inv_homography=inv_homography,
            frame_height=frame_height,
        )
        players_info: PlayersInfo = person_tracker.update(
            detections.players,
            inv_homography=inv_homography,
            frame_number=frame_count,
        )

        # Dibujar todo
        annotated = render(
            frame,
            ball_info,
            players_info,
            court_keypoints=stored_keypoints,
            homography_matrix=homography_matrix,
            inv_homography=inv_homography,
            show_minimap=config.SHOW_MINIMAP,
        )
        out.write(annotated)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video guardado: {video_out_path}")


if __name__ == "__main__":
    main()
