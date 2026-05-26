"""
Punto de entrada del sistema de tracking de tenis.
Orquestador: inferencia (Pilar A), cinemática/impacto (Pilar B), persistencia Parquet/Supabase (Pilar C).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import os
import pickle
import shutil
from collections import deque
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np
import torch

import config
from core.interfaces import BaseDetector, BaseTracker
from data.logger import SupabaseLogger
from detectors.yolo_pose_detector import YoloPoseDetector
from pipeline.impact_utils import (
    framedata_row,
    merge_pose_keypoints,
    snapshot_framedata,
    try_detect_stroke,
)
from tracknet import BallTrackerNet
from court_detector import CourtDetector
from inference import YoloDetector
from trackers import BallTracker, PlayerTracker
from schema import FrameData, BallInfo, PlayersInfo
from visualization import render


def _ensure_scipy():
    try:
        from scipy.spatial import distance  # noqa: F401
    except ImportError:
        print("Error: pip install scipy")
        raise SystemExit(1)


def _output_video_path(video_in_path: Optional[str] = None) -> str:
    """Ruta de salida: mismo nombre que el video de entrada dentro de VIDEO_OUT_FOLDER."""
    in_path = video_in_path if video_in_path is not None else config.VIDEO_IN_PATH
    return os.path.join(config.VIDEO_OUT_FOLDER, os.path.basename(in_path))


def _reencode_video_h264(video_path: str) -> bool:
    """
    Convierte el MP4 escrito por OpenCV (mp4v) a H.264 para reproducción en Streamlit/navegador.
    Usa el FFmpeg empaquetado de moviepy (imageio-ffmpeg); no requiere instalación a nivel de SO.
    """
    if not os.path.isfile(video_path):
        print(f"Re-codificación omitida: no existe {video_path}")
        return False

    base, ext = os.path.splitext(video_path)
    temp_path = f"{base}_h264{ext or config.VIDEO_OUT_EXTENSION}"

    clip = None
    try:
        from moviepy.editor import VideoFileClip
    except ImportError as exc:
        print(f"Advertencia: moviepy no instalado; el video queda en codec OpenCV ({exc})")
        return False

    try:
        print(f"Re-codificando a H.264 (libx264): {video_path}")
        clip = VideoFileClip(video_path)
        clip.write_videofile(
            temp_path,
            codec="libx264",
            audio=False,
            logger=None,
            ffmpeg_params=["-movflags", "faststart", "-pix_fmt", "yuv420p"],
        )
        clip.close()
        clip = None
        shutil.move(temp_path, video_path)
        print(f"Video compatible con navegador: {video_path}")
        return True
    except Exception as exc:
        print(f"Advertencia: re-codificación H.264 falló ({exc}); se conserva el archivo mp4v.")
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass
        if os.path.isfile(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


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


def main(
    detector: Optional[BaseDetector] = None,
    ball_tracker: Optional[BaseTracker] = None,
    player_tracker: Optional[BaseTracker] = None,
):
    """
    Args:
        detector: BaseDetector (por defecto YoloDetector).
        ball_tracker: BaseTracker concreto para pelota (por defecto BallTracker).
        player_tracker: BaseTracker concreto para jugadores (por defecto PlayerTracker).
    """
    _ensure_scipy()

    for dir_path in (
        config.VIDEO_OUT_FOLDER,
        os.path.dirname(config.VIDEO_IN_PATH),
        os.path.dirname(config.BALL_MODEL_PATH),
        config.PARQUET_STROKES_FOLDER,
    ):
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    video_basename = os.path.basename(config.VIDEO_IN_PATH)
    video_out_path = _output_video_path(config.VIDEO_IN_PATH)
    video_key = os.path.splitext(video_basename)[0]
    stubs_subdir = os.path.join(config.STUBS_FOLDER, video_key)
    ball_stubs_path = os.path.join(stubs_subdir, config.BALL_STUBS_NAME)
    player_stubs_path = os.path.join(stubs_subdir, config.PLAYER_STUBS_NAME)
    read_from_stubs = os.path.isfile(ball_stubs_path) and os.path.isfile(player_stubs_path)

    if detector is None:
        detector = YoloDetector()

    yolo_pose_detector = None
    if not read_from_stubs:
        yolo_pose_detector = YoloPoseDetector(
            conf=config.PERSON_CONFIDENCE,
            imgsz=config.PERSON_IMGSZ,
        )

    supabase_logger = SupabaseLogger()

    keypoint_model, device = _load_court_model()
    court_detector = CourtDetector(
        keypoint_model=keypoint_model,
        device=device,
        keypoint_input_width=config.KEYPOINT_INPUT_WIDTH,
        keypoint_input_height=config.KEYPOINT_INPUT_HEIGHT,
        n_frames_to_average=config.N_FRAMES_TO_AVERAGE,
    )
    if ball_tracker is None:
        ball_tracker = BallTracker()
    if player_tracker is None:
        player_tracker = PlayerTracker()

    cap = cv2.VideoCapture(config.VIDEO_IN_PATH)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir {config.VIDEO_IN_PATH}")
        raise SystemExit(1)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    print(f"Salida: {video_out_path}")

    video_id = supabase_logger.get_or_create_video(
        video_basename,
        {
            "source_path": config.VIDEO_IN_PATH,
            "fps": float(fps),
            "frame_width": frame_width,
            "frame_height": frame_height,
        },
    )
    if video_id:
        print(f"Supabase video_id: {video_id}")
    else:
        print("Supabase: sin video_id (cliente ausente o error); se siguen guardando Parquet locales.")

    strokes_parquet_dir = os.path.join(config.PARQUET_STROKES_FOLDER, video_key)
    os.makedirs(strokes_parquet_dir, exist_ok=True)

    ball_detections_list = []
    player_detections_list = []

    if read_from_stubs:
        print("Stubs encontrados. Cargando detecciones desde cache...")
        with open(ball_stubs_path, "rb") as f:
            ball_detections_list = pickle.load(f)
        with open(player_stubs_path, "rb") as f:
            player_detections_list = pickle.load(f)
        ball_detections_list = ball_tracker.interpolate_ball_positions(ball_detections_list)
        print(f"Stubs cargados: {len(ball_detections_list)} frames de pelota, {len(player_detections_list)} frames de jugadores")
    else:
        print("Stubs no encontrados. Procesando inferencia completa...")
        os.makedirs(stubs_subdir, exist_ok=True)

    frame_buffer: deque = deque(maxlen=20)
    last_impact_frame = -10**9
    stroke_parquet_idx = 0
    stroke_overlay_text = ""
    stroke_overlay_until = 0

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

        if read_from_stubs:
            frame_index = frame_count - 1
            if frame_index < len(ball_detections_list) and frame_index < len(player_detections_list):
                detections = FrameData(
                    ball=ball_detections_list[frame_index],
                    players=player_detections_list[frame_index],
                )
            else:
                detections = FrameData(ball=[], players=[])
        else:
            base: FrameData = detector.detect(frame)
            if not base.ball:
                last_position = ball_tracker.get_last_position()
                if last_position is not None:
                    localized_ball_detections = detector.detect_localized(frame, last_position)
                    if localized_ball_detections:
                        base.ball = localized_ball_detections

            pose_fd: FrameData = yolo_pose_detector.detect(frame)
            merged_players = merge_pose_keypoints(
                base.players,
                pose_fd.players,
                config.IMPACT_POSE_IOU_MIN,
            )
            detections = FrameData(ball=base.ball, players=merged_players)

            ball_detections_list.append(detections.ball)
            player_detections_list.append(detections.players)

        frame_buffer.append((frame_count, snapshot_framedata(detections)))

        sk, hm, inv = _setup_court_and_homography(
            court_detector, frame, frame_width, frame_height, frame_count
        )
        if sk is not None:
            stored_keypoints = sk
        if hm is not None:
            homography_matrix = hm
        if inv is not None:
            inv_homography = inv
            player_tracker.set_homography(inv_homography)

        ball_info: BallInfo = ball_tracker.update(
            frame_count,
            detections.ball,
            inv_homography=inv_homography,
            frame_height=frame_height,
        )
        players_info: PlayersInfo = player_tracker.update(
            detections.players,
            inv_homography=inv_homography,
            frame_number=frame_count,
        )

        ball_pos = ball_info.position
        if (
            ball_pos is not None
            and (frame_count - last_impact_frame) >= config.IMPACT_COOLDOWN_FRAMES
        ):
            ball_conf = float(detections.ball[0].conf) if detections.ball else 0.55
            players_by_id = {p.id: p for p in detections.players if p.id is not None}
            stroke_candidates = []
            for tid, pdata in players_info.active_tracks.items():
                pl = players_by_id.get(tid)
                if pl is None:
                    continue
                kp_track = pdata.get("keypoints")
                effective = replace(pl, keypoints=kp_track) if kp_track is not None else pl
                stroke = try_detect_stroke(
                    effective,
                    (float(ball_pos[0]), float(ball_pos[1])),
                    threshold_px=config.IMPACT_THRESHOLD_PX,
                    wrist_conf_min=config.IMPACT_WRIST_CONF_MIN,
                    ball_conf=ball_conf,
                )
                if stroke is not None:
                    stroke_candidates.append((tid, stroke))

            if stroke_candidates:
                best_tid, best_stroke = min(stroke_candidates, key=lambda x: x[1]["distance_px"])
                last_impact_frame = frame_count
                stroke_parquet_idx += 1
                label = f"GOLPE: {best_stroke['side']} | {best_stroke['vertical_zone']}"
                stroke_overlay_text = label
                stroke_overlay_until = frame_count + config.IMPACT_OVERLAY_FRAMES

                kinematics = dict(best_stroke)
                kinematics["frame_number"] = frame_count
                kinematics["video_key"] = video_key
                kinematics["chosen_track_id"] = best_tid

                if video_id:
                    supabase_logger.log_stroke(
                        {
                            "video_id": video_id,
                            "confidence_score": best_stroke["confidence_score"],
                            "kinematics": kinematics,
                        }
                    )

                parquet_rows = [
                    framedata_row(fn, fd) for fn, fd in list(frame_buffer)
                ]
                pq_name = f"stroke_{stroke_parquet_idx:04d}_f{frame_count}.parquet"
                pq_path = os.path.join(strokes_parquet_dir, pq_name)
                if SupabaseLogger.save_stroke_sequence(parquet_rows, pq_path):
                    print(f"Dataset Parquet (ventana {len(parquet_rows)} frames): {pq_path}")

        stroke_banner = stroke_overlay_text if frame_count <= stroke_overlay_until else None

        annotated = render(
            frame,
            ball_info,
            players_info,
            court_keypoints=stored_keypoints,
            homography_matrix=homography_matrix,
            inv_homography=inv_homography,
            show_minimap=config.SHOW_MINIMAP,
            stroke_banner=stroke_banner,
        )
        out.write(annotated)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video guardado: {video_out_path}")
    _reencode_video_h264(video_out_path)

    if not read_from_stubs:
        stubs_already_on_disk = os.path.isfile(ball_stubs_path) and os.path.isfile(player_stubs_path)
        if stubs_already_on_disk and not config.OVERWRITE_STUBS:
            print(
                "Stubs ya existen en disco; no se sobrescriben (OVERWRITE_STUBS=False). "
                "Poné OVERWRITE_STUBS=True en config.py para regenerar."
            )
        else:
            print("Guardando stubs para futuras ejecuciones...")
            os.makedirs(stubs_subdir, exist_ok=True)
            with open(ball_stubs_path, "wb") as f:
                pickle.dump(ball_detections_list, f)
            with open(player_stubs_path, "wb") as f:
                pickle.dump(player_detections_list, f)
            print(
                f"Stubs guardados: {len(ball_detections_list)} frames de pelota, "
                f"{len(player_detections_list)} frames de jugadores"
            )


if __name__ == "__main__":
    main()
