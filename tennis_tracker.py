import cv2
from ultralytics import YOLO
import os
import torch
import numpy as np
from tracknet import BallTrackerNet
from court_detector import CourtDetector
from visualization_utils import draw_keypoints, draw_ball_detections, draw_person_tracking, draw_minimap, save_final_heatmap
from trackers import BallTracker, PersonTracker

# --- Configuration ---
CONFIG = {
    # Modelos
    'PERSON_MODEL_VARIANT': 'yolov8n-pose.pt',
    'BALL_MODEL_PATH': 'models/best.pt',
    'KEYPOINT_MODEL_PATH': 'models/model_tennis_court_det.pt',
    
    # Rutas de video
    'VIDEO_IN_PATH': 'data/videos/test_video6.mp4',
    'VIDEO_OUT_FOLDER': 'output_videos',
    'VIDEO_OUT_BASENAME': 'output_tracking',
    'VIDEO_OUT_EXTENSION': '.mp4',
    
    # Configuración de detección
    'PERSON_CONFIDENCE': 0.25,
    'BALL_CONFIDENCE': 0.20,
    'PERSON_CLASS_ID': 0,
    'BALL_CLASS_ID': 0,
    
    # Configuración de pose estimation
    'PERSON_IMGSZ': 1280,  # Resolución para detectar jugador del fondo
    
    # Configuración de keypoints de cancha
    'KEYPOINT_INPUT_WIDTH': 640,
    'KEYPOINT_INPUT_HEIGHT': 360,
    'N_FRAMES_TO_AVERAGE': 5,
    
    # Visualización
    'SHOW_MINIMAP': True,
    
    # Configuración de interpolación de pelota
    'BALL_BUFFER_SIZE': 10,  # Tamaño del buffer para gap filling (esperar detección B)
    'BALL_SMOOTHING_WINDOW': 3,  # Ventana para moving average (últimas N posiciones)
    'MAX_INTERPOLATION_FRAMES': 10,  # Máximo de frames en un gap para aplicar gap filling bidireccional
    
    # Configuración de interpolación de personas
    'PERSON_MAX_INTERPOLATION_FRAMES': 15,  # Máximo de frames sin detección para interpolar
}

# --- Main Execution ---
if __name__ == "__main__":
    # Asegurarse de que scipy esté instalado
    try:
        from scipy.spatial import distance
    except ImportError as e:
        print(f"Error: Missing library 'scipy'. Please install it (e.g., pip install scipy)")
        exit()

    os.makedirs(CONFIG['VIDEO_OUT_FOLDER'], exist_ok=True)

    # Encontrar el siguiente nombre de archivo disponible
    counter = 1
    while True:
        potential_name = f"{CONFIG['VIDEO_OUT_BASENAME']}{counter}{CONFIG['VIDEO_OUT_EXTENSION']}"
        VIDEO_OUT_PATH = os.path.join(CONFIG['VIDEO_OUT_FOLDER'], potential_name)
        if not os.path.exists(VIDEO_OUT_PATH):
            break
        counter += 1

    # --- Load Models ---
    try:
        person_model = YOLO(CONFIG['PERSON_MODEL_VARIANT'])
        print(f"Modelo YOLOv8 para personas '{CONFIG['PERSON_MODEL_VARIANT']}' cargado.")
        ball_model = YOLO(CONFIG['BALL_MODEL_PATH'])
        print(f"Modelo YOLO para pelota '{CONFIG['BALL_MODEL_PATH']}' cargado.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {device}")
        keypoint_model = BallTrackerNet(out_channels=15)
        keypoint_model.load_state_dict(torch.load(CONFIG['KEYPOINT_MODEL_PATH'], map_location=device))
        keypoint_model.to(device)
        keypoint_model.eval()
        print(f"Modelo de puntos clave '{CONFIG['KEYPOINT_MODEL_PATH']}' cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        exit()

    # --- Inicializar CourtDetector ---
    court_detector = CourtDetector(
        keypoint_model=keypoint_model,
        device=device,
        keypoint_input_width=CONFIG['KEYPOINT_INPUT_WIDTH'],
        keypoint_input_height=CONFIG['KEYPOINT_INPUT_HEIGHT'],
        n_frames_to_average=CONFIG['N_FRAMES_TO_AVERAGE']
    )

    # --- Video I/O Setup ---
    cap = cv2.VideoCapture(CONFIG['VIDEO_IN_PATH'])
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video de entrada en {CONFIG['VIDEO_IN_PATH']}")
        exit()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video de entrada: {frame_width}x{frame_height} @ {fps:.2f} FPS")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Guardando video de salida en: {VIDEO_OUT_PATH}")

    # Variable para almacenar la matriz de homografía inversa (se calcula una sola vez)
    inv_homography = None

    # Inicializar trackers
    ball_tracker = BallTracker(
        buffer_size=CONFIG['BALL_BUFFER_SIZE'],
        smoothing_window=CONFIG['BALL_SMOOTHING_WINDOW'],
        max_interpolation_frames=CONFIG['MAX_INTERPOLATION_FRAMES']
    )
    
    person_tracker = PersonTracker(
        max_interpolation_frames=CONFIG['PERSON_MAX_INTERPOLATION_FRAMES']
    )

    frame_count = 0
    # Process Video
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Se terminó de leer el video o hubo un error.")
            break
        frame_count += 1
        print(f"Procesando frame {frame_count}...")

        # --- Seguimiento de Personas ---
        person_results = person_model.track(
            source=frame, 
            classes=[CONFIG['PERSON_CLASS_ID']], 
            conf=CONFIG['PERSON_CONFIDENCE'], 
            persist=True, 
            imgsz=CONFIG['PERSON_IMGSZ'],
            verbose=False
        )
        
        # Actualizar PersonTracker
        person_data = person_tracker.update(person_results, inv_homography, frame_count)
        
        # Dibujar tracking de personas
        annotated_frame = draw_person_tracking(frame, person_results)

        # --- Detección de Pelota con Interpolación ---
        ball_results = ball_model.predict(
            source=frame, 
            classes=[CONFIG['BALL_CLASS_ID']], 
            conf=CONFIG['BALL_CONFIDENCE'], 
            verbose=False
        )
        
        # Actualizar BallTracker
        ball_position, is_interpolated, event_type, is_bounce = ball_tracker.update(
            frame_count, ball_results, inv_homography, person_results, frame_height
        )
        
        # Dibujar pelota (detectada o interpolada)
        if ball_position is not None:
            center_x, center_y = ball_position
            if is_interpolated:
                # Posición interpolada (gap filled): dibujar en amarillo
                color = (0, 255, 255)  # Amarillo
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 8, color, 2)
                cv2.circle(annotated_frame, (int(center_x), int(center_y)), 8, color, 1)
                cv2.putText(annotated_frame, "GAP", (int(center_x) + 10, int(center_y) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # Posición detectada: dibujar normalmente
                annotated_frame = draw_ball_detections(annotated_frame, ball_results)
        
        # Dibujar indicador de pique si se detectó uno
        if is_bounce and ball_position is not None:
            center_x, center_y = ball_position
            # Dibujar círculo blanco grande
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 15, (255, 255, 255), -1)
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 15, (255, 255, 255), 3)
            # Dibujar texto 'BOUNCE'
            cv2.putText(annotated_frame, "BOUNCE", (int(center_x) - 40, int(center_y) - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Detección de Puntos Clave de la Cancha ---
        # Flujo: Detección -> Promediado -> Refinamiento -> Homografía -> Dibujo
        averaged_keypoints = court_detector.detect_keypoints(frame, frame_width, frame_height)
        
        # Aplicar refinamiento y homografía solo cuando se completen los frames de promediado
        if averaged_keypoints is not None and frame_count == CONFIG['N_FRAMES_TO_AVERAGE']:
            print("Aplicando refinamiento de puntos clave...")
            # Convertir a formato lista para refine_keypoints
            keypoints_list = []
            for kp in averaged_keypoints:
                if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                    keypoints_list.append((kp[0], kp[1]))
                elif isinstance(kp, np.ndarray) and len(kp) >= 2:
                    keypoints_list.append((kp[0], kp[1]))
                else:
                    keypoints_list.append((None, None))
            
            refined_kps = court_detector.refine_keypoints(frame, keypoints_list)
            
            # Convertir refined_kps (array numpy) a lista de tuplas para apply_homography
            refined_keypoints_list = []
            for kp in refined_kps:
                x_val = kp[0]
                y_val = kp[1]
                # Verificar si el valor es None o NaN
                if (x_val is None or y_val is None or 
                    (isinstance(x_val, float) and np.isnan(x_val)) or 
                    (isinstance(y_val, float) and np.isnan(y_val))):
                    refined_keypoints_list.append((None, None))
                else:
                    refined_keypoints_list.append((float(x_val), float(y_val)))
            
            # Aplicar homografía con los puntos refinados
            print("Aplicando corrección de homografía a puntos refinados...")
            final_keypoints = court_detector.apply_homography(refined_keypoints_list)
            
            # Calcular la matriz de homografía inversa una sola vez
            homography_matrix = court_detector.get_homography_matrix()
            if homography_matrix is not None:
                try:
                    inv_homography = np.linalg.inv(homography_matrix)
                    print("Matriz de homografía inversa calculada para el mini-mapa.")
                    # Actualizar PersonTracker con la homografía
                    person_tracker.set_homography(inv_homography)
                except np.linalg.LinAlgError:
                    print("Advertencia: No se pudo calcular la inversa de la matriz de homografía.")
                    inv_homography = None
            
            # Actualizar stored_keypoints con los puntos finales
            court_detector.stored_keypoints = final_keypoints
            stored_keypoints = final_keypoints
        else:
            # Usar los puntos almacenados (ya procesados) o los promediados si aún no se ha aplicado refinamiento
            stored_keypoints = court_detector.get_keypoints() if court_detector.get_keypoints() is not None else averaged_keypoints
        
        # --- Dibujar Puntos Clave ---
        annotated_frame = draw_keypoints(annotated_frame, stored_keypoints)
        
        # --- Dibujar Mini-mapa (solo si está habilitado) ---
        if CONFIG['SHOW_MINIMAP'] and inv_homography is not None:
            homography_matrix = court_detector.get_homography_matrix()
            if homography_matrix is not None:
                # Obtener historial de trayectoria para la estela
                trajectory_history = ball_tracker.get_trajectory_history()
                
                # Pasar la posición de la pelota (detectada o interpolada) al mini-mapa
                annotated_frame = draw_minimap(
                    annotated_frame, 
                    homography_matrix, 
                    person_results, 
                    ball_results,
                    minimap_size=200,
                    position='top-right',
                    inv_homography=inv_homography,
                    ball_position=ball_position,
                    trajectory_history=trajectory_history,
                    is_bounce=is_bounce
                )

        # --- Write Output Frame ---
        out.write(annotated_frame)

        # Optional real-time display
        # cv2.imshow("Tennis Tracking", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # --- Release Resources ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Procesamiento completo. Video guardado en {VIDEO_OUT_PATH}")
    
    # --- Generar Heatmap Final ---
    if inv_homography is not None:
        homography_matrix = court_detector.get_homography_matrix()
        if homography_matrix is not None:
            bounce_history = ball_tracker.get_bounce_history()
            save_final_heatmap(homography_matrix, inv_homography, bounce_history, VIDEO_OUT_PATH)
