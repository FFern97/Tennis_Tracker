"""
Configuración centralizada del sistema de tracking de tenis.
Todas las constantes en mayúsculas para uso en todo el proyecto.
"""

# --- Modelos ---
PERSON_MODEL_VARIANT = "yolov8n-pose.pt"
BALL_MODEL_PATH = "models/best.pt"
KEYPOINT_MODEL_PATH = "models/model_tennis_court_det.pt"

# --- Rutas de video (relativas al proyecto) ---
VIDEO_IN_PATH = "data/videos/test_video2.mp4"
VIDEO_OUT_FOLDER = "output_videos"
VIDEO_OUT_BASENAME = "output_tracking"
VIDEO_OUT_EXTENSION = ".mp4"

# --- Configuración de detección ---
PERSON_CONFIDENCE = 0.25
BALL_CONFIDENCE = 0.20
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 0

# --- Configuración de pose estimation ---
PERSON_IMGSZ = 1280  # Resolución para detectar jugador del fondo

# --- Configuración de keypoints de cancha ---
KEYPOINT_INPUT_WIDTH = 640
KEYPOINT_INPUT_HEIGHT = 360
N_FRAMES_TO_AVERAGE = 5

# --- Visualización ---
SHOW_MINIMAP = True

# --- EPI 2.0 (pelota) ---
BALL_VELOCITY_DAMPING = 0.75  # Factor acumulativo por frame de gap (velocidad decae rápido)
MAX_PREDICTION_FRAMES = 5  # Máximo de frames sin detección YOLO; si se supera, posición estrictamente None
BALL_REENTRY_ALPHA = 0.8  # Peso a la detección real en reentrada (snap rápido a trayectoria corregida)
BALL_VELOCITY_MAX_PX = 80  # Límite de velocidad (px/frame) al entrar en gap; si se supera se aplica damping de seguridad

# --- Configuración de interpolación de personas ---
PERSON_MAX_INTERPOLATION_FRAMES = 15  # Máximo de frames sin detección para interpolar
