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

# --- Configuración de interpolación de pelota ---
BALL_BUFFER_SIZE = 10  # Tamaño del buffer para gap filling (esperar detección B)
BALL_SMOOTHING_WINDOW = 3  # Ventana para moving average (últimas N posiciones)
MAX_INTERPOLATION_FRAMES = 10  # Máximo de frames en un gap para gap filling bidireccional

# --- Configuración de interpolación de personas ---
PERSON_MAX_INTERPOLATION_FRAMES = 15  # Máximo de frames sin detección para interpolar
