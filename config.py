"""
Configuración centralizada del sistema de tracking de tenis.
Todas las constantes en mayúsculas para uso en todo el proyecto.
"""

# --- Modelos ---
PERSON_MODEL_VARIANT = "yolov8n-pose.pt"
BALL_MODEL_PATH = "models/best.pt"
KEYPOINT_MODEL_PATH = "models/model_tennis_court_det.pt"

# --- Rutas de video (relativas al proyecto) ---
VIDEO_IN_PATH = "data/videos/test_video1.mp4"
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

# --- Detección localizada de pelota ---
BALL_ROI_SIZE = 320  # Tamaño del ROI (región de interés) para detección localizada de pelota

# --- Configuración de interpolación de personas ---
PERSON_MAX_INTERPOLATION_FRAMES = 15  # Máximo de frames sin detección para interpolar
PERSON_TRACK_HISTORY_MAX = 30  # Máximo de posiciones en historial por track (píxeles/cancha)
PERSON_KEYPOINT_VISIBILITY_THRESHOLD = 0.0  # Umbral de visibilidad de keypoint para desplazar en interpolación

# --- Tracker de pelota (estela / suavizado) ---
TRAJECTORY_HISTORY_SIZE = 10  # Puntos de trayectoria en pantalla para la estela
# Media móvil (pandas rolling) sobre x,y tras interpolación; 0 = desactivado (compat. golden master).
BALL_MOVING_AVERAGE_WINDOW = 0

# --- PERSISTENCIA ---
STUBS_FOLDER = "stubs"
BALL_STUBS_NAME = "ball_detections.pkl"
PLAYER_STUBS_NAME = "player_detections.pkl"
# Si False, main no sobrescribe .pkl ya existentes al finalizar inferencia (golden master / comparación).
OVERWRITE_STUBS = False

# --- Impacto / dataset (cinemática + Parquet + Supabase) ---
# Distancia máxima pelota–muñeca (px imagen) para candidato a golpe.
IMPACT_THRESHOLD_PX = 85.0
# Mínimo IoU entre caja del track (YOLO track) y caja pose (YoloPoseDetector) para copiar keypoints.
IMPACT_POSE_IOU_MIN = 0.22
# Confianza mínima del keypoint muñeca para considerar el impacto.
IMPACT_WRIST_CONF_MIN = 0.25
# Frames mínimos entre dos golpes registrados (evita disparos múltiples).
IMPACT_COOLDOWN_FRAMES = 15
# Duración del aviso visual tras detectar golpe.
IMPACT_OVERLAY_FRAMES = 18
# Carpeta base para secuencias Parquet por video: datasets/strokes/<video_key>/stroke_XXXX_fY.parquet
PARQUET_STROKES_FOLDER = "datasets/strokes"
