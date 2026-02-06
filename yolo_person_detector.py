import cv2
from ultralytics import YOLO
import os # To create output directory and list files
import re # To extract numbers from filenames

# --- Configuration ---
MODEL_VARIANT = 'yolov8n.pt' # Using YOLOv8 nano

# --- VIDEO PATHS ---
# Input video path (inside your data/videos folder)
VIDEO_IN_PATH = r'data\videos\test_video7.mp4' # <<< CHANGE THIS to your video file name

# --- Output Naming ---
VIDEO_OUT_FOLDER = 'output_videos'
VIDEO_OUT_BASENAME = 'output_tracking' # Base name for output files
VIDEO_OUT_EXTENSION = '.mp4'

CONFIDENCE_THRESHOLD = 0.4
CLASSES_TO_DETECT = [0] # 0 = person

# --- Main Execution ---
if __name__ == "__main__":
    # 0. Create output directory if it doesn't exist
    os.makedirs(VIDEO_OUT_FOLDER, exist_ok=True)

    # --- Find next available output filename ---
    counter = 1
    while True:
        potential_name = f"{VIDEO_OUT_BASENAME}{counter}{VIDEO_OUT_EXTENSION}"
        VIDEO_OUT_PATH = os.path.join(VIDEO_OUT_FOLDER, potential_name)
        if not os.path.exists(VIDEO_OUT_PATH):
            break # Found an available name
        counter += 1
    # --------------------------------------------

    # 1. Load the YOLOv8 model
    try:
        model = YOLO(MODEL_VARIANT)
        print(f"Modelo YOLOv8 '{MODEL_VARIANT}' cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo YOLO: {e}")
        exit()

    # 2. Open the input video file
    cap = cv2.VideoCapture(VIDEO_IN_PATH)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video de entrada en {VIDEO_IN_PATH}")
        exit()

    # Get video properties (width, height, fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video de entrada: {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # 3. Setup the output video writer (using the determined VIDEO_OUT_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, fps, (frame_width, frame_height))
    print(f"Guardando video de salida en: {VIDEO_OUT_PATH}") # Now shows the numbered filename

    frame_count = 0
    # 4. Process video frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Se terminó de leer el video o hubo un error.")
            break

        frame_count += 1
        # Process every Nth frame if desired (e.g., frame_count % 3 == 0)
        # For now, processing every frame
        if True: # Replace True with your frame skipping condition if needed
             print(f"Procesando frame {frame_count}...")

             # Perform tracking
             results = model.track(source=frame, classes=CLASSES_TO_DETECT, conf=CONFIDENCE_THRESHOLD, persist=True)

             # Get annotated frame
             if results:
                 annotated_frame = results[0].plot()
             else:
                 annotated_frame = frame

             # Write frame to output
             out.write(annotated_frame)

        # Optional real-time display (slows processing)
        # cv2.imshow("Seguimiento de Personas (Video)", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # 5. Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Procesamiento completo. Video guardado en {VIDEO_OUT_PATH}")