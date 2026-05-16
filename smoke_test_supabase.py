import sys
from pathlib import Path

# Añadimos /src al path para que el entorno reconozca los módulos
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from data.logger import SupabaseLogger

def test_live_connection():
    print("🚀 Iniciando prueba de conexión con Supabase...")
    
    try:
        logger = SupabaseLogger()
        
        # 1. Probar registro de video
        video_id = logger.get_or_create_video(
            "test_video_001.mp4", 
            {"resolution": "1080p", "source": "smoke_test", "player": "Facundo"}
        )
        
        if video_id:
            print(f"✅ Video registrado/encontrado con ID: {video_id}")
        else:
            print("❌ El logger no devolvió un ID de video. Revisar credenciales en .env")
            return

        # 2. Probar registro de golpe (Stroke)
        stroke_data = {
            "video_id": video_id,
            "impact_frame": 100,      # El frame exacto del golpe
            "frame_start": 85,       # Inicio de la ventana para la LSTM
            "frame_end": 105,        # Fin de la ventana
            "confidence_score": 0.85,
            "kinematics": {
                "side": "forehand",
                "zone": "mid",
                "velocity": [12.5, -3.2]
            }
        }
        
        res = logger.log_stroke(stroke_data)
        if res:
            print("🔥 ¡Éxito! El golpe se registró correctamente en la tabla 'strokes'.")
        else:
            print("❌ No se pudo registrar el golpe.")

    except Exception as e:
        print(f"💥 Error inesperado durante el test: {e}")

if __name__ == "__main__":
    test_live_connection()