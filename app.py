import streamlit as st
import os
import pandas as pd
import io
import contextlib
from datetime import datetime, timezone

import config
from main import main as run_pipeline
from src.data.logger import SupabaseLogger

# Inicializar Logger para la pestaña de Auditoría
logger = SupabaseLogger()
HITL_REVIEWER_NAME = "Juan"


def _insert_annotation(stroke_id: str, label_human: str) -> None:
    logger.client.table("annotations").insert({
        "stroke_id": stroke_id,
        "label_human": label_human,
        "reviewer_name": HITL_REVIEWER_NAME,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }).execute()

st.set_page_config(page_title="Tennis Data Factory", page_icon="🎾", layout="wide")

# --- ESTILOS CUSTOM ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    .stroke-card { border: 1px solid #ddd; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

tab_ingesta, tab_auditoria = st.tabs(["📥 Ingesta de Videos", "🕵️ Auditoría HITL"])

# ==========================================
# PESTAÑA 1: INGESTA (Liberada de buffers)
# ==========================================
with tab_ingesta:
    st.header("Procesamiento de Videos")
    video_folder = os.path.dirname(config.VIDEO_IN_PATH)
    
    if os.path.exists(video_folder):
        available_videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        selected_video_name = st.selectbox("Elegí el video para procesar:", available_videos, key="ingesta_select")
        video_path = os.path.join(video_folder, selected_video_name)
        
        if st.button("🚀 Iniciar Pipeline", type="primary"):
            original_video_in = config.VIDEO_IN_PATH
            config.VIDEO_IN_PATH = video_path
            
            with st.status("⏳ Procesando... Mirá la consola de Cursor para ver el progreso frame por frame.", expanded=True) as status:
                st.info("💡 Tu terminal (la pantalla negra) te mostrará el avance real. Esto puede demorar varios minutos dependiendo del video.")
                
                try:
                    # Ejecutamos directo. Todo el print() del main.py saldrá en tu terminal de Cursor/VSCode.
                    run_pipeline() 
                    
                    st.success("¡Video procesado!")
                    status.update(label="Procesamiento completo", state="complete")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error durante el procesamiento: {e}")
                    status.update(label="Falló el procesamiento", state="error")
            
            config.VIDEO_IN_PATH = original_video_in
    else:
        st.warning(f"No se encontró la carpeta {video_folder}.")

# ==========================================
# PESTAÑA 2: AUDITORÍA HITL
# ==========================================
with tab_auditoria:
    st.header("Verificación de Golpes (Human-in-the-Loop)")
    
    try:
        videos_db = logger.client.table("videos").select("id, filename").execute()
        video_options = {v['filename']: v['id'] for v in videos_db.data}
        
        if not video_options:
            st.info("No hay videos en la base de datos todavía.")
        else:
            selected_v_name = st.selectbox("Seleccioná el video a auditar:", list(video_options.keys()))
            selected_v_id = video_options[selected_v_name]
            
            # 1. Cargar golpes ordenados por fecha de creación (seguro)
            res = logger.client.table("strokes")\
                .select("*")\
                .eq("video_id", selected_v_id)\
                .order("created_at")\
                .execute()
            
            strokes = res.data
            
            if not strokes:
                st.info("No hay golpes registrados para este video.")
            else:
                # Filtros rápidos
                pendientes = [s for s in strokes if s.get('requires_review')]
                st.metric("Golpes Totales", len(strokes), f"{len(pendientes)} requieren revisión")
                
                # Selector de golpe específico
                stroke_idx = st.number_input("Ir al golpe #", min_value=1, max_value=len(strokes), value=1) - 1
                curr_stroke = strokes[stroke_idx]
                
                # Extraer el frame_number de la columna impact_frame o del JSON kinematics
                frame_num = curr_stroke.get('impact_frame') or curr_stroke.get('kinematics', {}).get('frame_number', 0)
                
                # --- INTERFAZ DE AUDITORÍA ---
                col_vid, col_form = st.columns([2, 1])
                
                with col_vid:
                    st.subheader(f"Golpe en Frame: {frame_num}")
                    
                    out_path = os.path.join(config.VIDEO_OUT_FOLDER, selected_v_name)
                    
                    if os.path.exists(out_path):
                        start_sec = max(0, int(frame_num / 30) - 1)
                        nombre_archivo = os.path.basename(out_path)
                        url_estatica = f"/app/static/output_videos/{nombre_archivo}"
                        st.video(url_estatica, format="video/mp4", start_time=start_sec)
                    else:
                        st.warning(f"No se encontró el video local en: {out_path}")
                    
                    with st.expander("Ver JSON Cinemático"):
                        st.json(curr_stroke['kinematics'])

                with col_form:
                    st.write("### Clasificación IA")
                    st.write(f"**Lado:** {curr_stroke.get('side_detected', 'N/A')}")
                    st.write(f"**Zona:** {curr_stroke.get('zone_detected', 'N/A')}")
                    st.write(f"**Confianza:** {curr_stroke.get('confidence_score', 0):.2f}")
                    
                    st.divider()
                    st.write("### 🛠️ Corrección Humana")
                    
                    c1, c2 = st.columns(2)
                    if c1.button("✅ Todo OK", key="ok"):
                        label = curr_stroke.get("side_detected") or curr_stroke.get("kinematics", {}).get("side", "forehand")
                        logger.client.table("strokes").update({"requires_review": False}).eq("id", curr_stroke["id"]).execute()
                        _insert_annotation(curr_stroke["id"], label)
                        st.success("Verificado")
                        st.rerun()
                    
                    if c2.button("🗑️ Descartar", key="del"):
                        logger.client.table("strokes").delete().eq("id", curr_stroke['id']).execute()
                        st.warning("Golpe eliminado")
                        st.rerun()
                    
                    st.write("---")
                    current_side = curr_stroke.get('side_detected', 'forehand')
                    # Asegurar que el index no falle si viene nulo
                    radio_index = 0 if current_side == 'forehand' else 1
                    
                    new_side = st.radio("Corregir Lado:", ["forehand", "backhand"], index=radio_index)
                    
                    if st.button("💾 Guardar Corrección"):
                        logger.client.table("strokes").update({
                            "side_detected": new_side,
                            "requires_review": False,
                        }).eq("id", curr_stroke["id"]).execute()
                        _insert_annotation(curr_stroke["id"], new_side)
                        st.success(f"Cambiado a {new_side}")
                        st.rerun()

    except Exception as e:
        st.error(f"Error conectando a Supabase: {e}")