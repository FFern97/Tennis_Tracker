# existing-arch.md — Estado del codebase

> Generado por /sdd-scan el 2026-06-27
> Commit base: 85b1e8e
> Este archivo es DESCRIPTIVO (qué hay), no PRESCRIPTIVO (qué debería haber).
> Las restricciones acá son no negociables salvo decisión registrada en DECISIONS.md.

## Stack
- Lenguaje: Python 3 (versión específica TBD por el equipo; se pinneará en próximo update)
- Framework CV: PyTorch 2.9 + Ultralytics YOLOv8 + OpenCV 4.x
- UI: Streamlit (`app.py`, `.streamlit/config.toml`)
- Runtime: local (venv); GPU opcional vía CUDA
- Gestor de paquetes: pip (`requirements.txt`)

## source_root
Layout híbrido definitivo — no hay paquete instalable único:
- **Raíz del repo**: orquestación, config, inferencia modular (`inference.py`, `core/`, `trackers/`, `court_detector.py`, `geometry_utils.py`, `visualization*.py`)
- **`src/`**: pilares B/C — `schema.py`, `analytics/`, `pipeline/`, `data/`, `detectors/`
- `main.py` y `app.py` insertan `src/` en `sys.path` al arrancar

## Estructura
```
Tennis/
├── main.py, app.py, config.py
├── core/interfaces.py
├── inference.py, court_detector.py, geometry_utils.py, tracknet.py
├── trackers/
├── src/
│   ├── schema.py
│   ├── detectors/yolo_pose_detector.py
│   ├── analytics/kinematics.py
│   ├── pipeline/impact_utils.py
│   └── data/logger.py
├── tests/
├── models/          # .pt (gitignored)
├── data/videos/     # entrada (gitignored *.mp4)
├── static/output_videos/
├── datasets/strokes/  # Parquet (gitignored)
├── stubs/             # pickle (gitignored)
└── SDD: CLAUDE.md, graph/, specs/, drafts/, metrics/
```

## Patrones inquebrantables
- Constantes de tuning en `config.py` (MAYÚSCULAS); no hardcodear paths de modelos fuera de ahí
- Orquestador (`main.py`) depende de `BaseDetector` / `BaseTracker`; implementaciones inyectables en tests
- Datos entre módulos via dataclasses en `src/schema.py`
- Fallos de Supabase no tumbar el pipeline: `SupabaseLogger` devuelve `None` y loguea
- Golden master: stubs `.pkl` en `stubs/<video_key>/`; `OVERWRITE_STUBS=False` preserva referencias
- `PlayerTracker` es canónica; `PersonTracker` es alias de compatibilidad
- `HITL_REVIEWER_NAME` en `app.py` es revisor único fijo (no placeholder); multi-usuario requiere entrada en `DECISIONS.md`
- Cobertura pytest-cov omite scripts legacy y glue — ver `.coveragerc`

## Scripts legacy
- `tennis_tracker.py` y `yolo_person_detector.py` están presentes como histórico
- **Restricción SDD**: comandos futuros no los modifican salvo decisión explícita en `DECISIONS.md`
- Entry point activo: `main.py` (CLI) y `app.py` (Streamlit)

## Tests
- Framework: pytest ≥8, pytest-cov
- Ubicación: `tests/`; `pythonpath = src` en `pytest.ini`
- Comando: `pytest` (reporte terminal + `htmlcov/`)
- Golden master: `python tests/test_golden_master.py`
- Smoke Supabase: `smoke_test_supabase.py` (manual, requiere `.env`)

## Persistencia / Data
- **Supabase** (opcional): tablas `videos`, `strokes`, `annotations`; credenciales en `.env`
- **Parquet**: `datasets/strokes/<video_key>/` vía `SupabaseLogger.save_stroke_sequence`
- **Stubs**: cache pickle en `stubs/`
- **Modelos/videos**: fuera de Git (`.gitignore`: `*.pt`, `*.mp4`, `.env`)

## Estado / Estilos / Integraciones
- Estado: sesión Streamlit en `app.py`; sin state manager global
- Estilos: CSS inline mínimo en pestañas Streamlit
- Integraciones: Supabase Python SDK, Ultralytics, moviepy/ffmpeg (H.264 post-proceso)

## Restricciones de deploy / entorno
- Target actual: solo dev local. Otros targets (Docker, cloud GPU) requieren decisión registrada en `DECISIONS.md`
- Sin CI/CD en repo
- Modelos manuales en `models/`; videos en `data/videos/`
- Streamlit sirve estáticos desde `static/output_videos/` (`enableStaticServing = true`)

## Drift tracking
- Generado contra commit: 85b1e8e75bc1e87ee7180ee30067f871eef9165b
- Re-scan sugerido si: cambian dependencias mayores, se agregan/eliminan carpetas top-level, o pasan >2 sprints
