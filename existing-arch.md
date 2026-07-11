# existing-arch.md — Estado del codebase

> Generado por /sdd-scan el 2026-06-27
> Commit base: 85b1e8e
> Actualizado post-refactor slice 4 (2026-07-10)
> Este archivo es DESCRIPTIVO (qué hay), no PRESCRIPTIVO (qué debería haber).
> Las restricciones acá son no negociables salvo decisión registrada en DECISIONS.md.

## Stack
- Lenguaje: Python 3 (versión específica TBD por el equipo; se pinneará en próximo update)
- Framework CV: PyTorch 2.9 + Ultralytics YOLOv8 + OpenCV 4.x
- UI: Streamlit (`app.py`, `.streamlit/config.toml`)
- Runtime: local (venv); GPU opcional vía CUDA
- Gestor de paquetes: pip (`requirements.txt`)

## source_root
Layout unificado bajo `src/` — imports full-path (`from src.<paquete>.<módulo> import …`):
- **Raíz del repo**: entry points (`main.py`, `app.py`), `config.py`, tests y artefactos SDD
- **`src/`**: todo el código de dominio — `schema.py`, `core/`, `vision_tracking/`, `trackers/`, `visualization/`, `analytics/`, `pipeline/`, `data/`, `detectors/`
- Sin `sys.path.insert` en producción; resolución vía paquete `src` con `pythonpath = .` en pytest

## Estructura
```
Tennis/
├── main.py, app.py, config.py
├── src/
│   ├── schema.py
│   ├── core/interfaces.py
│   ├── vision_tracking/   (inference, court_detector, geometry_utils, tracknet)
│   ├── trackers/          (ball_tracker, player_tracker)
│   ├── visualization/     (visualization, visualization_utils)
│   ├── analytics/kinematics.py
│   ├── pipeline/impact_utils.py
│   ├── data/logger.py
│   └── detectors/yolo_pose_detector.py
├── tests/
│   ├── smoke/       # smoke tests manuales (Supabase, requiere .env)
│   └── fixtures/court_images/  # PNG de canchas para court detection (~8 MB en git)
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
- `PlayerTracker` es la clase canónica de tracking de jugadores
- `HITL_REVIEWER_NAME` en `app.py` es revisor único fijo (no placeholder); multi-usuario requiere entrada en `DECISIONS.md`
- Cobertura pytest-cov omite glue y módulos de inferencia pesada — ver `.coveragerc`

## Entry points
- **CLI**: `main.py` — pipeline completo de tracking
- **UI**: `app.py` — Streamlit (ingesta + auditoría HITL)
- Scripts legacy `tennis_tracker.py` / `yolo_person_detector.py` eliminados (T011, ver `DECISIONS.md`)

## Tests
- Framework: pytest ≥8, pytest-cov
- Ubicación: `tests/`; `pythonpath = .` en `pytest.ini`
- Comando: `pytest` (reporte terminal + `htmlcov/`)
- Golden master: `python tests/test_golden_master.py`
- Smoke Supabase: `tests/smoke/smoke_test_supabase.py` (manual, requiere `.env`; `python -m tests.smoke.smoke_test_supabase`)
- Fixtures de imágenes: `tests/fixtures/court_images/` (15 PNG de canchas para desarrollo/tests de court detection; ~8 MB en git)

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
