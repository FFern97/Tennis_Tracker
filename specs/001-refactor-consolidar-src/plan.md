# plan.md — 001-refactor-consolidar-src

## Stack (brownfield — `existing-arch.md`)
- Python 3, pip + `requirements.txt` (no pnpm)
- pytest ≥8, pytest-cov, golden master en `tests/test_golden_master.py`
- PyTorch/Ultralytics/OpenCV — sin cambios de dependencias

## source_root (estado actual → target)
- **Actual:** híbrido raíz + `src/`; `sys.path.insert` en entry points
- **Target:** dominio bajo `src/<dominio>/`; solo `main.py`, `app.py`, `config.py` en raíz; sin `sys.path.insert`

## Estructura objetivo
Ver `input.md` §5: `src/visualization/`, `src/trackers/`, `src/vision_tracking/`, `src/core/interfaces.py`; `schema/analytics/pipeline/data/detectors` sin mover.

## Moves por slice (`git mv`)
| Slice | Origen | Destino |
|-------|--------|---------|
| 1 | `visualization.py`, `visualization_utils.py` | `src/visualization/` |
| 2 | `trackers/` | `src/trackers/` |
| 3 | `inference.py`, `court_detector.py`, `geometry_utils.py`, `tracknet.py`, `core/interfaces.py` | `src/vision_tracking/`, `src/core/` |
| 4 | `tennis_tracker.py`, `yolo_person_detector.py` | eliminar |

## Crear (vacíos)
`src/__init__.py` (slice 1), `src/visualization/__init__.py`, `src/trackers/__init__.py`, `src/vision_tracking/__init__.py`, `src/core/__init__.py`, `baseline-coverage.md` (T000)

## Modificar por slice
- **Slice 1:** imports full-path para módulos movidos; `pytest.ini` → `pythonpath = . src` (coexistencia: full-path movidos + bare no movidos); `.coveragerc`, `graph/domain.yaml`; entry points solo imports mecánicos; `sys.path.insert` **permanece**
- **Slices 2–3:** imports full-path + config + graph; `pytest.ini` sigue en `pythonpath = . src`
- **Slice 4:** quitar `sys.path.insert`; imports full-path restantes (T012.5); `pytest.ini` → `pythonpath = .` (sin fallback bare); eliminar alias `PersonTracker`; `existing-arch.md`, `DECISIONS.md`, `graph/domain.yaml`

## Verificación por slice
`pytest -q` · `python tests/test_golden_master.py` · `pytest --cov` vs `baseline-coverage.md` · `git grep` paths viejos

## Commits
Directo en `main`: `refactor(slice-N): <descripción>`
