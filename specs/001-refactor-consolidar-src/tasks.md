# tasks.md — 001-refactor-consolidar-src

### Pre-implementación
- **T000** (US: —) Generar `specs/001-refactor-consolidar-src/baseline-coverage.md` con % total y tabla por módulo de los 9 archivos afectados: `visualization.py`, `visualization_utils.py`, `trackers/ball_tracker.py`, `trackers/player_tracker.py`, `inference.py`, `court_detector.py`, `geometry_utils.py`, `tracknet.py`, `core/interfaces.py`. Incluir comando reproducible exacto. ✓ archivo existe antes de slice 1.
- **T001** (US-3) Validar entorno brownfield: `pytest -q` exit 0; `python tests/test_golden_master.py` exit 0.

### Slice 1 — visualization
- **T002** (US-1) `git mv` `visualization.py` + `visualization_utils.py` → `src/visualization/`; crear `src/__init__.py` y `src/visualization/__init__.py` vacíos. ✓ archivos en destino; sin `visualization.py` en raíz.
- **T003** (US-2, US-4) Imports full-path en call sites + tests; actualizar `.coveragerc`, `pytest.ini` (`pythonpath = . src`), `graph/domain.yaml`. ✓ pytest + golden exit 0; cov ≥ baseline; `grep pythonpath pytest.ini` muestra `pythonpath = . src`.
- **T004** (US-1) Commit `refactor(slice-1): mover visualization a src/`. ✓ `git log -1 --oneline` contiene mensaje.

### Slice 2 — trackers
- **T005** (US-1) `git mv` `trackers/` → `src/trackers/`. ✓ `git grep -E 'from trackers|import trackers'` → 0 hits.
- **T006** (US-2, US-4) Imports full-path + config + gates. ✓ pytest + golden + cov ≥ baseline módulos trackers.
- **T007** (US-1) Commit `refactor(slice-2): mover trackers a src/`. ✓ `git log -1`.

### Slice 3 — vision_tracking + core
- **T008** (US-1) Verificar deps circulares tracknet↔court_detector; `git mv` 4 módulos → `src/vision_tracking/`; `core/interfaces.py` → `src/core/interfaces.py`; `src/core/__init__.py` vacío. ✓ paths en destino.
- **T009** (US-2, US-4) Imports full-path cross-domain; eliminar alias `PersonTracker`; entrada en `DECISIONS.md` justificando eliminación del alias (referenciando patrón documentado en `existing-arch.md`); config + graph. ✓ gates verdes; callers usan `PlayerTracker`; DECISIONS.md actualizado.
- **T010** (US-1) Commit `refactor(slice-3): mover vision_tracking y core a src/`. ✓ `git log -1`.

### Slice 4 — legacy cleanup
- **T011** (US-5) `git grep` 0 refs a legacy; eliminar `tennis_tracker.py`, `yolo_person_detector.py`; entrada en `DECISIONS.md`. ✓ archivos ausentes; DECISIONS.md actualizado.
- **T012** (US-2, US-6) Quitar `sys.path.insert` en `main.py`/`app.py`; imports full-path en entry points; `pytest.ini` → `pythonpath = .`; actualizar `existing-arch.md` + `graph/domain.yaml`. ✓ `git grep sys.path.insert` → 0 en producción; `grep pythonpath pytest.ini` → `pythonpath = .` (sin `src`).
- **T013** (US-2) Migrar imports bare restantes a full-path para módulos no movidos (`src.schema`, `src.analytics`, `src.pipeline`, `src.data`, `src.detectors`) en cualquier archivo del repo. ✓ `git grep -E "^from (schema|analytics|pipeline|data|detectors)"` → 0 hits fuera de imports ya full-path bajo `src/` mismo.
- **T014** (US-4, US-6) Gate final + commit `refactor(slice-4): legacy cleanup y docs`. ✓ raíz sin módulos de dominio; pytest + golden + cov ≥ baseline.
