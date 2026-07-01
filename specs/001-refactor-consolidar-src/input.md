# input.md — Refactor: consolidar layout bajo src/

> Feature ID: `001-refactor-consolidar-src`
> Generado por `/sdd-refine` el 2026-06-27
> Borrador fuente: `drafts/refactor-consolidar-src.md`
> Modo: brownfield (`existing-arch.md` generado en scan `ff1bbed`)

---

## 1. PROBLEMA

El codebase tiene un layout **híbrido raíz + `src/`** documentado en `existing-arch.md`. Funciona hoy, pero genera fricción recurrente para el desarrollador:

- No es obvio dónde ubicar código nuevo de un dominio.
- Coexisten estilos de import (bare vs bajo `src/`) y el hack `sys.path.insert(0, 'src')` en entry points.
- `graph/domain.yaml` describe dominios de negocio, pero varios módulos viven fuera de `src/`, rompiendo la regla mental `dominio → src/<dominio>/`.

Esta feature resuelve esa fricción **sin cambiar comportamiento del pipeline** ni de la UI Streamlit. Es la primera feature SDD del proyecto; entra por el ciclo completo (no `/sdd-fix`) porque toca muchos archivos y exige verificación con golden master.

---

## 2. USUARIO

**Usuario primario:** el desarrollador del repo (solo dev, sin equipo externo ni reviewers).

**Qué necesita lograr:**

- Abrir el proyecto y ubicar la lógica de un dominio en `src/<dominio>/` sin excepciones.
- Agregar código nuevo siguiendo convención única de imports (`from src.<dominio>.<módulo> import …`).
- Ejecutar slices atómicos en `main` con gates verdes, sabiendo que cada commit deja el repo consistente.

**Fuera de alcance como "usuario":** el operador HITL en Streamlit. No debe notar cambios; eso es **invariante técnica** (pytest + golden master), no criterio "como usuario HITL puedo…".

---

## 3. DONE CRITERIA

### Criterios globales (post slice 4)

- [ ] **Raíz limpia:** cero archivos de dominio en la raíz; solo entry points `main.py`, `app.py`, `config.py` + artefactos SDD/config del proyecto.
- [ ] **`graph/domain.yaml`** refleja rutas reales bajo `src/`; sin nota de layout "híbrido".
- [ ] **`existing-arch.md`** sección `source_root` actualizada: layout unificado bajo `src/`, sin mención de híbrido ni `sys.path.insert`.
- [ ] **`sys.path.insert(0, 'src')` eliminado** de `main.py` y `app.py` (y cualquier otro código de producción).
- [ ] **Imports full-path:** todo import entre módulos de `src/` usa `from src.<paquete>.<módulo> import …`.
- [ ] **`git grep`** confirma 0 referencias a paths viejos de módulos movidos.
- [ ] **`pytest -q`** → exit 0.
- [ ] **`python tests/test_golden_master.py`** → exit 0, sin diffs.
- [ ] **`pytest --cov`** → total ≥ baseline; módulos afectados no bajan vs baseline (regla por slice, ver abajo).
- [ ] **`DECISIONS.md`** registra eliminación de `tennis_tracker.py` y `yolo_person_detector.py` (requerido por restricción SDD del scan).
- [ ] **Alias `PersonTracker` eliminado**; callers usan `PlayerTracker`.
- [ ] **`src/__init__.py`** existe y está vacío; sub-paquetes nuevos con `__init__.py` vacíos.

### Criterios por slice

Cada slice cierra con **un commit** en `main`:

```
refactor(slice-N): <descripción>
```

Gate obligatorio por slice:

1. `pytest -q` → exit 0
2. `python tests/test_golden_master.py` → exit 0, sin diffs
3. `pytest --cov` con dos verificaciones:
   - Total ≥ total del baseline (`specs/001-refactor-consolidar-src/baseline-coverage.md`)
   - Módulos afectados por la slice: cobertura por módulo no baja vs baseline (o vs valor tras slice anterior si la línea ya cambió por refactor previo)
4. `git grep` → 0 imports al path viejo del dominio movido en esa slice
5. `graph/domain.yaml` actualizado con nuevas rutas del dominio

**Slices:**

| # | Alcance | Destino |
|---|---------|---------|
| 1 | `visualization.py`, `visualization_utils.py` | `src/visualization/` |
| 2 | `trackers/` (carpeta entera) | `src/trackers/` |
| 3 | `inference.py`, `court_detector.py`, `geometry_utils.py`, `tracknet.py`, `core/interfaces.py` | `src/vision_tracking/` + `src/core/interfaces.py` |
| 4 | Legacy cleanup + cierre documentación | Eliminar `tennis_tracker.py`, `yolo_person_detector.py`; quitar `sys.path.insert`; actualizar `existing-arch.md`; entrada `DECISIONS.md` |

### Baseline de cobertura (primer task de `/sdd-implement`, **antes** del move del slice 1)

- Archivo: `specs/001-refactor-consolidar-src/baseline-coverage.md`
- Contenido: % total + tabla por módulo de los **9 archivos** movidos en slices 1–3:
  - `visualization.py`, `visualization_utils.py`
  - `trackers/ball_tracker.py`, `trackers/player_tracker.py`, `trackers/__init__.py`
  - `inference.py`, `court_detector.py`, `geometry_utils.py`, `tracknet.py`
  - (`core/interfaces.py` se cuenta en baseline aunque se mueva en slice 3)
- Incluir comando exacto reproducible (ej. `pytest --cov` con flags de `pytest.ini`).

### Rollback

Si una slice rompe algo no detectado por el gate: `git revert <hash>` del commit de esa slice; slices anteriores permanecen intactas.

---

## 4. OUT OF SCOPE (v1)

- Cambiar **firmas públicas** (agregar/quitar params, renombrar funciones/clases exportadas).
- Dividir funciones grandes o refactorizar APIs internas más allá de renombrar params/funciones **privadas** no exportadas.
- Convertir funciones libres ↔ métodos.
- Refactor de tests más allá de actualizar imports.
- Migrar a `pyproject.toml` / Poetry / uv.
- Pinnear versión Python mínima.
- Reorganizar `requirements.txt` en dev/prod.
- **Cambiar comportamiento** del pipeline, Streamlit o persistencia.
- **Tocar constantes de tuning** en `config.py` (solo paths de config global permanecen en raíz; valores MAYÚSCULAS intactos).
- **Reordenar lógica de orquestación** en `main.py` / `app.py`.
- **Tocar** `src/analytics/`, `src/pipeline/`, `src/data/`, `src/detectors/`, `src/schema.py` (ya están en `src/`, no se mueven).
- **Re-exports** en `__init__.py` (cambiarían API pública del paquete).
- Tooling AST automático (rope, bowler, libcst).
- Branch/PRs (solo dev local, commits directos en `main`).
- Tests manuales Streamlit como gate de aceptación.

---

## 5. RESTRICCIONES TÉCNICAS

### De `existing-arch.md` (no negociables)

- Constantes de tuning en `config.py` (MAYÚSCULAS).
- Orquestador depende de `BaseDetector` / `BaseTracker` (DIP).
- Dataclasses en `src/schema.py` sin mover.
- `SupabaseLogger` puede fallar sin tumbar el pipeline.
- Golden master: stubs `.pkl` en `stubs/<video_key>/`; `OVERWRITE_STUBS=False`.
- `HITL_REVIEWER_NAME` en `app.py` intacto.
- Exclusiones de `.coveragerc` se mantienen salvo ajuste de paths por moves.

### Convenciones acordadas en refine

| Tema | Regla |
|------|-------|
| **Branch** | Directo en `main`; 4 commits atómicos |
| **Moves** | `git mv` (preservar history) |
| **`__init__.py`** | Vacíos en `src/`, `src/visualization/`, `src/trackers/`, `src/vision_tracking/`, `src/core/`; crear `src/__init__.py` en slice 1 |
| **Imports** | Full package path: `from src.trackers.ball_tracker import BallTracker` |
| **`sys.path.insert`** | Permanece en slices 1–3; se elimina en slice 4 |
| **Entry points** | `main.py`, `app.py`, `config.py` en raíz; solo cambios mecánicos de import permitidos |
| **Migración imports** | Manual: `git grep` + búsqueda VS Code |
| **Slice 3 layout** | Plano en `src/vision_tracking/`; ABCs en `src/core/interfaces.py` (cross-domain) |
| **Slice 4 legacy** | `git grep` previo confirma 0 imports a `tennis_tracker` / `yolo_person_detector`; entrada en `DECISIONS.md` obligatoria |

### Estructura objetivo post-feature

```
src/
├── __init__.py                    # vacío (slice 1)
├── core/
│   ├── __init__.py                # vacío
│   └── interfaces.py              # slice 3
├── visualization/                 # slice 1
│   ├── __init__.py
│   ├── visualization.py
│   └── visualization_utils.py
├── trackers/                      # slice 2
│   ├── __init__.py
│   ├── ball_tracker.py
│   └── player_tracker.py
├── vision_tracking/               # slice 3
│   ├── __init__.py
│   ├── inference.py
│   ├── court_detector.py
│   ├── geometry_utils.py
│   └── tracknet.py
├── schema.py                      # sin mover
├── analytics/                     # sin mover
├── pipeline/                      # sin mover
├── data/                          # sin mover
└── detectors/                     # sin mover

main.py, app.py, config.py         # raíz (orchestration)
```

### Riesgos documentados

- Dependencias circulares potenciales `tracknet.py` ↔ `court_detector.py` — verificar antes del move slice 3.
- Pickles del golden master pueden depender de paths de módulo — gate `test_golden_master.py` detecta regresiones.
- `core/interfaces.py` usado por múltiples dominios — por eso vive en `src/core/`, no en `src/vision_tracking/`.

### Archivos de config a actualizar por slice

- `pytest.ini` (`pythonpath`, paths de cov si aplica)
- `.coveragerc` (paths omit/source tras moves)
- `graph/domain.yaml` (rutas `files` por dominio)
- `existing-arch.md` (slice 4, sección `source_root` + estructura)

---

## 6. UI / FLUJO

**No hay cambio de UI ni flujo de usuario.**

- Streamlit (`app.py`): pestañas Ingesta y Auditoría HITL se comportan igual.
- CLI (`main.py`): mismo pipeline frame a frame, misma salida de video y persistencia.
- **Invariante:** salidas del pipeline (video, stubs, Parquet, Supabase) idénticas al estado pre-refactor según golden master y tests unitarios.

**Flujo de trabajo del desarrollador (ciclo SDD canónico):**

1. `/sdd-generate` → produce `constitution.md`, `spec.md`, `plan.md`, `tasks.md` en `specs/001-refactor-consolidar-src/`
2. `/sdd-validate` → gate de cobertura vs `input.md`
3. `/sdd-implement` task 0: generar `specs/001-refactor-consolidar-src/baseline-coverage.md`
4. Slice 1 → commit → gate verde
5. Slice 2 → commit → gate verde
6. Slice 3 → commit → gate verde
7. Slice 4 → commit → gate verde + docs (`DECISIONS.md`, `existing-arch.md`, `graph/domain.yaml`)
8. `/sdd-checklist` → criterios manuales
9. `/sdd-review` → cierre feature

**Verificación manual opcional (no gate):** correr `streamlit run app.py` tras slice 4 para confirmar ingesta/auditoría; no bloquea cierre si gates automáticos pasan.

---

## Apéndice — Preguntas abiertas del draft (resueltas)

| Pregunta draft | Resolución |
|----------------|------------|
| Nombre subdirectorio vision_tracking | `src/vision_tracking/` |
| ¿`core/interfaces.py` en `src/core/` o distribuido? | `src/core/interfaces.py` centralizado |
| Estructura interna vision_tracking | Archivos planos (sin subcarpetas court/detection/geometry) |
| `git mv` vs copy+delete | `git mv` |
| `__init__.py` re-exports | No; vacíos |
| Branch strategy | Directo en `main`, 4 commits |
| Tooling | Manual con `git grep` |
