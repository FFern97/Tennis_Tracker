# Draft — Refactor: consolidar layout bajo src/

## Contexto

El scan brownfield (commit 85b1e8e) documentó el layout actual como
"híbrido raíz+src/" descriptivo. Funciona, pero crea fricción para features
futuras: dónde poner código nuevo, qué imports usar, qué patrones seguir.

Esta es la primera feature SDD del proyecto. La metí por el ciclo
completo (no por /sdd-fix) porque toca demasiados archivos y requiere
verificación rigurosa con golden master.

## Objetivo

Unificar todo el código de dominio bajo `src/` siguiendo los dominios
del graph (`graph/domain.yaml`). Entry points (`main.py`, `app.py`,
`config.py`) quedan en raíz. Scripts legacy se eliminan.

Después de la feature, abrir un archivo nuevo del proyecto debe ser
predecible: domain → src/<dominio>/<archivo>.py.

## Alcance — 4 slices secuenciales

Cada slice = 1 commit atómico (mover archivos + actualizar imports +
tests verdes + golden master OK + cobertura ≥ baseline + actualizar
graph/domain.yaml). Si una slice rompe, no destruye las anteriores.

1. **Slice 1 — visualization**
   `visualization.py` + `visualization_utils.py` → `src/visualization/`

2. **Slice 2 — trackers**
   `trackers/` (carpeta entera) → `src/trackers/`

3. **Slice 3 — vision_tracking core**
   `inference.py`, `court_detector.py`, `geometry_utils.py`, `tracknet.py`,
   `core/interfaces.py` → `src/vision_tracking/` (estructura interna
   TBD durante /sdd-refine)

4. **Slice 4 — legacy cleanup**
   Eliminar `tennis_tracker.py` y `yolo_person_detector.py`.
   Requiere entrada en `DECISIONS.md` (el scan los marcó como
   "no modificar salvo decisión registrada") y verificación previa
   con `git grep` de que nadie los importa.

## En scope

- Mover archivos a `src/<dominio>/` con cut-clean atómico
- Renombrar parámetros internos para consistencia (frame_idx vs idx)
- Renombrar funciones privadas / internas (no exportadas)
- Convertir todo a snake_case si quedan inconsistencias
- Eliminar el alias `PersonTracker = PlayerTracker`
- Eliminar código muerto detectado (branches no alcanzables, imports no usados)
- Actualizar `pytest.ini` (`pythonpath`), `.coveragerc`, `graph/domain.yaml`
- Eliminar el hack de `sys.path.insert(0, 'src')` en `main.py` y `app.py`
  si el nuevo layout lo hace innecesario
- Actualizar imports en TODOS los call sites en el mismo commit del move

## Fuera de scope (v1)

- Cambios de firmas públicas (agregar/quitar params, cambiar nombres
  de funciones/clases exportadas)
- Dividir funciones grandes en sub-funciones
- Convertir funciones libres a métodos o viceversa
- Refactor de APIs internas más allá del renombrado de params
- Refactor de tests más allá de actualizar imports
- Migrar a `pyproject.toml` / Poetry / uv
- Pinneo de versión Python (queda pendiente en existing-arch.md)
- Reorganizar `requirements.txt` en dev/prod
- Tocar `main.py`, `app.py`, `config.py` (entry points + config global
  quedan en raíz, son parte del dominio `orchestration`)
- Tocar archivos bajo `src/analytics/`, `src/pipeline/`, `src/data/`,
  `src/detectors/`, `src/schema.py` (ya están en src/, no se mueven)

## Restricciones técnicas (de existing-arch.md, no negociables)

- Constantes de tuning siguen en `config.py` (MAYÚSCULAS)
- Orquestador depende de abstracciones `BaseDetector` / `BaseTracker`
- Dataclasses de `src/schema.py` mantienen su lugar
- `SupabaseLogger` puede fallar sin tumbar pipeline
- Golden master: stubs `.pkl` en `stubs/<video_key>/` con
  `OVERWRITE_STUBS=False`
- `HITL_REVIEWER_NAME` es revisor único fijo (no se toca)
- Cobertura pytest-cov mantiene exclusiones de `.coveragerc`

## Gate de éxito por slice

Cada slice cierra cuando:
- `pytest -q` → exit 0
- `python tests/test_golden_master.py` → exit 0, sin diffs
- `pytest --cov` → cobertura ≥ baseline capturado en primer task de
  /sdd-implement
- `git grep` confirma 0 imports al path viejo
- `graph/domain.yaml` actualizado para reflejar nuevas rutas del dominio
- Commit dedicado con mensaje "refactor(slice-N): <descripción>"

## Riesgos conocidos

- `tracknet.py` puede tener dependencias circulares con
  `court_detector.py` — verificar antes de mover
- `core/interfaces.py` define ABCs usadas por múltiples dominios;
  su nuevo path va a tocar muchos imports
- Tests que importan via `pythonpath = src` pueden necesitar ajustes
  específicos
- Golden master usa pickles serializados con paths de módulos viejos;
  pueden invalidarse si Python no encuentra la clase original

## Preguntas abiertas (para /sdd-refine)

- Nombre exacto del subdirectorio para vision_tracking core
  (¿`src/vision_tracking/`, `src/vision/`, otro?)
- ¿`core/interfaces.py` queda en `src/core/` o se distribuye según dominio?
- ¿Estructura interna de `src/vision_tracking/`: archivos sueltos o
  agrupados (court/, detection/, geometry/)?
- ¿Usar `git mv` vs copy+delete para preservar history?
- ¿`__init__.py` con exports explícitos o vacíos?
- ¿Branch strategy: feature branch única con merge final, o 4 PRs
  consecutivos a main?
- ¿Tooling de migración (rope, bowler, libcst) o manual con grep?

## Métricas de éxito post-feature

- Tiempo entre "abrir archivo del proyecto" y "ubicar lógica de dominio
  X" disminuye (cualitativo, ojo)
- 0 hits de `git grep "sys.path.insert"` en código de producción
- `graph/domain.yaml` refleja el layout real sin nota de "híbrido"
- `existing-arch.md` actualizado en sección source_root
