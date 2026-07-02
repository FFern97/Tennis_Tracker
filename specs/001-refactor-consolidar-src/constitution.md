# constitution.md — 001-refactor-consolidar-src

> Complementa `existing-arch.md`. No contradice sus patrones salvo decisión en `DECISIONS.md`.

## MUST

1. **Comportamiento idéntico** — Tras cada slice: `pytest -q` y golden master exit 0 sin diffs.
2. **Layout por dominio** — Código de dominio bajo `src/<dominio>/`; entry points solo en raíz.
3. **Imports full-path (estado final)** — Al cierre de la feature (post-slice-4): todo import entre módulos de `src/` usa `from src.<paquete>.<módulo> import …`. Estado transitorio (slices 1–3): módulos movidos usan full-path inmediatamente; módulos no movidos mantienen imports bare (permitidos por `pythonpath = . src`) hasta T013.
4. **Moves atómicos** — `git mv` + imports actualizados en el mismo commit; un commit por slice en `main`.
5. **Gates por slice** — pytest, golden master, cobertura ≥ baseline, `git grep` sin paths viejos, `graph/domain.yaml` actualizado.
6. **DIP intacto** — Orquestador depende de `BaseDetector` / `BaseTracker` (`src/core/interfaces.py` post slice 3).
7. **`__init__.py` vacíos** — Sin re-exports; callers importan del módulo específico.
8. **`config.py` intacto** — Constantes MAYÚSCULAS sin cambio de valor; solo entry points reciben imports mecánicos.
9. **Golden master** — Stubs en `stubs/<video_key>/`; `OVERWRITE_STUBS=False`.
10. **Cierre documental** — Slice 4: `existing-arch.md`, `graph/domain.yaml`, `DECISIONS.md`.
11. **Alias `PersonTracker`** — Se elimina en slice 3. Requiere entrada en `DECISIONS.md` porque `existing-arch.md` lo documenta como patrón vigente.

## PROHIBITED

1. Cambiar firmas públicas, reordenar orquestación o tocar `HITL_REVIEWER_NAME`.
2. Mover `src/analytics/`, `src/pipeline/`, `src/data/`, `src/detectors/`, `src/schema.py`.
3. Re-exports en `__init__.py`, tooling AST automático, branch/PRs.
4. Quitar `sys.path.insert` antes del slice 4.
5. Eliminar legacy sin `git grep` previo ni entrada en `DECISIONS.md`.
6. Tests Streamlit manuales como gate de aceptación.
7. Modificar `tennis_tracker.py` / `yolo_person_detector.py` salvo eliminación en slice 4.
