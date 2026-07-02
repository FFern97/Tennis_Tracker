# spec.md — 001-refactor-consolidar-src

## US-1 — Ubicación predecible por dominio
**Como** desarrollador del repo  
**Quiero** que la lógica de cada dominio viva en `src/<dominio>/`  
**Para** ubicar código sin consultar el layout híbrido

**Given** slice 4 completado  
**When** busco módulos de visualization, trackers o vision_tracking  
**Then** solo existen bajo `src/visualization/`, `src/trackers/`, `src/vision_tracking/` y ABCs en `src/core/interfaces.py`

## US-2 — Imports full-path
**Given** slices 1–3 completados  
**When** un módulo en `src/` importa otro de `src/`  
**Then** usa `from src.<paquete>.<módulo> import …` (verificable en diff)

## US-3 — Pipeline sin regresión
**Given** cualquier slice cerrado  
**When** ejecuto `pytest -q` y `python tests/test_golden_master.py`  
**Then** ambos exit 0; golden master sin diffs

## US-4 — Cobertura no regresiva
**Given** `baseline-coverage.md` generado (T000)  
**When** ejecuto `pytest --cov` tras cada slice  
**Then** total ≥ baseline y módulos movidos en esa slice no bajan vs baseline

## US-5 — Legacy eliminado con trazabilidad
**Given** slice 4  
**When** `git grep` confirma 0 imports a scripts legacy  
**Then** `tennis_tracker.py` y `yolo_person_detector.py` eliminados y decisión en `DECISIONS.md`

## US-6 — Documentación alineada
**Given** slice 4 completado  
**When** leo `existing-arch.md` y `graph/domain.yaml`  
**Then** `source_root` describe layout unificado bajo `src/` sin “híbrido” ni `sys.path.insert`

## Measurable Process Outcomes (DX)
- **DX-001**: Implementación completa con **< 8** ciclos de autocorrección (rework) en total.
- **DX-002**: Densidad de ambigüedad **0** — sin consultas de aclaración durante `/sdd-implement`.

## Fuera de scope (v1)
- Cambiar firmas públicas — contrato estable del pipeline (`input.md` §4).
- Dividir funciones o convertir libres↔métodos — solo move + imports.
- Refactor de tests más allá de actualizar imports.
- Migrar a pyproject/Poetry/uv; pin Python; split `requirements.txt`.
- Cambiar comportamiento de pipeline, Streamlit o persistencia.
- Tocar constantes de tuning en `config.py`.
- Reordenar lógica de orquestación en `main.py` / `app.py`.
- Mover `src/analytics/`, `src/pipeline/`, `src/data/`, `src/detectors/`, `src/schema.py`.
- Re-exports en `__init__.py`.
- Tooling AST (rope, bowler, libcst) — manual con grep acordado.
- Branch/PRs — commits directos en `main`.
- Tests manuales Streamlit como gate.
