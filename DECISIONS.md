# DECISIONS.md — Decisiones arquitectónicas

Registro de decisiones que se apartan de patrones documentados en `existing-arch.md`
o que requieren justificación explícita por convenciones SDD.

---

## 2026-07-10 — Eliminación del alias `PersonTracker`

**Contexto**: `existing-arch.md` documenta como patrón vigente:
"`PlayerTracker` es la clase canónica; `PersonTracker` es alias de compatibilidad."

**Decisión**: Eliminar `PersonTracker` como parte del refactor Slice 3
(feature `001-refactor-consolidar-src`, T009).

**Justificación**: El alias no tiene consumidores externos (repo solo-dev sin
librería publicada). Mantenerlo agrega ruido en `__init__.py` que contradice
MUST 7 (init vacíos, sin re-exports). Todos los call sites migran a `PlayerTracker`.

**Impacto**: Callers en `src/schema.py` (docstring actualizado), `tennis_tracker.py` (legacy,
se elimina en Slice 4). Sin impacto en API externa.

## 2026-07-10 — Actualización de baseline de cobertura post Slice 3

**Contexto**: T009 eliminó 4 stmts que estaban al 100% cubiertos
(alias `PersonTracker` + re-exports del `__init__.py`), causando drop
de TOTAL 88.22% → 88.16% y `player_tracker.py` 91.43% → 91.37%.

**Decisión**: Actualizar `baseline-coverage.md` con los nuevos valores.
El gate de cobertura de Slice 4 compara contra el baseline actualizado.

**Justificación**: El drop es contracción intencional del denominador
(eliminación de código), no regresión de tests. Aplicar el gate literal
contra 88.22% penalizaría la eliminación de código muerto/redundante,
contradiciendo el objetivo de MUST 7.

## 2026-07-10 — T011: Eliminación de scripts legacy en raíz

**Contexto**: `tennis_tracker.py` y `yolo_person_detector.py` permanecían en la raíz
como histórico pre-consolidación (`existing-arch.md`). Slice 4 T011 cierra el legacy
cleanup de la feature `001-refactor-consolidar-src`.

**Decisión**: Eliminar ambos scripts con `git rm` (`tennis_tracker.py`, `yolo_person_detector.py`).

**Justificación**: 0 callers externos (git grep: imports anclados + grep plano sin blockers
en código/CI). `tennis_tracker.py` tenía imports rotos a paths pre-consolidación
(`tracknet`, `court_detector`, `visualization_utils`, `trackers`) y a `PersonTracker`
(eliminado en Slice 3). `yolo_person_detector.py` era standalone (solo deps externas),
sin uso en el pipeline consolidado. Reemplazados funcionalmente por `main.py` (entry point)
+ `src/vision_tracking/` + `src/trackers/`.

**Impacto**: Sin impacto en tests ni cobertura (ambos en `omit` de `.coveragerc`;
TOTAL permanece 88.16%). Referencias documentales en specs/ y `features.yaml` se
actualizan en T014.

## 2026-07-10 — T012/T013/T014: Golden master unpickler + __init__ cleanup

**Contexto**: La migración a full-path imports (T013) rompió la deserialización de
los stubs .pkl del golden master, que serializan clases como `schema.Detection`
(módulo bare, ya inexistente tras reducir pythonpath a `.` en T012).

**Decisión**: (a) `_StubUnpickler` en test_golden_master.py remapea `schema.*` →
`src.schema.*` en deserialización, sin regenerar los pickles. (b) Vaciado de
`src/detectors/__init__.py` (re-export muerto de YoloPoseDetector, 0 callers vía
`from src.detectors import`).

**Justificación**: (a) Regenerar los stubs habría rebaseline-ado el golden master
a la salida del código actual, destruyendo la señal de regresión; el remapeo de
módulo preserva las referencias originales intactas. (b) US-7 y MUST 7 exigen
`__init__.py` bajo `src/` vacíos sin re-exports; el re-export no tenía callers.

**Impacto**: Golden master sigue [OK] PASSED contra stubs originales. Cobertura
TOTAL sin movimiento (88.16%). 0 hits de `from src.detectors import` post-cleanup.
