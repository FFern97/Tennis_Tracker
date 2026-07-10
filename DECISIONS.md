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
