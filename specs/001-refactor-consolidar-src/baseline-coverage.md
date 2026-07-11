# baseline-coverage.md — Snapshot pre-refactor

> Generado por T000 (`/sdd-implement`) el 2026-07-09
> Commit base: `e66adc9fcc9cc2984b65bed9421fa451a4773197`
> Feature: `001-refactor-consolidar-src`
> **Baseline vigente post Slice 3** — ver sección «Actualización — 2026-07-10».

## Comando reproducible

Desde la raíz del repo (usa `addopts` de `pytest.ini` + `.coveragerc`):

```bash
pytest
```

Equivalente explícito (flags inyectados por `pytest.ini` post-slice-3):

```bash
pytest --cov=src/core --cov=src/trackers --cov=src/vision_tracking \
  --cov=analytics --cov=src/data --cov=detectors --cov=pipeline \
  --cov-config=.coveragerc --cov-report=term-missing --cov-report=html -q
```

## Resultado del snapshot (vigente post Slice 3)

| Métrica | Valor |
|---------|-------|
| Tests | 66 passed |
| **TOTAL cobertura** | **88.13%** (648 stmts, 56 miss, 220 branches, 39 BrPart) |
| Plataforma | win32, Python 3.12.4 |
| Duración | ~9s (post-slice-3) |

## Tabla por módulo — 9 archivos afectados (slices 1–3)

Paths **post-slice-3** (layout unificado bajo `src/`).

| # | Módulo (path post-slice-3) | Stmts | Miss | Cover | Gate aplicable |
|---|----------------------------|------:|-----:|------:|----------------|
| 1 | `src/visualization/visualization.py` | — | — | — | solo golden master |
| 2 | `src/visualization/visualization_utils.py` | — | — | — | solo golden master |
| 3 | `src/trackers/ball_tracker.py` | 67 | 4 | **90.11%** | cov + golden master |
| 4 | `src/trackers/player_tracker.py` | 97 | 4 | **91.37%** | cov + golden master |
| 5 | `src/vision_tracking/inference.py` | 73 | 2 | **94.74%** | cov + golden master |
| 6 | `src/vision_tracking/court_detector.py` | — | — | — | solo golden master |
| 7 | `src/vision_tracking/geometry_utils.py` | 51 | 0 | **98.59%** | cov + golden master |
| 8 | `src/vision_tracking/tracknet.py` | — | — | — | solo golden master |
| 9 | `src/core/interfaces.py` | 12 | 2 | **83.33%** | cov + golden master |

## ⚠️ Módulos omitidos de coverage (política heredada de .coveragerc)

Estos 4 archivos afectados por el refactor NO son medidos por pytest-cov:

- `src/visualization/visualization.py` (slice 1)
- `src/visualization/visualization_utils.py` (slice 1)
- `src/vision_tracking/court_detector.py` (slice 3)
- `src/vision_tracking/tracknet.py` (slice 3)

Consecuencia para el gate:

- Coverage % **no aplica** para estos módulos en su slice.
- Golden master (`tests/test_golden_master.py`) es el **único gate de regresión de comportamiento** para ellos. Cualquier cambio silencioso en su lógica se detecta como diff en el video/pkl generado.
- Al ajustar paths en `.coveragerc` post-move, mantener la política omit (no promover a medido).

Cambiar esta política requiere entrada en `DECISIONS.md` (fuera de scope de esta feature).

### Notas

- **5 módulos medidos** (filas 3–5, 7, 9): cobertura por módulo **no debe bajar** vs estos valores tras cada slice restante.
- **TOTAL global** (88.13%): no debe bajar tras ningún slice restante.

## Gate de comparación (por slice)

1. `pytest -q` → exit 0
2. `python tests/test_golden_master.py` → exit 0, sin diffs
3. TOTAL ≥ **88.13%**
4. Por módulo medido movido en la slice: `Cover ≥` valor de la tabla
5. Por módulo omitido movido en la slice: golden master sin diffs es el único gate aplicable (ver sección ⚠️ arriba)

## Actualización — 2026-07-10 (post Slice 3, T009)

Baseline actualizado tras eliminación intencional de código en T009:

- `PersonTracker = PlayerTracker` (1 stmt en `player_tracker.py`)
- Re-exports en `trackers/__init__.py` (3 stmts: 2 imports + 1 `__all__`)

Los 4 stmts estaban al 100% cubiertos. Drop es contracción del denominador,
no regresión de tests. Gate slice 4 compara contra este baseline actualizado.

- TOTAL nuevo: **88.16%** (650 stmts, 56 miss, 220 branches, 39 BrPart)
- `player_tracker.py` nuevo: **91.37%** (97 stmts, 4 miss)

Valores pre-slice-3 (T000 original): TOTAL **88.22%** (654 stmts); `player_tracker.py` **91.43%** (98 stmts). Resto de módulos medidos sin cambio en %.
