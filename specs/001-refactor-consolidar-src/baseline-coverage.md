# baseline-coverage.md — Snapshot pre-refactor

> Generado por T000 (`/sdd-implement`) el 2026-07-09
> Commit base: `e66adc9fcc9cc2984b65bed9421fa451a4773197`
> Feature: `001-refactor-consolidar-src`
> **No modificar** salvo re-baseline explícito antes de slice 1.

## Comando reproducible

Desde la raíz del repo (usa `addopts` de `pytest.ini` + `.coveragerc`):

```bash
pytest
```

Equivalente explícito (flags inyectados por `pytest.ini`):

```bash
pytest --cov=core --cov=trackers --cov=inference --cov=geometry_utils \
  --cov=analytics --cov=src/data --cov=detectors --cov=pipeline \
  --cov-config=.coveragerc --cov-report=term-missing --cov-report=html -q
```

## Resultado del snapshot

| Métrica | Valor |
|---------|-------|
| Tests | 66 passed |
| **TOTAL cobertura** | **88.22%** (654 stmts, 56 miss, 220 branches, 39 BrPart) |
| Plataforma | win32, Python 3.12.4 |
| Duración | ~20.37s |

## Tabla por módulo — 9 archivos afectados (slices 1–3)

Paths **pre-refactor** (layout híbrido raíz + `src/`).

| # | Módulo (path pre-refactor) | Stmts | Miss | Cover | Gate aplicable |
|---|----------------------------|------:|-----:|------:|----------------|
| 1 | `visualization.py` | — | — | — | solo golden master |
| 2 | `visualization_utils.py` | — | — | — | solo golden master |
| 3 | `trackers/ball_tracker.py` | 67 | 4 | **90.11%** | cov + golden master |
| 4 | `trackers/player_tracker.py` | 98 | 4 | **91.43%** | cov + golden master |
| 5 | `inference.py` | 73 | 2 | **94.74%** | cov + golden master |
| 6 | `court_detector.py` | — | — | — | solo golden master |
| 7 | `geometry_utils.py` | 51 | 0 | **98.59%** | cov + golden master |
| 8 | `tracknet.py` | — | — | — | solo golden master |
| 9 | `core/interfaces.py` | 12 | 2 | **83.33%** | cov + golden master |

## ⚠️ Módulos omitidos de coverage (política heredada de .coveragerc)

Estos 4 archivos afectados por el refactor NO son medidos por pytest-cov:

- `visualization.py` (slice 1)
- `visualization_utils.py` (slice 1)
- `court_detector.py` (slice 3)
- `tracknet.py` (slice 3)

Consecuencia para el gate:

- Coverage % **no aplica** para estos módulos en su slice.
- Golden master (`tests/test_golden_master.py`) es el **único gate de regresión de comportamiento** para ellos. Cualquier cambio silencioso en su lógica se detecta como diff en el video/pkl generado.
- Al ajustar paths en `.coveragerc` post-move, mantener la política omit (no promover a medido).

Cambiar esta política requiere entrada en `DECISIONS.md` (fuera de scope de esta feature).

### Notas

- **5 módulos medidos** (filas 3–5, 7, 9): cobertura por módulo **no debe bajar** vs estos valores tras cada slice.
- **TOTAL global** (88.22%): no debe bajar tras ningún slice.

## Gate de comparación (por slice)

1. `pytest -q` → exit 0
2. `python tests/test_golden_master.py` → exit 0, sin diffs
3. TOTAL ≥ **88.22%**
4. Por módulo medido movido en la slice: `Cover ≥` valor de la tabla
5. Por módulo omitido movido en la slice: golden master sin diffs es el único gate aplicable (ver sección ⚠️ arriba)
