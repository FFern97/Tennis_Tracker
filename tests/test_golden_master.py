"""
Golden master: comparación semántica de listas serializadas en .pkl (detecciones por frame).

Uso:
  python tests/test_golden_master.py

Opcional — comparar dos carpetas (mismo layout que stubs/<video_key>/):
  set TENNIS_GOLDEN_REFERENCE=tests/golden/test_video1
  python tests/test_golden_master.py
  (por defecto la referencia es la misma que candidata: autocomprobación + validación estructural)
"""
from __future__ import annotations

import io
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _p in (SRC, ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schema import Detection, PlayerDetection  # noqa: E402

# Nombres de archivo (alineados con config)
BALL_NAME = "ball_detections.pkl"
PLAYER_NAME = "player_detections.pkl"
DEFAULT_VIDEO_KEY = "test_video1"
RTOL = 1e-5
ATOL = 1e-5


def load_pickle_list(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def _float_close(a: float, b: float) -> bool:
    return abs(a - b) <= ATOL + RTOL * abs(b)


def compare_detection(a: Detection, b: Detection) -> tuple[bool, str]:
    if not _float_close(a.x, b.x) or not _float_close(a.y, b.y) or not _float_close(a.conf, b.conf):
        return False, f"Detection distinta: a=({a.x},{a.y},{a.conf}) b=({b.x},{b.y},{b.conf})"
    if a.id != b.id:
        return False, f"id distinto: {a.id!r} vs {b.id!r}"
    return True, ""


def compare_player_detection(a: PlayerDetection, b: PlayerDetection) -> tuple[bool, str]:
    ok, msg = compare_detection(a, b)
    if not ok:
        return ok, msg
    if len(a.xyxy) != len(b.xyxy):
        return False, "xyxy longitud distinta"
    for ua, ub in zip(a.xyxy, b.xyxy):
        if not _float_close(float(ua), float(ub)):
            return False, f"xyxy distinto: {a.xyxy} vs {b.xyxy}"
    if (a.keypoints is None) ^ (b.keypoints is None):
        return False, "keypoints None mismatch"
    if a.keypoints is not None and b.keypoints is not None:
        if a.keypoints.shape != b.keypoints.shape:
            return False, f"keypoints shape {a.keypoints.shape} vs {b.keypoints.shape}"
        if not np.allclose(a.keypoints, b.keypoints, rtol=RTOL, atol=ATOL):
            return False, "keypoints valores distintos"
    return True, ""


def compare_ball_frame_lists(actual: list, expected: list) -> tuple[bool, str]:
    if len(actual) != len(expected):
        return False, f"cantidad de frames distinta: {len(actual)} vs {len(expected)}"
    for i, (fr_a, fr_b) in enumerate(zip(actual, expected)):
        if len(fr_a) != len(fr_b):
            return False, f"frame {i}: cantidad de detecciones {len(fr_a)} vs {len(fr_b)}"
        for j, (da, db) in enumerate(zip(fr_a, fr_b)):
            if not isinstance(da, Detection) or not isinstance(db, Detection):
                return False, f"frame {i} det {j}: tipo inválido"
            ok, msg = compare_detection(da, db)
            if not ok:
                return False, f"frame {i} det {j}: {msg}"
    return True, ""


def compare_player_frame_lists(actual: list, expected: list) -> tuple[bool, str]:
    if len(actual) != len(expected):
        return False, f"cantidad de frames jugadores distinta: {len(actual)} vs {len(expected)}"
    for i, (fr_a, fr_b) in enumerate(zip(actual, expected)):
        if len(fr_a) != len(fr_b):
            return False, f"frame {i} jugadores: {len(fr_a)} vs {len(fr_b)}"
        for j, (pa, pb) in enumerate(zip(fr_a, fr_b)):
            if not isinstance(pa, PlayerDetection) or not isinstance(pb, PlayerDetection):
                return False, f"frame {i} jug {j}: tipo inválido"
            ok, msg = compare_player_detection(pa, pb)
            if not ok:
                return False, f"frame {i} jug {j}: {msg}"
    return True, ""


def compare_pickle_ball_lists(path_a: Path, path_b: Path) -> tuple[bool, str]:
    return compare_ball_frame_lists(load_pickle_list(path_a), load_pickle_list(path_b))


def compare_pickle_player_lists(path_a: Path, path_b: Path) -> tuple[bool, str]:
    return compare_player_frame_lists(load_pickle_list(path_a), load_pickle_list(path_b))


def pickle_roundtrip_equal(obj: object) -> tuple[bool, str]:
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    buf.seek(0)
    loaded = pickle.load(buf)
    if not isinstance(obj, list) or not isinstance(loaded, list):
        return False, "se esperaba list en raíz"
    sample = None
    for fr in obj:
        if fr:
            sample = fr[0]
            break
    if sample is None:
        if len(obj) != len(loaded):
            return False, "listas vacías / longitud distinta tras round-trip"
        if not all(len(a) == len(b) for a, b in zip(obj, loaded)):
            return False, "frames vacíos distinto conteo"
        return True, ""
    if isinstance(sample, PlayerDetection):
        return compare_player_frame_lists(obj, loaded)
    if isinstance(sample, Detection):
        return compare_ball_frame_lists(obj, loaded)
    return False, "tipo de elemento no soportado para round-trip"


def validate_stub_pair(candidate_dir: Path) -> tuple[bool, str]:
    ball_p = candidate_dir / BALL_NAME
    player_p = candidate_dir / PLAYER_NAME
    if not ball_p.is_file():
        return False, f"No existe {ball_p}"
    if not player_p.is_file():
        return False, f"No existe {player_p}"
    try:
        balls = load_pickle_list(ball_p)
        players = load_pickle_list(player_p)
    except Exception as e:
        return False, f"pickle.load falló: {e}"
    if not isinstance(balls, list) or not isinstance(players, list):
        return False, "raíz del pickle debe ser list"
    if len(balls) != len(players):
        return False, f"len(ball)={len(balls)} != len(players)={len(players)}"
    for i, fr in enumerate(balls):
        if not isinstance(fr, list):
            return False, f"frame ball {i} no es list"
        for d in fr:
            if not isinstance(d, Detection):
                return False, f"frame ball {i}: elemento no Detection"
    for i, fr in enumerate(players):
        if not isinstance(fr, list):
            return False, f"frame player {i} no es list"
        for p in fr:
            if not isinstance(p, PlayerDetection):
                return False, f"frame player {i}: elemento no PlayerDetection"
    ok, msg = pickle_roundtrip_equal(balls)
    if not ok:
        return False, f"round-trip ball: {msg}"
    ok, msg = pickle_roundtrip_equal(players)
    if not ok:
        return False, f"round-trip player: {msg}"
    return True, ""


def run_golden_master() -> int:
    candidate_dir = ROOT / "stubs" / DEFAULT_VIDEO_KEY
    ref_env = os.environ.get("TENNIS_GOLDEN_REFERENCE", "").strip()
    reference_dir = Path(ref_env) if ref_env else candidate_dir
    if not reference_dir.is_absolute():
        reference_dir = (ROOT / reference_dir).resolve()

    ok, msg = validate_stub_pair(candidate_dir)
    if not ok:
        print(f"[FAIL] (validacion candidata): {msg}")
        return 1

    if reference_dir.resolve() == candidate_dir.resolve():
        ok, msg = compare_pickle_ball_lists(candidate_dir / BALL_NAME, candidate_dir / BALL_NAME)
        if not ok:
            print(f"[FAIL] (autocheck ball): {msg}")
            return 1
        ok, msg = compare_pickle_player_lists(candidate_dir / PLAYER_NAME, candidate_dir / PLAYER_NAME)
        if not ok:
            print(f"[FAIL] (autocheck player): {msg}")
            return 1
    else:
        ok, msg = validate_stub_pair(reference_dir)
        if not ok:
            print(f"[FAIL] (validacion referencia): {msg}")
            return 1
        ok, msg = compare_pickle_ball_lists(candidate_dir / BALL_NAME, reference_dir / BALL_NAME)
        if not ok:
            print(f"[FAIL] (ball vs golden): {msg}")
            return 1
        ok, msg = compare_pickle_player_lists(candidate_dir / PLAYER_NAME, reference_dir / PLAYER_NAME)
        if not ok:
            print(f"[FAIL] (player vs golden): {msg}")
            return 1

    # ASCII para consolas Windows (cp1252); significado equivalente a check verde
    print("[OK] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_golden_master())
