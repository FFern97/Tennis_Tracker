"""
Registro de metadata y golpes en Supabase; exportación local a Parquet para entrenamiento.
Los fallos de red no deben tumbar el pipeline principal: se registran y se devuelven valores opcionales.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

CONFIDENCE_REVIEW_THRESHOLD = 0.45

# Raíz del proyecto (src/data/logger.py -> parents[2])
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env")


def _load_supabase_client(url: str, key: str):
    from supabase import create_client

    return create_client(url, key)


class SupabaseLogger:
    """
    Cliente opcional a Supabase + utilidad Parquet.

    Variables de entorno: ``SUPABASE_URL``, ``SUPABASE_KEY`` (cargadas desde ``.env`` en la raíz del repo).

    Tablas esperadas (Postgres/Supabase):

    - ``videos``: al menos ``id``, ``filename`` (único lógico), ``metadata`` (json/jsonb).
    - ``strokes``: al menos ``video_id``, ``confidence_score``, ``requires_review``, ``kinematics`` (json/jsonb).
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        key: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self._url = url if url is not None else os.getenv("SUPABASE_URL")
        self._key = key if key is not None else os.getenv("SUPABASE_KEY")
        self._client = client
        if self._client is None and self._url and self._key:
            try:
                self._client = _load_supabase_client(self._url, self._key)
            except Exception as e:
                logger.warning("SupabaseLogger: no se pudo crear el cliente Supabase: %s", e)
                self._client = None

    @property
    def client(self) -> Any:
        return self._client

    def get_or_create_video(
        self,
        filename: str,
        metadata: Mapping[str, Any],
    ) -> Optional[str]:
        """
        Busca ``filename`` en ``videos``. Si existe, devuelve ``id`` (str); si no, inserta y devuelve el nuevo ``id``.
        Si no hay cliente o falla la red/API, devuelve ``None``.
        """
        if not self._client:
            logger.debug("get_or_create_video: cliente Supabase no disponible.")
            return None
        try:
            sel = (
                self._client.table("videos")
                .select("id")
                .eq("filename", filename)
                .limit(1)
                .execute()
            )
            rows = getattr(sel, "data", None) or []
            if rows:
                vid = rows[0].get("id")
                return str(vid) if vid is not None else None

            ins = (
                self._client.table("videos")
                .insert({"filename": filename, "metadata": dict(metadata)})
                .execute()
            )
            inserted = getattr(ins, "data", None) or []
            if not inserted:
                logger.warning("get_or_create_video: inserción sin filas devueltas para %s.", filename)
                return None
            vid = inserted[0].get("id")
            return str(vid) if vid is not None else None
        except Exception as e:
            logger.warning("get_or_create_video falló (%s): %s", filename, e)
            return None

    def log_stroke(self, stroke_data: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """
        Inserta una fila en ``strokes``.

        ``stroke_data`` debe incluir al menos:

        - ``video_id``
        - ``confidence_score`` (float)
        - ``kinematics``: dict con salidas agregadas de cinemática (ángulos, lado, zona, velocidad, etc.).

        Si ``confidence_score < 0.45``, se fuerza ``requires_review = True`` (HITL).
        """
        if not self._client:
            logger.debug("log_stroke: cliente Supabase no disponible.")
            return None
        try:
            video_id = stroke_data["video_id"]
            confidence = float(stroke_data["confidence_score"])
            kinematics = dict(stroke_data.get("kinematics", {}))

            row = {
                "video_id": video_id,
                "confidence_score": confidence,
                "requires_review": confidence < CONFIDENCE_REVIEW_THRESHOLD,
                "kinematics": kinematics,
            }
            res = self._client.table("strokes").insert(row).execute()
            rows = getattr(res, "data", None) or []
            return rows[0] if rows else None
        except Exception as e:
            logger.warning("log_stroke falló: %s", e)
            return None

    @staticmethod
    def save_stroke_sequence(
        frame_buffer: list,
        path: str | Path,
    ) -> bool:
        """
        Persiste la secuencia de frames (lista de filas: dicts o estructuras homogéneas) en Parquet local.
        Crea directorios padre si hace falta. Devuelve ``True`` si se escribió bien.
        """
        try:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(frame_buffer)
            df.to_parquet(dest, index=False)
            return True
        except Exception as e:
            logger.warning("save_stroke_sequence falló (%s): %s", path, e)
            return False
