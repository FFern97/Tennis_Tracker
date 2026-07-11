"""Tests de `data.logger` con cliente Supabase mockeado."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.logger import CONFIDENCE_REVIEW_THRESHOLD, SupabaseLogger


@pytest.fixture
def mock_supabase_table():
    """Cadena table -> select/insert -> ... -> execute con .data."""
    table = MagicMock()
    return table


@pytest.fixture
def mock_client(mock_supabase_table):
    client = MagicMock()
    client.table.return_value = mock_supabase_table
    return client


def _exec_result(data):
    r = MagicMock()
    r.data = data
    return r


def test_get_or_create_video_returns_existing_id(mock_client, mock_supabase_table):
    mock_supabase_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = _exec_result(
        [{"id": "vid-1"}]
    )
    log = SupabaseLogger(client=mock_client)

    vid = log.get_or_create_video("clip.mp4", {"fps": 30})

    assert vid == "vid-1"
    mock_client.table.assert_called_with("videos")
    mock_supabase_table.select.assert_called_with("id")
    mock_supabase_table.select.return_value.eq.assert_called_once_with("filename", "clip.mp4")


def test_get_or_create_video_inserts_when_missing(mock_client, mock_supabase_table):
    sel_exec = mock_supabase_table.select.return_value.eq.return_value.limit.return_value.execute
    sel_exec.return_value = _exec_result([])

    ins_chain = MagicMock()
    ins_chain.execute.return_value = _exec_result([{"id": "new-vid"}])
    mock_supabase_table.insert.return_value = ins_chain

    log = SupabaseLogger(client=mock_client)

    vid = log.get_or_create_video(
        "nuevo.mp4",
        {"fps": 24, "frame_width": 1920, "frame_height": 1080},
    )

    assert vid == "new-vid"
    mock_supabase_table.insert.assert_called_once()
    call_kw = mock_supabase_table.insert.call_args[0][0]
    assert call_kw["filename"] == "nuevo.mp4"
    assert call_kw["metadata"] == {"fps": 24, "frame_width": 1920, "frame_height": 1080}
    assert call_kw["fps"] == pytest.approx(24.0)
    assert call_kw["width"] == 1920
    assert call_kw["height"] == 1080


def test_get_or_create_video_returns_none_on_exception(mock_client, mock_supabase_table):
    mock_supabase_table.select.return_value.eq.return_value.limit.return_value.execute.side_effect = (
        RuntimeError("network")
    )
    log = SupabaseLogger(client=mock_client)
    assert log.get_or_create_video("x.mp4", {}) is None


def test_get_or_create_video_no_client():
    log = SupabaseLogger(url="", key="")
    assert log.get_or_create_video("a.mp4", {}) is None


def test_log_stroke_sets_requires_review_below_threshold(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result(
        [{"id": "stroke-1"}]
    )
    log = SupabaseLogger(client=mock_client)

    row = log.log_stroke(
        {
            "video_id": "v1",
            "confidence_score": CONFIDENCE_REVIEW_THRESHOLD - 0.1,
            "kinematics": {"side": "forehand", "vertical_zone": "mid"},
        }
    )

    assert row == {"id": "stroke-1"}
    mock_supabase_table.insert.assert_called_once()
    payload = mock_supabase_table.insert.call_args[0][0]
    assert payload["requires_review"] is True
    assert payload["confidence_score"] == pytest.approx(CONFIDENCE_REVIEW_THRESHOLD - 0.1)
    assert payload["kinematics"]["side"] == "forehand"
    assert payload["side_detected"] == "forehand"
    assert payload["zone_detected"] == "mid"


def test_log_stroke_sets_impact_frame_from_kinematics(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result([{"id": "s-frame"}])
    log = SupabaseLogger(client=mock_client)

    log.log_stroke(
        {
            "video_id": "v1",
            "confidence_score": 0.7,
            "kinematics": {"frame_number": 731, "side": "forehand"},
        }
    )

    payload = mock_supabase_table.insert.call_args[0][0]
    assert payload["impact_frame"] == 731


def test_log_stroke_maps_zone_and_velocity(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result([{"id": "s3"}])
    log = SupabaseLogger(client=mock_client)

    log.log_stroke(
        {
            "video_id": "v1",
            "confidence_score": 0.8,
            "kinematics": {
                "side": "backhand",
                "zone": "high",
                "velocity": [12.5, -3.2],
            },
        }
    )

    payload = mock_supabase_table.insert.call_args[0][0]
    assert payload["side_detected"] == "backhand"
    assert payload["zone_detected"] == "high"
    assert payload["avg_velocity_x"] == pytest.approx(12.5)
    assert payload["avg_velocity_y"] == pytest.approx(-3.2)


def test_log_stroke_requires_review_false_above_threshold(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result([{"id": "s2"}])
    log = SupabaseLogger(client=mock_client)

    log.log_stroke(
        {
            "video_id": "v1",
            "confidence_score": 0.9,
            "kinematics": {},
        }
    )

    payload = mock_supabase_table.insert.call_args[0][0]
    assert payload["requires_review"] is False


def test_log_stroke_returns_none_on_failure(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.side_effect = OSError("timeout")
    log = SupabaseLogger(client=mock_client)
    assert log.log_stroke({"video_id": "v", "confidence_score": 1.0, "kinematics": {}}) is None


def test_save_stroke_sequence_writes_parquet(tmp_path):
    frames = [
        {"frame": 0, "ball_x": 1.0, "ball_y": 2.0},
        {"frame": 1, "ball_x": 1.5, "ball_y": 2.1},
    ]
    out = tmp_path / "seq.parquet"
    assert SupabaseLogger.save_stroke_sequence(frames, out) is True

    df = pd.read_parquet(out)
    assert len(df) == 2
    assert list(df.columns) == ["frame", "ball_x", "ball_y"]


def test_save_stroke_sequence_returns_false_on_error(tmp_path):
    bad_path = tmp_path / "nope" / "\x00bad"  # inválido en muchos SO; usar mock

    with patch("src.data.logger.pd.DataFrame", side_effect=ValueError("bad data")):
        ok = SupabaseLogger.save_stroke_sequence([], tmp_path / "a.parquet")
        assert ok is False


def test_init_lazy_client_uses_env(monkeypatch):
    fake = MagicMock()
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key")

    with patch("src.data.logger._load_supabase_client", return_value=fake) as loader:
        log = SupabaseLogger()
        assert log.client is fake
        loader.assert_called_once_with("https://x.supabase.co", "key")


def test_init_client_creation_failure(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "key")
    with patch("src.data.logger._load_supabase_client", side_effect=ConnectionError("wifi")):
        log = SupabaseLogger()
        assert log.client is None


def test_get_or_create_video_insert_no_rows_returned(mock_client, mock_supabase_table):
    mock_supabase_table.select.return_value.eq.return_value.limit.return_value.execute.return_value = _exec_result([])
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result([])
    log = SupabaseLogger(client=mock_client)
    assert log.get_or_create_video("orphan.mp4", {}) is None


def test_log_stroke_insert_returns_empty_data(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result([])
    log = SupabaseLogger(client=mock_client)
    assert log.log_stroke({"video_id": "v", "confidence_score": 1.0, "kinematics": {}}) is None


def test_requires_review_boundary(mock_client, mock_supabase_table):
    mock_supabase_table.insert.return_value.execute.return_value = _exec_result([{"ok": True}])
    log = SupabaseLogger(client=mock_client)
    log.log_stroke({"video_id": "v", "confidence_score": CONFIDENCE_REVIEW_THRESHOLD, "kinematics": {}})
    payload = mock_supabase_table.insert.call_args[0][0]
    assert payload["requires_review"] is False
