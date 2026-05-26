"""Fusión track/pose, candidatos de impacto y filas Parquet para el dataset."""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, List, Optional

import numpy as np

from analytics.kinematics import (
    classify_side,
    classify_vertical_zone,
    detect_impact_candidate,
    get_distance,
)
from schema import FrameData, PlayerDetection

# Índices COCO (17 keypoints)
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10


def iou_xyxy(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def merge_pose_keypoints(
    tracked_players: List[PlayerDetection],
    pose_players: List[PlayerDetection],
    iou_min: float,
) -> List[PlayerDetection]:
    """
    Conserva IDs del modelo con tracking; refina keypoints con la salida de pose (IoU).
    """
    if not pose_players:
        return list(tracked_players)
    merged: List[PlayerDetection] = []
    for tp in tracked_players:
        t_xyxy = tuple(float(x) for x in tp.xyxy)
        best_iou = 0.0
        best_kp: Optional[np.ndarray] = None
        for pp in pose_players:
            p_xyxy = tuple(float(x) for x in pp.xyxy)
            ov = iou_xyxy(t_xyxy, p_xyxy)
            if ov > best_iou:
                best_iou = ov
                if pp.keypoints is not None:
                    best_kp = pp.keypoints.copy()
                else:
                    best_kp = None
        if best_iou >= iou_min and best_kp is not None:
            merged.append(replace(tp, keypoints=best_kp))
        else:
            merged.append(tp)
    return merged


def snapshot_framedata(fd: FrameData) -> FrameData:
    balls = [replace(b) for b in fd.ball]
    players: List[PlayerDetection] = []
    for p in fd.players:
        kp = p.keypoints.copy() if p.keypoints is not None else None
        players.append(replace(p, keypoints=kp))
    return FrameData(ball=balls, players=players)


def framedata_row(frame_number: int, fd: FrameData) -> dict[str, Any]:
    balls = [{"x": b.x, "y": b.y, "conf": b.conf, "id": b.id} for b in fd.ball]
    pl: List[dict[str, Any]] = []
    for p in fd.players:
        pl.append(
            {
                "x": p.x,
                "y": p.y,
                "conf": p.conf,
                "id": p.id,
                "xyxy": [float(v) for v in p.xyxy],
                "keypoints": p.keypoints.tolist() if p.keypoints is not None else None,
            }
        )
    return {
        "frame_number": frame_number,
        "balls_json": json.dumps(balls),
        "players_json": json.dumps(pl),
    }


def _point_from_kp(kp: np.ndarray, idx: int, conf_min: float) -> Optional[tuple[float, float]]:
    if kp.shape[0] <= idx:
        return None
    x, y, c = float(kp[idx][0]), float(kp[idx][1]), float(kp[idx][2])
    if c < conf_min:
        return None
    return (x, y)


def shoulder_midpoint(keypoints: np.ndarray, conf_min: float) -> Optional[tuple[float, float]]:
    ls = _point_from_kp(keypoints, KP_LEFT_SHOULDER, conf_min)
    rs = _point_from_kp(keypoints, KP_RIGHT_SHOULDER, conf_min)
    if ls is not None and rs is not None:
        return ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    return ls or rs


def head_reference(keypoints: np.ndarray, conf_min: float) -> Optional[tuple[float, float]]:
    nose = _point_from_kp(keypoints, KP_NOSE, conf_min)
    if nose is not None:
        return nose
    le = _point_from_kp(keypoints, KP_LEFT_EYE, conf_min)
    re = _point_from_kp(keypoints, KP_RIGHT_EYE, conf_min)
    if le is not None and re is not None:
        return ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    return le or re


def _bbox_center(xyxy: tuple) -> tuple[float, float]:
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pick_best_wrist(
    keypoints: np.ndarray,
    ball_xy: tuple[float, float],
    wrist_conf_min: float,
) -> Optional[dict[str, Any]]:
    best: Optional[tuple[float, tuple[float, float], float, str]] = None
    for idx, side in ((KP_LEFT_WRIST, "left"), (KP_RIGHT_WRIST, "right")):
        pt = _point_from_kp(keypoints, idx, wrist_conf_min)
        if pt is None:
            continue
        conf = float(keypoints[idx][2])
        dist = get_distance(ball_xy, pt)
        cand = (dist, pt, conf, side)
        if best is None or dist < best[0]:
            best = cand
    if best is None:
        return None
    dist, pt, conf, side = best
    return {"distance_px": dist, "xy": pt, "conf": conf, "side": side}


def try_detect_stroke(
    player: PlayerDetection,
    ball_xy: tuple[float, float],
    *,
    threshold_px: float,
    wrist_conf_min: float,
    ball_conf: float,
) -> Optional[dict[str, Any]]:
    """
    Si la pelota está lo bastante cerca de una muñeca visible, devuelve metadata de golpe
    (lado, zona vertical, confianza agregada, etc.).
    """
    kp = player.keypoints
    if kp is None or kp.shape[0] < 17:
        return None

    wrist = _pick_best_wrist(kp, ball_xy, wrist_conf_min)
    if wrist is None:
        return None
    if not detect_impact_candidate(ball_xy, wrist["xy"], threshold_px):
        return None

    torso = shoulder_midpoint(kp, wrist_conf_min)
    if torso is None:
        torso = _bbox_center(player.xyxy)

    shoulder_pt = shoulder_midpoint(kp, wrist_conf_min)
    if shoulder_pt is None:
        shoulder_pt = torso

    head_pt = head_reference(kp, wrist_conf_min)
    if head_pt is None:
        head_pt = shoulder_pt

    side = classify_side(ball_xy, torso)
    vertical_zone = classify_vertical_zone(ball_xy, shoulder_pt, head_pt)

    confidence_score = float(min(wrist["conf"], ball_conf))

    return {
        "side": side,
        "vertical_zone": vertical_zone,
        "wrist_side": wrist["side"],
        "distance_px": float(wrist["distance_px"]),
        "wrist_confidence": float(wrist["conf"]),
        "ball_confidence": float(ball_conf),
        "confidence_score": confidence_score,
        "track_id": player.id,
    }
