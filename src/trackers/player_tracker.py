"""
Tracker de jugadores: IDs YOLO, homografía opcional e interpolación en píxeles.
"""
import cv2
import numpy as np

from src.core.interfaces import BaseTracker
from src.schema import PlayerDetection, PlayersInfo
import config


class PlayerTracker(BaseTracker):
    """
    Rastrea jugadores con tracking directo basado en track_id de YOLO.
    Interpolación simple en píxeles cuando se pierde un track.
    """

    def __init__(self):
        self.max_interpolation_frames = config.PERSON_MAX_INTERPOLATION_FRAMES
        self._track_history_max = config.PERSON_TRACK_HISTORY_MAX
        self._kp_visibility_threshold = config.PERSON_KEYPOINT_VISIBILITY_THRESHOLD
        self.tracks = {}
        self.inv_homography = None
        self.homography = None

    def set_homography(self, inv_homography):
        """Establece la matriz de homografía inversa (Frame -> Cancha)."""
        self.inv_homography = inv_homography
        if inv_homography is not None:
            try:
                self.homography = np.linalg.inv(inv_homography)
            except np.linalg.LinAlgError:
                self.homography = None

    def update(
        self,
        player_detections: list,
        inv_homography=None,
        frame_number=None,
    ):
        """
        Actualiza el tracker con detecciones de jugadores (PlayerDetection).
        """
        if inv_homography is not None:
            self.set_homography(inv_homography)

        active_track_ids = set()
        for d in player_detections:
            if not isinstance(d, PlayerDetection) or d.id is None:
                continue

            track_id = d.id
            active_track_ids.add(track_id)

            position_px = (d.x, d.y)

            position_court = None
            if self.inv_homography is not None:
                pt = np.array([[[d.x, d.y]]], dtype=np.float32)
                out = cv2.perspectiveTransform(pt, self.inv_homography)
                if out is not None and len(out) > 0:
                    position_court = (float(out[0][0][0]), float(out[0][0][1]))

            current_keypoints = d.keypoints.copy() if d.keypoints is not None else None

            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "last_frame": frame_number or 0,
                    "last_position_px": position_px,
                    "last_position_court": position_court,
                    "last_keypoints": current_keypoints,
                    "history_px": [],
                    "history_court": [],
                    "history_keypoints": [],
                    "missing_frames": 0,
                }
            else:
                self.tracks[track_id]["last_frame"] = frame_number or 0
                self.tracks[track_id]["last_position_px"] = position_px
                self.tracks[track_id]["last_position_court"] = position_court
                self.tracks[track_id]["last_keypoints"] = current_keypoints
                self.tracks[track_id]["missing_frames"] = 0

            self.tracks[track_id]["history_px"].append(position_px)
            self.tracks[track_id]["history_court"].append(position_court)
            if current_keypoints is not None:
                self.tracks[track_id]["history_keypoints"].append(
                    current_keypoints.copy()
                )

            while len(self.tracks[track_id]["history_px"]) > self._track_history_max:
                self.tracks[track_id]["history_px"].pop(0)
                self.tracks[track_id]["history_court"].pop(0)
                if self.tracks[track_id]["history_keypoints"]:
                    self.tracks[track_id]["history_keypoints"].pop(0)

        active_tracks_data = {}
        all_positions = []

        for track_id, track_data in self.tracks.items():
            if track_id in active_track_ids:
                active_tracks_data[track_id] = {
                    "px": track_data["last_position_px"],
                    "court": track_data["last_position_court"],
                    "is_interpolated": False,
                    "keypoints": track_data["last_keypoints"],
                }
                all_positions.append(
                    {
                        "track_id": track_id,
                        "px": track_data["last_position_px"],
                        "court": track_data["last_position_court"],
                        "is_interpolated": False,
                    }
                )
            else:
                track_data["missing_frames"] += 1

                if track_data["missing_frames"] <= self.max_interpolation_frames:
                    if len(track_data["history_px"]) >= 2:
                        last_pos = track_data["history_px"][-1]
                        prev_pos = track_data["history_px"][-2]
                        vel_x = last_pos[0] - prev_pos[0]
                        vel_y = last_pos[1] - prev_pos[1]

                        interpolated_px = (
                            last_pos[0] + vel_x * track_data["missing_frames"],
                            last_pos[1] + vel_y * track_data["missing_frames"],
                        )

                        interpolated_court = None
                        if self.inv_homography is not None and interpolated_px is not None:
                            pt_px = np.array(
                                [[[interpolated_px[0], interpolated_px[1]]]], dtype=np.float32
                            )
                            pt_court = cv2.perspectiveTransform(pt_px, self.inv_homography)
                            if pt_court is not None and len(pt_court) > 0:
                                interpolated_court = (
                                    float(pt_court[0][0][0]),
                                    float(pt_court[0][0][1]),
                                )

                        estimated_keypoints = None
                        if (
                            track_data["last_keypoints"] is not None
                            and interpolated_px is not None
                        ):
                            last_px = track_data["last_position_px"]
                            offset_x = interpolated_px[0] - last_px[0]
                            offset_y = interpolated_px[1] - last_px[1]

                            estimated_keypoints = track_data["last_keypoints"].copy()
                            thr = self._kp_visibility_threshold
                            for kp_idx in range(len(estimated_keypoints)):
                                if estimated_keypoints[kp_idx][2] > thr:
                                    estimated_keypoints[kp_idx][0] += offset_x
                                    estimated_keypoints[kp_idx][1] += offset_y

                        active_tracks_data[track_id] = {
                            "px": interpolated_px,
                            "court": interpolated_court,
                            "is_interpolated": True,
                            "keypoints": estimated_keypoints,
                        }
                        all_positions.append(
                            {
                                "track_id": track_id,
                                "px": interpolated_px,
                                "court": interpolated_court,
                                "is_interpolated": True,
                            }
                        )

        tracks_to_remove = [
            tid
            for tid, data in self.tracks.items()
            if data["missing_frames"] > self.max_interpolation_frames
        ]
        for tid in tracks_to_remove:
            del self.tracks[tid]

        return PlayersInfo(
            active_tracks=active_tracks_data,
            all_positions=all_positions,
        )

    def get_track_history(self, track_id):
        if track_id in self.tracks:
            t = self.tracks[track_id]
            return (
                t["history_px"].copy(),
                t["history_court"].copy(),
                t["history_keypoints"].copy(),
            )
        return (None, None, None)

    def get_all_tracks(self):
        return self.tracks.copy()

