from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from slam.feature_detection import FeatureDetectionFrame
from slam.stereo_matching import StereoMatchingFrame

# Pyramidal Lucas-Kanade: 21x21 window / 3 levels, the standard VIO choice (e.g. VINS-Mono):
# survives motion blur and a fast frame's displacement without a bigger window hurting
# localization.
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
# Forward-backward round-trip tolerance [px]: track prev->cur->prev and drop points that don't
# land back near the start. KLT can drift onto a similar patch without setting its failure flag.
FB_THRESHOLD_PX = 1.0
# Seed new tracks from this frame's ORB keypoints whenever the alive count drops below this,
# keeping density roughly constant.
TARGET_TRACK_COUNT = 300
# Don't seed a new track this close to an already-alive one -- new points should fill gaps left by
# lost tracks, not cluster redundantly on top of survivors.
MIN_TRACK_SEPARATION_PX = 8.0
# Bounds [m] on a seed's triangulated depth (matches slam.py's DEPTH_MIN/DEPTH_MAX). A few stereo
# matches survive the ratio test and epipolar gate but triangulate to a negative or absurd depth;
# reject them before they become a track.
SEED_DEPTH_MIN = 0.3
SEED_DEPTH_MAX = 40.0


@dataclass
class OpticalFlowFrame:
    timestamp_ns: int
    track_uv: dict[int, tuple[float, float]] = field(default_factory=dict)  # track id -> cam0 (u, v)
    # track id -> stereo-triangulated 3D point in the cam0 frame, for tracks seeded at this
    # frame only. It is a one-time depth that seeds a landmark's initial guess; the graph
    # refines the position from later reprojection observations.
    seeded_point_cam0: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class OpticalFlowResult:
    frames: list[OpticalFlowFrame]
    elapsed_s: float


class OpticalFlowTracker:
    """Owns one continuous KLT tracking state, advanced one frame at a time via add_frame.

    Tracks cam0 points frame to frame with pyramidal KLT instead of re-matching ORB descriptors
    between keyframes, which broke on V1_03_difficult's fast rotations. KLT only has to survive one
    frame's motion (~0.05 s) rather than a keyframe gap's (~0.15-0.5 s), and a track's stable id
    gives temporal identity for free.

    Seeds come from the frame's already-detected ORB keypoints, and only if they have a valid
    epipolar-consistent stereo match (from the StereoMatchingFrame computed before this tracker sees
    the frame), so every track has a metric depth at birth.
    """

    def __init__(self) -> None:
        self._prev_img: Optional[np.ndarray] = None
        self._prev_pts = np.zeros((0, 1, 2), dtype=np.float32)
        self._prev_ids = np.zeros((0,), dtype=np.int64)
        self._next_track_id = 0

    def add_frame(
        self, fd: FeatureDetectionFrame, sm_frame: StereoMatchingFrame, cam0_img: np.ndarray,
    ) -> OpticalFlowFrame:
        """Propagate alive tracks via forward/backward KLT against `cam0_img`, drop ones that fail
        validation, then replenish from this frame's ORB keypoints / stereo matches up to
        TARGET_TRACK_COUNT. Reads only this frame and the tracker's carried-forward state.
        """
        if self._prev_img is not None and len(self._prev_pts):
            # cv2's stub requires `nextPts` as a non-Optional MatLike, but passing None (let
            # OpenCV allocate the output) is the standard idiom.
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_img, cam0_img, self._prev_pts, None,  # type: ignore[call-overload]
                winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA)
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                cam0_img, self._prev_img, next_pts, None,  # type: ignore[call-overload]
                winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA)
            status = status.reshape(-1).astype(bool)
            back_status = back_status.reshape(-1).astype(bool)
            fb_err = np.linalg.norm((self._prev_pts - back_pts).reshape(-1, 2), axis=1)
            h, w = cam0_img.shape
            pts_flat = next_pts.reshape(-1, 2)
            in_bounds = (pts_flat[:, 0] >= 0) & (pts_flat[:, 0] < w) & (pts_flat[:, 1] >= 0) & (pts_flat[:, 1] < h)
            keep = status & back_status & (fb_err < FB_THRESHOLD_PX) & in_bounds
            cur_pts = next_pts[keep]
            cur_ids = self._prev_ids[keep]
        else:
            cur_pts = np.zeros((0, 1, 2), dtype=np.float32)
            cur_ids = np.zeros((0,), dtype=np.int64)

        seeded_point_cam0: dict[int, np.ndarray] = {}
        need = TARGET_TRACK_COUNT - len(cur_pts)
        if need > 0 and sm_frame.matches:
            depth_ok = (sm_frame.points_3d[2] > SEED_DEPTH_MIN) & (sm_frame.points_3d[2] < SEED_DEPTH_MAX)
            candidates = [
                (m, sm_frame.points_3d[:, col])
                for col, m in enumerate(sm_frame.matches) if depth_ok[col]
            ]

            if candidates and len(cur_pts):
                alive = cur_pts.reshape(-1, 2)
                candidate_pts = np.array(
                    [fd.cam0_keypoints[m.queryIdx].pt for m, _ in candidates], dtype=np.float32)
                d = np.linalg.norm(candidate_pts[:, None, :] - alive[None, :, :], axis=2)
                far_enough = d.min(axis=1) >= MIN_TRACK_SEPARATION_PX
                candidates = [c for c, keep_k in zip(candidates, far_enough) if keep_k]

            new_pts_list = []
            new_ids_list = []
            for m, point_3d in candidates[:need]:
                tid = self._next_track_id
                self._next_track_id += 1
                new_pts_list.append(fd.cam0_keypoints[m.queryIdx].pt)
                new_ids_list.append(tid)
                seeded_point_cam0[tid] = point_3d

            if new_pts_list:
                new_pts = np.array(new_pts_list, dtype=np.float32).reshape(-1, 1, 2)
                new_ids = np.array(new_ids_list, dtype=np.int64)
                cur_pts = np.vstack([cur_pts, new_pts]) if len(cur_pts) else new_pts
                cur_ids = np.concatenate([cur_ids, new_ids]) if len(cur_ids) else new_ids

        track_uv = {int(tid): (float(pt[0, 0]), float(pt[0, 1])) for tid, pt in zip(cur_ids, cur_pts)}
        self._prev_img, self._prev_pts, self._prev_ids = cam0_img, cur_pts, cur_ids
        return OpticalFlowFrame(
            timestamp_ns=fd.timestamp_ns, track_uv=track_uv, seeded_point_cam0=seeded_point_cam0)
