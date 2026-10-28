from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from slam.data import EuRoCMAVData
from slam.stereo_matching import StereoGeometry

# Pyramidal Lucas-Kanade, 3 levels: survives a fast frame's displacement, while a small window keeps
# localization sharp and the cost low (VINS-Mono uses 21x21; 15x15 was as accurate here and faster).
LK_WIN_SIZE = (15, 15)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
# Forward-backward round-trip tolerance [px]: track A->B->A and drop points that don't land back
# near the start. KLT can drift onto a similar patch without setting its failure flag.
FB_THRESHOLD_PX = 1.0
# The same checks for the cam0 -> cam1 match, looser than frame-to-frame tracking: the two cameras
# differ in blur, exposure and viewpoint, so the match is noisier, and the epipolar (Sampson) gate
# below catches wrong matches anyway.
STEREO_FB_THRESHOLD_PX = 2.0
STEREO_SAMPSON_PX = 2.5
# Detect new corners whenever the alive count drops below this, keeping density roughly constant.
TARGET_TRACK_COUNT = 150
# Skip the corner detector until at least this many tracks are missing, so it isn't run for a
# handful of points every frame.
MIN_REPLENISH = 30
# Shi-Tomasi corner quality relative to the strongest corner in the image.
CORNER_QUALITY = 0.01
# Don't start a track this close to an already-alive one -- new points should fill gaps left by
# lost tracks, not cluster redundantly on top of survivors.
MIN_TRACK_SEPARATION_PX = 8.0
# Bounds [m] on a new track's triangulated depth (matches slam.py's DEPTH_MIN/DEPTH_MAX). A few
# stereo matches pass the epipolar gate but triangulate to a negative or absurd depth; reject them
# before they become a track.
SEED_DEPTH_MIN = 0.3
SEED_DEPTH_MAX = 40.0
# A stereo point is only usable in front of the camera.
STEREO_MIN_DEPTH = 0.1


@dataclass
class OpticalFlowFrame:
    timestamp_ns: int
    track_uv: dict[int, tuple[float, float]] = field(default_factory=dict)  # track id -> cam0 (u, v)
    # Tracks that also have a valid stereo measurement at this frame (KLT into cam1, passing the
    # forward-backward check and the epipolar gate): track id -> cam1 (u, v), and track id -> the
    # triangulated 3D point in the cam0 frame. Always the same keys.
    stereo_uv1: dict[int, tuple[float, float]] = field(default_factory=dict)
    stereo_point_cam0: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class OpticalFlowResult:
    frames: list[OpticalFlowFrame]
    elapsed_s: float


class OpticalFlowTracker:
    """Owns one continuous tracking state, advanced one frame at a time via add_frame.

    cam0 corners (Shi-Tomasi) are followed frame to frame with pyramidal KLT, so a track only has
    to survive one frame's motion (~0.05 s) and carries a stable id; no descriptors are involved.
    Each alive track is also KLT-tracked into cam1 and triangulated, so it has a metric depth. A
    new corner only becomes a track if it has a valid stereo measurement at birth.
    """

    def __init__(self, data: EuRoCMAVData) -> None:
        self._geom = StereoGeometry(data)
        self._prev_img: Optional[np.ndarray] = None
        self._prev_pts = np.zeros((0, 2), dtype=np.float32)
        # cam1 position at the previous frame, NaN where the track had no valid stereo there
        self._prev_pts1 = np.zeros((0, 2), dtype=np.float32)
        self._prev_ids = np.zeros((0,), dtype=np.int64)
        self._next_track_id = 0

    @staticmethod
    def _lk(img_a: np.ndarray, img_b: np.ndarray, pts: np.ndarray, guess: Optional[np.ndarray] = None):
        flags = 0
        if guess is not None:
            flags = cv2.OPTFLOW_USE_INITIAL_FLOW
            guess = guess.reshape(-1, 1, 2).astype(np.float32).copy()
        out, status, _ = cv2.calcOpticalFlowPyrLK(
            img_a, img_b, pts.reshape(-1, 1, 2), guess,  # type: ignore[call-overload]
            winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA, flags=flags)
        return out.reshape(-1, 2), status.reshape(-1).astype(bool)

    def _track_cam0(self, cam0_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Propagate alive tracks into this frame with forward/backward KLT. Returns the surviving
        (cam0 points, previous cam1 points shifted by the cam0 motion as a stereo guess, ids).
        """
        if self._prev_img is None or not len(self._prev_pts):
            return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros((0,), np.int64))
        next_pts, status = self._lk(self._prev_img, cam0_img, self._prev_pts)
        back_pts, back_status = self._lk(cam0_img, self._prev_img, next_pts, guess=self._prev_pts)
        fb_err = np.linalg.norm(self._prev_pts - back_pts, axis=1)
        h, w = cam0_img.shape
        in_bounds = (next_pts[:, 0] >= 0) & (next_pts[:, 0] < w) & (next_pts[:, 1] >= 0) & (next_pts[:, 1] < h)
        keep = status & back_status & (fb_err < FB_THRESHOLD_PX) & in_bounds
        guess1 = self._prev_pts1[keep] + (next_pts[keep] - self._prev_pts[keep])
        return next_pts[keep], guess1, self._prev_ids[keep]

    def _detect_corners(self, cam0_img: np.ndarray, alive: np.ndarray, need: int) -> np.ndarray:
        mask = np.full(cam0_img.shape, 255, dtype=np.uint8)
        for u, v in alive:
            cv2.circle(mask, (int(round(u)), int(round(v))), int(MIN_TRACK_SEPARATION_PX), (0,), -1)
        corners = cv2.goodFeaturesToTrack(
            cam0_img, maxCorners=need, qualityLevel=CORNER_QUALITY,
            minDistance=MIN_TRACK_SEPARATION_PX, mask=mask)
        if corners is None:
            return np.zeros((0, 2), np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def _stereo(
        self, cam0_img: np.ndarray, cam1_img: np.ndarray, pts0: np.ndarray, guess1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """KLT each cam0 point into cam1 and validate. Returns (valid mask, cam1 points, 3D points in
        cam0 frame, shape (3, N)); the last two are only meaningful where the mask is set.
        """
        pts1, status = self._lk(cam0_img, cam1_img, pts0, guess=guess1)
        back_pts, back_status = self._lk(cam1_img, cam0_img, pts1, guess=pts0)
        valid = status & back_status & (np.linalg.norm(pts0 - back_pts, axis=1) < STEREO_FB_THRESHOLD_PX)
        h, w = cam1_img.shape
        valid &= (pts1[:, 0] >= 0) & (pts1[:, 0] < w) & (pts1[:, 1] >= 0) & (pts1[:, 1] < h)
        points_3d = np.zeros((3, len(pts0)))
        idx = np.flatnonzero(valid)
        if len(idx):
            u0 = self._geom.undistort0(pts0[idx])
            u1 = self._geom.undistort1(pts1[idx])
            ok = self._geom.sampson_sq(u0, u1) < STEREO_SAMPSON_PX ** 2
            idx, u0, u1 = idx[ok], u0[ok], u1[ok]
            valid[:] = False
            if len(idx):
                p3 = self._geom.triangulate(u0, u1)
                in_front = p3[2] > STEREO_MIN_DEPTH
                valid[idx[in_front]] = True
                points_3d[:, idx[in_front]] = p3[:, in_front]
        return valid, pts1, points_3d

    def add_frame(self, ts: int, cam0_img: np.ndarray, cam1_img: np.ndarray) -> OpticalFlowFrame:
        cur_pts, guess1, cur_ids = self._track_cam0(cam0_img)

        num_old = len(cur_pts)
        need = TARGET_TRACK_COUNT - num_old
        if need >= MIN_REPLENISH:
            corners = self._detect_corners(cam0_img, cur_pts, need)
            cur_pts = np.vstack([cur_pts, corners])
            guess1 = np.vstack([guess1, corners])  # no prior: start the cam1 search at the same pixel
        # Tracks that had no cam1 position carry NaN guesses; fall back to the cam0 position.
        guess1 = np.where(np.isnan(guess1), cur_pts, guess1)

        valid = np.zeros(len(cur_pts), dtype=bool)
        pts1 = np.zeros((len(cur_pts), 2), np.float32)
        points_3d = np.zeros((3, len(cur_pts)))
        if len(cur_pts):
            valid, pts1, points_3d = self._stereo(cam0_img, cam1_img, cur_pts, guess1)

        # A new corner is kept only if its stereo measurement is valid and its depth plausible.
        keep = np.ones(len(cur_pts), dtype=bool)
        new = np.arange(len(cur_pts)) >= num_old
        depth_ok = (points_3d[2] > SEED_DEPTH_MIN) & (points_3d[2] < SEED_DEPTH_MAX)
        keep[new] = (valid & depth_ok)[new]
        cur_pts, pts1, points_3d, valid = cur_pts[keep], pts1[keep], points_3d[:, keep], valid[keep]
        num_new = int(new[keep].sum())
        new_ids = np.arange(self._next_track_id, self._next_track_id + num_new, dtype=np.int64)
        self._next_track_id += num_new
        cur_ids = np.concatenate([cur_ids, new_ids])

        frame = OpticalFlowFrame(
            timestamp_ns=ts,
            track_uv={int(t): (float(p[0]), float(p[1])) for t, p in zip(cur_ids, cur_pts)},
            stereo_uv1={int(t): (float(p[0]), float(p[1])) for t, p, v in zip(cur_ids, pts1, valid) if v},
            stereo_point_cam0={int(t): points_3d[:, i] for i, (t, v) in enumerate(zip(cur_ids, valid)) if v},
        )
        self._prev_img = cam0_img
        self._prev_pts = cur_pts
        self._prev_pts1 = np.where(valid[:, None], pts1, np.nan).astype(np.float32)
        self._prev_ids = cur_ids
        return frame
