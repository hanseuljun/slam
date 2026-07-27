import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult
from slam.stereo_matching import match_and_triangulate_stereo

# Pyramidal Lucas-Kanade parameters. 21x21/3 levels is the standard VIO choice (e.g. VINS-Mono) --
# large enough a window to survive motion blur, enough pyramid levels to handle a fast frame's
# displacement without needing a bigger window (which would hurt localization accuracy).
LK_WIN_SIZE = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
# Forward-backward round-trip tolerance [px]: track prev->cur, then cur->prev, and drop any point
# that doesn't land back near where it started. KLT can drift onto a nearby similar-looking patch
# without ever setting its own failure flag; this is the standard correctness gate for that.
FB_THRESHOLD_PX = 1.0
# Replenish new seed tracks (from this frame's already-detected ORB keypoints) whenever the alive
# count drops below this, up to the same target -- keeps density roughly constant as tracks are
# lost to occlusion, leaving the frame, or failing the forward-backward check.
TARGET_TRACK_COUNT = 300
# Don't seed a new track this close to an already-alive one -- new points should fill gaps left by
# lost tracks, not cluster redundantly on top of survivors.
MIN_TRACK_SEPARATION_PX = 8.0
# Sanity bounds [m] on a seed's triangulated depth -- matches slam.py's DEPTH_MIN/DEPTH_MAX. Even
# after the ratio test and epipolar gate, a small fraction of stereo matches are still spurious
# (repeated texture, near-parallel rays) and triangulate to a negative or absurd depth; reject
# those before they become a track rather than seed a landmark with an unusable initial guess.
SEED_DEPTH_MIN = 0.3
SEED_DEPTH_MAX = 40.0


@dataclass
class OpticalFlowFrame:
    timestamp_ns: int
    track_uv: dict[int, tuple[float, float]] = field(default_factory=dict)  # track id -> cam0 (u, v)
    # track id -> stereo-triangulated 3D point in the cam0 frame, for tracks first seen (seeded)
    # at this frame only -- a track's one-time depth, computed once at birth and never again (see
    # slam.py's LandmarkObservation: it seeds a landmark's initial 3D guess, the graph refines the
    # actual position from many later reprojection observations, not from repeated triangulation).
    seeded_point_cam0: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class OpticalFlowResult:
    frames: list[OpticalFlowFrame]
    elapsed_s: float


class OpticalFlowSolver:
    """Tracks cam0 points continuously frame-to-frame via pyramidal KLT, rather than
    independently re-detecting and re-matching ORB descriptors at each keyframe.

    A point's track id is stable for as long as it stays alive, so downstream code gets temporal
    identity for free -- no re-matching needed. This is the standard technique real-time VIO
    systems (VINS-Mono, OKVIS, ORB-SLAM's tracking thread) use for exactly the failure mode that
    broke keyframe-to-keyframe ORB matching on V1_03_difficult's fast-rotation segments: KLT only
    ever has to survive one frame's motion (~0.05s here) instead of a keyframe gap's (~0.15-0.5s),
    so it never needs to *re-identify* a point after enough appearance change has accumulated to
    fool a descriptor.

    Seed points are drawn from this frame's already-detected ORB keypoints (feature_detection.py)
    rather than a separate detector call, reusing work already done upstream. A candidate only
    becomes a track if it also finds a valid, epipolar-consistent cam0<->cam1 stereo match (the
    same geometry stereo_matching.py uses, via match_and_triangulate_stereo) -- this gives every
    track a metric depth at birth instead of seeding blind, undepthed points, and the match is
    only ever computed once, at seed time (see OpticalFlowFrame.seeded_point_cam0).
    """

    def __init__(
        self, data: EuRoCMAVData, feature_detection_result: FeatureDetectionResult,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress: float = 0.0

    def run(self) -> OpticalFlowResult:
        frames = self._feature_detection_result.frames
        n = len(frames)
        result_frames: list[OpticalFlowFrame] = []
        t0 = time.monotonic()

        prev_img: Optional[np.ndarray] = None
        prev_pts = np.zeros((0, 1, 2), dtype=np.float32)
        prev_ids = np.zeros((0,), dtype=np.int64)
        next_track_id = 0

        for i, fd in enumerate(frames):
            if self._cancel_event.is_set():
                break
            cur_img = cv2.imread(str(self._data.get_cam0_image_path(fd.timestamp_ns)), cv2.IMREAD_GRAYSCALE)

            if prev_img is not None and len(prev_pts):
                # cv2's stub requires `nextPts` as a non-Optional MatLike, but passing None (let
                # OpenCV allocate the output) is the standard idiom.
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_img, cur_img, prev_pts, None,  # type: ignore[call-overload]
                    winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA)
                back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    cur_img, prev_img, next_pts, None,  # type: ignore[call-overload]
                    winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA)
                status = status.reshape(-1).astype(bool)
                back_status = back_status.reshape(-1).astype(bool)
                fb_err = np.linalg.norm((prev_pts - back_pts).reshape(-1, 2), axis=1)
                h, w = cur_img.shape
                pts_flat = next_pts.reshape(-1, 2)
                in_bounds = (pts_flat[:, 0] >= 0) & (pts_flat[:, 0] < w) & (pts_flat[:, 1] >= 0) & (pts_flat[:, 1] < h)
                keep = status & back_status & (fb_err < FB_THRESHOLD_PX) & in_bounds
                cur_pts = next_pts[keep]
                cur_ids = prev_ids[keep]
            else:
                cur_pts = np.zeros((0, 1, 2), dtype=np.float32)
                cur_ids = np.zeros((0,), dtype=np.int64)

            seeded_point_cam0: dict[int, np.ndarray] = {}
            if len(cur_pts) < TARGET_TRACK_COUNT and fd.cam0_keypoints:
                candidate_idx = list(range(len(fd.cam0_keypoints)))
                candidate_pts = np.array([fd.cam0_keypoints[k].pt for k in candidate_idx], dtype=np.float32)
                if len(cur_pts):
                    alive = cur_pts.reshape(-1, 2)
                    d = np.linalg.norm(candidate_pts[:, None, :] - alive[None, :, :], axis=2)
                    far_enough = d.min(axis=1) >= MIN_TRACK_SEPARATION_PX
                    candidate_idx = [k for k, keep_k in zip(candidate_idx, far_enough) if keep_k]

                if candidate_idx:
                    cand_cam0_kps = [fd.cam0_keypoints[k] for k in candidate_idx]
                    cand_cam0_descs = fd.cam0_descriptors[candidate_idx]
                    stereo_matches, points_3d = match_and_triangulate_stereo(
                        self._data, cand_cam0_kps, cand_cam0_descs, fd.cam1_keypoints, fd.cam1_descriptors)
                    depth_ok = (points_3d[2] > SEED_DEPTH_MIN) & (points_3d[2] < SEED_DEPTH_MAX)
                    stereo_matches = [m for m, ok in zip(stereo_matches, depth_ok) if ok]
                    points_3d = points_3d[:, depth_ok]

                    need = TARGET_TRACK_COUNT - len(cur_pts)
                    new_pts_list = []
                    new_ids_list = []
                    for col, m in enumerate(stereo_matches[:need]):
                        tid = next_track_id
                        next_track_id += 1
                        new_pts_list.append(cand_cam0_kps[m.queryIdx].pt)
                        new_ids_list.append(tid)
                        seeded_point_cam0[tid] = points_3d[:, col]

                    if new_pts_list:
                        new_pts = np.array(new_pts_list, dtype=np.float32).reshape(-1, 1, 2)
                        new_ids = np.array(new_ids_list, dtype=np.int64)
                        cur_pts = np.vstack([cur_pts, new_pts]) if len(cur_pts) else new_pts
                        cur_ids = np.concatenate([cur_ids, new_ids]) if len(cur_ids) else new_ids

            track_uv = {int(tid): (float(pt[0, 0]), float(pt[0, 1])) for tid, pt in zip(cur_ids, cur_pts)}
            result_frames.append(OpticalFlowFrame(
                timestamp_ns=fd.timestamp_ns, track_uv=track_uv, seeded_point_cam0=seeded_point_cam0))

            prev_img, prev_pts, prev_ids = cur_img, cur_pts, cur_ids
            self.progress = (i + 1) / n

        elapsed_s = time.monotonic() - t0
        return OpticalFlowResult(frames=result_frames, elapsed_s=elapsed_s)
