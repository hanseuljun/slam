from collections import defaultdict
from dataclasses import dataclass
import threading
import time
import traceback
from typing import Callable, Optional

import cv2
import gtsam
import numpy as np

from slam.data import EuRoCMAVData, ImuSample
from slam.feature_detection import FeatureDetectionFrame, FeatureDetectionResult
from slam.frontend import FrontendFrameComputer, FrontendResult
from slam.optical_flow import OpticalFlowFrame, OpticalFlowResult
from slam.orb_vocabulary import ORBVocabulary
from slam.stereo_matching import StereoMatchingFrame, StereoMatchingResult
from slam.util import quaternion_to_rotation_matrix

RPE_DELTA_S = 1.0  # RPE window: how far apart (in time) the two poses being compared are [s]


@dataclass
class SlamGroundTruthResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamImuResult:
    times: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocities: np.ndarray
    linear_accelerations: np.ndarray


@dataclass
class SlamPnpResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray
    elapsed_time: float = 0.0


@dataclass
class SlamGtsamResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    velocities: np.ndarray
    biases: np.ndarray  # per-keyframe IMU bias, shape (K, 6): [accel(3), gyro(3)]
    position_errors: np.ndarray  # per-keyframe position error vs nearest GT sample [m], shape (K,)
    # ATE: whole-trajectory yaw+translation alignment, then per-keyframe error. Yaw-only
    # because gravity makes roll/pitch observable in VIO (Zhang & Scaramuzza, IROS 2018).
    ate_position_errors: np.ndarray  # shape (K,) [m]
    ate_rotation_errors: np.ndarray  # shape (K,) [deg]
    # RPE: relative motion error over a fixed time window. Needs no alignment, so it isolates
    # local accuracy from drift. NaN where no keyframe falls within the window of the
    # trajectory's end.
    rpe_translation_errors: np.ndarray  # shape (K,) [m]
    rpe_rotation_errors: np.ndarray  # shape (K,) [deg]
    reprojection_rmse: np.ndarray  # per-keyframe landmark reprojection RMSE [px], shape (K,)
    landmark_counts: np.ndarray  # per-keyframe number of observed landmarks, shape (K,)
    # World-frame position of every inserted landmark, shape (M, 3). Only populated at
    # finalize(); empty in live partial results.
    landmark_positions: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray
    linear_accelerations: np.ndarray
    elapsed_time: float = 0.0


@dataclass
class SlamExtraResult:
    gravity: np.ndarray
    linear_accelerations_in_world: np.ndarray


@dataclass
class SlamResult:
    gt: SlamGroundTruthResult
    imu: SlamImuResult
    pnp: SlamPnpResult
    gtsam: SlamGtsamResult
    extra: SlamExtraResult



def _mats_to_rvecs(rotation_matrices: np.ndarray) -> np.ndarray:
    return np.array([cv2.Rodrigues(R)[0].flatten() for R in rotation_matrices])


def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    T[:3, 3] = tvec.flatten()
    return T


def _to_gtsam_pose3(T: np.ndarray) -> gtsam.Pose3:
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(*T[:3, 3]))


def _elapsed_s(timestamp_ns: int, reference_ns: int) -> float:
    return (timestamp_ns - reference_ns) / 1e9


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R such that R @ a is parallel to b (both unit-normalized)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        # a and b are antiparallel; rotate 180 deg about any axis orthogonal to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _rotation_angle_deg(rotation_matrix: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix (its distance from identity), in degrees."""
    cos_angle = np.clip((np.trace(rotation_matrix) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _nearest_sorted_indices(sorted_reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """For each query, the index into sorted_reference of the closest value. Binary search instead
    of a dense broadcast, which is O(n*m) in memory (~6.7GB for 29k x 29k samples).
    """
    idx = np.searchsorted(sorted_reference, queries, side="left")
    idx = np.clip(idx, 1, len(sorted_reference) - 1)
    left, right = sorted_reference[idx - 1], sorted_reference[idx]
    return idx - ((queries - left) <= (right - queries))


def _yaw_translation_align(est_positions: np.ndarray, gt_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares yaw (rotation about z) + translation aligning est_positions to gt_positions.

    Yaw-only rather than full 3D (Umeyama): the nav frame is gravity-aligned and roll/pitch are
    observable for VIO, so a full rotation could hide a real tilt error. Closed form: the optimal
    translation is the centroid offset, and theta = atan2(B, A) maximizes A*cos(theta) +
    B*sin(theta).

    Needs a few meters of horizontal travel; over a short or stationary window the yaw, and so the
    ATE rotation error, is noisy.
    """
    p_mean, q_mean = est_positions.mean(axis=0), gt_positions.mean(axis=0)
    p, q = est_positions - p_mean, gt_positions - q_mean
    a = float(np.sum(q[:, 0] * p[:, 0] + q[:, 1] * p[:, 1]))
    b = float(np.sum(q[:, 1] * p[:, 0] - q[:, 0] * p[:, 1]))
    theta = np.arctan2(b, a)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = q_mean - R @ p_mean
    return R, t


def _compute_ate(
    positions: np.ndarray, rotation_matrices: np.ndarray,
    gt_positions: np.ndarray, gt_rotation_matrices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Absolute Trajectory Error: align the whole estimated trajectory to ground truth once (yaw +
    translation), then report per-keyframe position/rotation error.
    """
    R, t = _yaw_translation_align(positions, gt_positions)
    aligned_positions = positions @ R.T + t
    position_errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)
    aligned_rotations = np.einsum('ij,kjl->kil', R, rotation_matrices)
    rotation_errors = np.array([
        _rotation_angle_deg(gt_rotation_matrices[i].T @ aligned_rotations[i])
        for i in range(len(positions))
    ])
    return position_errors, rotation_errors


def _compute_rpe(
    times: np.ndarray, positions: np.ndarray, rotation_matrices: np.ndarray,
    gt_positions: np.ndarray, gt_rotation_matrices: np.ndarray, delta_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative Pose Error: for each keyframe, compare estimated vs ground-truth relative motion
    over the next delta_s seconds. Needs no alignment.
    """
    K = len(times)
    translation_errors = np.full(K, np.nan)
    rotation_errors = np.full(K, np.nan)
    for i in range(K):
        if times[i] + delta_s > times[-1]:
            continue
        j = int(np.argmin(np.abs(times - (times[i] + delta_s))))
        if j <= i:
            continue
        rel_est = np.linalg.inv(rotation_matrices[i]) @ (positions[j] - positions[i])
        rel_est_R = rotation_matrices[i].T @ rotation_matrices[j]
        rel_gt = gt_rotation_matrices[i].T @ (gt_positions[j] - gt_positions[i])
        rel_gt_R = gt_rotation_matrices[i].T @ gt_rotation_matrices[j]
        translation_errors[i] = float(np.linalg.norm(rel_est - rel_gt))
        rotation_errors[i] = _rotation_angle_deg(rel_gt_R.T @ rel_est_R)
    return translation_errors, rotation_errors


def _get_ground_truth_result(
    data: EuRoCMAVData,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
) -> SlamGroundTruthResult:
    samples = [s for s in data.ground_truth_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    times = np.array([_elapsed_s(s.timestamp_ns, first_timestamp_ns) for s in samples])
    positions = np.array([s.position for s in samples])
    rotation_matrices = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in samples])

    angular_velocities = []
    for j in range(len(samples) - 1):
        rotation = rotation_matrices[j].T @ rotation_matrices[j + 1]
        rotation_vector, _ = cv2.Rodrigues(rotation)
        dt = _elapsed_s(samples[j + 1].timestamp_ns, samples[j].timestamp_ns)
        angular_velocity = rotation_vector.flatten() / dt
        angular_velocities.append(angular_velocity)
    angular_velocities = np.array(angular_velocities)

    return SlamGroundTruthResult(
        times=times,
        positions=positions,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocity_times=np.array([_elapsed_s(s.timestamp_ns, first_timestamp_ns) for s in samples[:-1]]),
        angular_velocities=angular_velocities,
    )


def _get_imu_result(
    data: EuRoCMAVData,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
    gt_rotation_matrices: np.ndarray,
    first_gt_timestamp_ns: int,
) -> SlamImuResult:
    samples = [s for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    times = np.array([_elapsed_s(s.timestamp_ns, first_timestamp_ns) for s in samples])
    linear_accelerations = np.array([s.linear_acceleration for s in samples])
    angular_velocities = np.array([s.angular_velocity for s in samples])
    rotation_matrices_list = []
    prev_rotation_matrix = np.eye(3)
    for angular_velocity in angular_velocities:
        rotation_matrix, _ = cv2.Rodrigues(angular_velocity / data.imu0_rate_hz)
        prev_rotation_matrix = prev_rotation_matrix @ rotation_matrix
        rotation_matrices_list.append(prev_rotation_matrix)

    timestamps_ns = np.array([s.timestamp_ns for s in samples])
    index_closest_to_first_gt_sample = np.argmin(np.abs(timestamps_ns - first_gt_timestamp_ns))
    compensation_rotation_matrix = gt_rotation_matrices[0] @ rotation_matrices_list[index_closest_to_first_gt_sample].T
    for i in range(len(rotation_matrices_list)):
        rotation_matrices_list[i] = compensation_rotation_matrix @ rotation_matrices_list[i]

    rotation_matrices = np.array(rotation_matrices_list)

    return SlamImuResult(
        times=times,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocities=angular_velocities,
        linear_accelerations=linear_accelerations,
    )


def _run_pnp_step(
    data: EuRoCMAVData,
    points_3d: np.ndarray,
    stereo_matches: list,
    cam0_descriptors0: np.ndarray,
    cam0_keypoints1,
    cam0_descriptors1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    cam0_idx_to_3d_idx = {m.queryIdx: i for i, m in enumerate(stereo_matches)}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    temporal_matches = bf.knnMatch(cam0_descriptors0, cam0_descriptors1, k=2)
    temporal_good_matches = [m for m, n in temporal_matches if m.distance < 0.75 * n.distance]

    object_points = []
    image_points = []
    for m in temporal_good_matches:
        if m.queryIdx in cam0_idx_to_3d_idx:
            idx_3d = cam0_idx_to_3d_idx[m.queryIdx]
            object_points.append(points_3d[:, idx_3d])
            image_points.append(cam0_keypoints1[m.trainIdx].pt)

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    intrinsics_matrix = data.cam0_intrinsics.to_matrix()
    dist_coeffs = np.array([
        data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1, data.cam0_intrinsics.p2,
    ])

    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(object_points, image_points, intrinsics_matrix, dist_coeffs)
    if not success:
        raise RuntimeError(f"cv2.solvePnPRansac failed. len(object_points): {len(object_points)}")

    # cv2's stub types `inliers` too loosely for numpy's __getitem__ overloads to match, even
    # though this is ordinary integer-array indexing at runtime.
    inlier_object_points = object_points[inliers.flatten()]  # type: ignore[index]
    inlier_image_points = image_points[inliers.flatten()]  # type: ignore[index]
    projected, _ = cv2.projectPoints(inlier_object_points, rotation_vector, translation_vector, intrinsics_matrix, dist_coeffs)
    reprojection_error = np.mean(np.linalg.norm(inlier_image_points - projected.reshape(-1, 2), axis=1))

    # cv2.solvePnPRansac returns the world<-camera transform; the rest of this module treats
    # poses as camera<-world (a camera's pose expressed in world coordinates), so invert here.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rotation_vector = -rotation_vector
    translation_vector = -rotation_matrix.T @ translation_vector

    return rotation_vector, translation_vector, len(temporal_good_matches), reprojection_error


@dataclass(frozen=True)
class LoopClosureCandidate:
    from_frame: int              # earlier keyframe (into stereo_matching_result.frames)
    to_frame: int                # later keyframe ("query") that revisits from_frame
    body_relative_pose: np.ndarray  # 4x4, from_frame_body <- to_frame_body
    num_matches: int
    reprojection_error: float


@dataclass(frozen=True)
class LoopClosureFactor:
    """A closure ready to insert into the pose graph. from_frame/to_frame are frame indices, not
    node indices.
    """
    from_frame: int
    to_frame: int
    body_relative_pose: np.ndarray  # 4x4, from_frame_body <- to_frame_body
    rot_sigma: float   # rad
    trans_sigma: float  # m
    noise_mode: str    # "gaussian" | "huber" | "dcs" -- see _run_gtsam's noise construction


@dataclass(frozen=True)
class _LoopClosureEdge:
    """LoopClosureFactor (or an auto-detected LoopClosureCandidate) with both endpoints resolved
    to graph node indices."""
    from_node: int
    to_node: int
    body_relative_pose: np.ndarray
    rot_sigma: float
    trans_sigma: float
    noise_mode: str


# Sanity bound on a verified closure's translation magnitude [m]. Match count and reprojection
# error don't catch degenerate PnP/RANSAC solutions (~1% of verified pairs on V2_03_difficult had
# translations in the millions of meters).
LOOP_CLOSURE_MAX_TRANSLATION_M = 20.0

# Noise for an auto-detected closure. Plain Gaussian: Huber and DCS cap or zero the pull of a
# correct closure when the prior disagreement is many sigma, which it is after a long blackout.
# False-positive protection lives in _LoopClosureDetector's verification gate.
LOOP_CLOSURE_ROT_SIGMA = 0.02   # rad
LOOP_CLOSURE_TRANS_SIGMA = 0.05  # m


# Candidates close together in both query and anchor time (chained: each compared to the previous
# cluster member) count as one revisit event.
LOOP_CLOSURE_CLUSTER_GAP_S = 2.0


class _LoopClosureDetector:
    """Detects loop closures one keyframe at a time; add_keyframe(node, frame) only searches nodes
    0..node-1.

    A bag-of-words shortlist (ORBVocabulary, see orb_vocabulary.py) ranks earlier keyframes cheaply,
    the best of the shortlist is re-scored with full descriptors, and the winner is verified by
    PnP/RANSAC (_run_pnp_step). A candidate must clear min_matches and the translation sanity bound.
    The vocabulary is a fixed asset trained offline on a different sequence, so using it adds no
    lookahead.

    Only the best earlier candidate is kept per query, and same-event clusters are consolidated into
    one representative (the member with the most temporal matches): a stay-put stretch otherwise
    rediscovers the same match at every keyframe, overcounting one observation as many. A cluster
    closes when a later non-matching candidate arrives or on flush(), a delay bounded by
    LOOP_CLOSURE_CLUSTER_GAP_S.

    One representative per cluster is a first guess and likely too aggressive; keeping a small
    spread per cluster is the obvious next step.
    """

    # Only the strongest PROBE_N descriptors (ORB returns them response-ordered) are assigned
    # to vocabulary words, bounding the cost. BoW only builds a SHORTLIST_N shortlist; those
    # are re-scored with full descriptors before picking the winner.
    PROBE_N = 500
    SHORTLIST_N = 5

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        min_temporal_gap_s: float = 10.0,
        min_matches: int = 200,
        vocab: Optional[ORBVocabulary] = None,
    ) -> None:
        self._data = data
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self._min_temporal_gap_s = min_temporal_gap_s
        self._min_matches = min_matches
        self._vocab = vocab if vocab is not None else ORBVocabulary.load()
        self._body_T_cam0 = data.cam0_extrinsics
        self._cam0_T_body = np.linalg.inv(self._body_T_cam0)
        self._first_ts = data.cam_timestamps_ns[0]
        self._kf_times: list[float] = []   # elapsed_s per node added so far
        self._kf_frames: list[int] = []    # original frame index per node added so far
        self._bow_vectors: list[dict[int, float]] = []      # this detector's own BoW vector, per node
        self._inverted_index: dict[int, list[int]] = {}     # vocabulary word id -> nodes containing it
        # (from_node, to_node, candidate) triples of the currently-open, not-yet-closed cluster.
        self._open_cluster: list[tuple[int, int, LoopClosureCandidate]] = []

    def add_keyframe(self, node: int, frame: int) -> list[_LoopClosureEdge]:
        """Search for a revisit at this keyframe among earlier ones, feeding any match into the
        consolidation state machine. Returns edges from a cluster that just closed (its to_node may
        be several keyframes back).
        """
        q_time = _elapsed_s(self._sm_result.frames[frame].timestamp_ns, self._first_ts)
        q_desc_full = self._fd_result.frames[frame].cam0_descriptors
        query_vec = self._vocab.transform(q_desc_full[:self.PROBE_N]) if q_desc_full is not None else {}
        result = self._search(frame, q_time, query_vec)
        # Index this node's own BoW vector *after* searching -- it must never match itself, and
        # only needs to be visible to later queries, never this one.
        self._kf_times.append(q_time)
        self._kf_frames.append(frame)
        self._bow_vectors.append(query_vec)
        for word in query_vec:
            self._inverted_index.setdefault(word, []).append(node)
        if result is None:
            return []
        from_node, candidate = result
        return self._offer(node, from_node, candidate)

    def flush(self) -> list[_LoopClosureEdge]:
        """Call once after the last add_keyframe: closes whatever cluster is still open."""
        return self._close_cluster()

    def _bow_shortlist(self, q_time: float, query_vec: dict[int, float]) -> list[int]:
        """Nodes sharing a vocabulary word with the query, ranked by BoW similarity (dot product of
        L2-normalized TF-IDF vectors) via the inverted index, at least min_temporal_gap_s older.
        """
        scores: dict[int, float] = {}
        for word, qw in query_vec.items():
            for c in self._inverted_index.get(word, ()):
                scores[c] = scores.get(c, 0.0) + qw * self._bow_vectors[c][word]
        eligible = [(s, c) for c, s in scores.items() if q_time - self._kf_times[c] >= self._min_temporal_gap_s]
        eligible.sort(key=lambda x: -x[0])
        return [c for _, c in eligible[:self.SHORTLIST_N]]

    def _search(
        self, frame: int, q_time: float, query_vec: dict[int, float],
    ) -> Optional[tuple[int, LoopClosureCandidate]]:
        if not query_vec:
            return None
        shortlist = self._bow_shortlist(q_time, query_vec)
        if not shortlist:
            return None
        q_desc_full = self._fd_result.frames[frame].cam0_descriptors

        # Re-score the BoW shortlist with full descriptors and pick the true best among them.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        best_score, best_c = -1, -1
        for c in shortlist:
            c_desc_full = self._fd_result.frames[self._kf_frames[c]].cam0_descriptors
            raw = bf.knnMatch(c_desc_full, q_desc_full, k=2)
            score = sum(1 for pair in raw if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)
            if score > best_score:
                best_score, best_c = score, c
        if best_score < self._min_matches:
            return None

        c_frame = self._kf_frames[best_c]
        c_sm = self._sm_result.frames[c_frame]
        c_fd = self._fd_result.frames[c_frame]
        q_fd = self._fd_result.frames[frame]
        if not c_sm.matches:
            return None
        try:
            rvec, tvec, num_matches, reproj_err = _run_pnp_step(
                self._data, c_sm.points_3d, c_sm.matches,
                c_fd.cam0_descriptors, q_fd.cam0_keypoints, q_fd.cam0_descriptors)
        except Exception:
            return None
        if num_matches < self._min_matches or float(np.linalg.norm(tvec)) > LOOP_CLOSURE_MAX_TRANSLATION_M:
            return None

        pnp_cam0 = _rvec_tvec_to_transform(rvec, tvec)
        rel_pose_body = self._body_T_cam0 @ pnp_cam0 @ self._cam0_T_body
        return best_c, LoopClosureCandidate(c_frame, frame, rel_pose_body, num_matches, float(reproj_err))

    def _offer(self, to_node: int, from_node: int, candidate: LoopClosureCandidate) -> list[_LoopClosureEdge]:
        if not self._open_cluster:
            self._open_cluster = [(from_node, to_node, candidate)]
            return []
        prev_from, prev_to, _ = self._open_cluster[-1]
        same_event = (abs(self._kf_times[to_node] - self._kf_times[prev_to]) <= LOOP_CLOSURE_CLUSTER_GAP_S
                      and abs(self._kf_times[from_node] - self._kf_times[prev_from]) <= LOOP_CLOSURE_CLUSTER_GAP_S)
        if same_event:
            self._open_cluster.append((from_node, to_node, candidate))
            return []
        finalized = self._close_cluster()
        self._open_cluster = [(from_node, to_node, candidate)]
        return finalized

    def _close_cluster(self) -> list[_LoopClosureEdge]:
        if not self._open_cluster:
            return []
        best_from, best_to, best = max(self._open_cluster, key=lambda x: x[2].num_matches)
        self._open_cluster = []
        return [_LoopClosureEdge(
            best_from, best_to, best.body_relative_pose,
            LOOP_CLOSURE_ROT_SIGMA, LOOP_CLOSURE_TRANS_SIGMA, "gaussian")]


def _get_pnp_result(
    data: EuRoCMAVData,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    pnp_poses: list[np.ndarray],
) -> SlamPnpResult:
    # pnp_poses is the per-frame cam0 dead-reckoning trajectory from the shared keyframe scan
    # (_scan_keyframes) -- no separate PnP pass is run here.
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames])
    # Anchor to GT at the first GT sample inside the window (after start_s), not the dataset's
    # first, which precedes the window.
    first_gt_sample = next(s for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns)
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt_sample.timestamp_ns))

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt_sample.quaternion)
    world_T_body_first[:3, 3] = first_gt_sample.position

    pnp_body_poses = [body_T_cam0 @ T @ cam0_T_body for T in pnp_poses]
    T_comp = world_T_body_first @ np.linalg.inv(pnp_body_poses[closest_cam_index])
    pnp_world_T_body = np.array([T_comp @ T for T in pnp_body_poses])

    pnp_times = np.array([
        _elapsed_s(stereo_matching_result.frames[i].timestamp_ns, first_timestamp_ns)
        for i in range(len(pnp_world_T_body))
    ])

    angular_velocities = []
    for i in range(len(pnp_world_T_body) - 1):
        rotation_matrix = pnp_world_T_body[i, :3, :3].T @ pnp_world_T_body[i + 1, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        angular_velocities.append(rotation_vector.flatten() * data.cam0_rate_hz)

    rotation_matrices = pnp_world_T_body[:, :3, :3]
    return SlamPnpResult(
        times=pnp_times,
        positions=pnp_world_T_body[:, :3, 3],
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocity_times=pnp_times[:-1],
        angular_velocities=np.array(angular_velocities),
    )


@dataclass(frozen=True)
class LandmarkObservation:
    """One sighting of a landmark at a keyframe. uv0/uv1 are undistorted pixels in cam0/cam1;
    point_cam0 is the stereo-triangulated point, used only to seed the landmark's initial position.
    """
    node: int  # graph-node index (into keyframe_indices) that observed the landmark
    uv0: np.ndarray
    uv1: np.ndarray
    point_cam0: np.ndarray


# A stereo-matched cam0 keypoint is matched to the optical-flow track within this many pixels. KLT
# tracks drift sub-pixel, so a small radius identifies the same corner without descriptors. 3.0
# was too tight on V1_03_difficult's fast rotations; keep it under
# optical_flow.MIN_TRACK_SEPARATION_PX (8) so it can't straddle two tracks.
LANDMARK_SNAP_PX = 6.0


def _snap_obs_to_tracks(
    fd: FeatureDetectionFrame, sm: StereoMatchingFrame, optical_flow_frame: OpticalFlowFrame,
) -> dict[int, int]:
    """obs index (into sm.matches) -> optical-flow track id for the nearest live track within
    LANDMARK_SNAP_PX. Each track claims at most one obs (its closest), since _LandmarkTrackBuilder
    and the graph assume one observation per track per node.
    """
    if not sm.matches or not optical_flow_frame.track_uv:
        return {}
    obs_pts = np.array([fd.cam0_keypoints[m.queryIdx].pt for m in sm.matches], dtype=np.float32)
    track_ids = list(optical_flow_frame.track_uv.keys())
    track_pts = np.array([optical_flow_frame.track_uv[tid] for tid in track_ids], dtype=np.float32)
    d = np.linalg.norm(obs_pts[:, None, :] - track_pts[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    nearest_dist = d[np.arange(len(obs_pts)), nearest]
    obs_to_flow_tid: dict[int, int] = {}
    claimed_tracks: set[int] = set()
    for i in np.argsort(nearest_dist):
        if nearest_dist[i] >= LANDMARK_SNAP_PX:
            break
        track_id = track_ids[nearest[i]]
        if track_id in claimed_tracks:
            continue
        obs_to_flow_tid[int(i)] = track_id
        claimed_tracks.add(track_id)
    return obs_to_flow_tid


class _LandmarkTrackBuilder:
    """Chains stereo-matched features into persistent landmark tracks, one keyframe at a time via
    add_keyframe.

    Only cam0-cam1 stereo inliers are tracked, so every observation has a metric depth for
    initialization; observations are pre-undistorted for a Cal3_S2 pinhole model. Correspondence
    across keyframes comes from optical flow's track ids, not descriptor re-matching (which fails on
    fast rotation, see OpticalFlowTracker).

    There is deliberately no minimum-track-length filter here, since a track's final length isn't
    known where it starts; _process_node_landmarks decides per node whether a track is mature enough
    for the graph.
    """

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        optical_flow_result: OpticalFlowResult,
        depth_min: float, depth_max: float,
    ) -> None:
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self._of_result = optical_flow_result
        self._depth_min = depth_min
        self._depth_max = depth_max
        self._K0 = data.cam0_intrinsics.to_matrix()
        self._K1 = data.cam1_intrinsics.to_matrix()
        self._dist0 = np.array([data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
                                data.cam0_intrinsics.p1, data.cam0_intrinsics.p2])
        self._dist1 = np.array([data.cam1_intrinsics.k1, data.cam1_intrinsics.k2,
                                data.cam1_intrinsics.p1, data.cam1_intrinsics.p2])
        self.tracks: dict[int, list[LandmarkObservation]] = {}
        self._next_tid = 0
        # optical-flow track id (as of the *last* add_keyframe call) -> our track id.
        self._prev_flow_tid_to_tid: dict[int, int] = {}

    def add_keyframe(self, node: int, frame: int) -> list[int]:
        """Process the keyframe at graph node `node` (original frame `frame`), extending
        self.tracks. Returns the track ids that got a new observation here (continued or new).
        """
        sm = self._sm_result.frames[frame]
        touched: list[int] = []
        if not sm.matches:
            self._prev_flow_tid_to_tid = {}
            return touched
        fd = self._fd_result.frames[frame]
        q_idx = [m.queryIdx for m in sm.matches]
        t_idx = [m.trainIdx for m in sm.matches]
        uv0 = np.array([fd.cam0_keypoints[i].pt for i in q_idx], dtype=np.float64)
        uv1 = np.array([fd.cam1_keypoints[i].pt for i in t_idx], dtype=np.float64)
        uv0u = cv2.undistortPoints(uv0.reshape(-1, 1, 2), self._K0, self._dist0, P=self._K0).reshape(-1, 2)
        uv1u = cv2.undistortPoints(uv1.reshape(-1, 1, 2), self._K1, self._dist1, P=self._K1).reshape(-1, 2)
        pts = sm.points_3d.T  # (M, 3), row i <-> sm.matches[i]

        obs_to_flow_tid = _snap_obs_to_tracks(fd, sm, self._of_result.frames[frame])
        cur_flow_tid_to_tid: dict[int, int] = {}
        for i in range(len(sm.matches)):
            flow_tid = obs_to_flow_tid.get(i)
            if flow_tid is None:
                continue
            landmark_obs = LandmarkObservation(node, uv0u[i], uv1u[i], pts[i])
            tid = self._prev_flow_tid_to_tid.get(flow_tid)
            if tid is None:
                # New track: gate its depth on this first observation.
                if not (self._depth_min < landmark_obs.point_cam0[2] < self._depth_max):
                    continue
                tid = self._next_tid
                self._next_tid += 1
                self.tracks[tid] = []
            self.tracks[tid].append(landmark_obs)
            cur_flow_tid_to_tid[flow_tid] = tid
            touched.append(tid)
        self._prev_flow_tid_to_tid = cur_flow_tid_to_tid
        return touched


def _scan_keyframes(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    N: int,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Batch version of _KeyframeScanner (unused by the app).

    A PnP from the current reference keyframe to frame i gives covisibility (match count) and
    relative pose; a keyframe opens on weak overlap, enough motion, or a hard time cap, all clamped
    by a minimum gap. Also returns the chained poses as a per-frame dead-reckoning trajectory.
    """
    MIN_GAP = 3                     # never place keyframes closer than this (avoid duplicate nodes)
    # Time-based, not frame-count, so a camera frame-rate drop doesn't silently widen the gap
    # (e.g. V2_03_difficult's exposure glitch).
    MAX_GAP_S = 0.75                # force a keyframe at least this often (bound IMU preint. drift)
    COVIS_RATIO = 0.6               # new keyframe once covisibility falls below this * baseline
    TRANS_THRESH = 0.2              # ... or once translation since the reference exceeds this [m]
    ROT_THRESH = np.deg2rad(10.0)   # ... or rotation exceeds this [rad]
    # Probe with only the strongest PROBE_N descriptors (ORB returns them response-ordered):
    # the covisibility ratio is scale-free and brute-force matching dominates the cost.
    PROBE_N = 500
    # A node on a "dead-vision" frame (blur, low texture) gets no reprojection factors, so
    # ISAM2 explains the mismatch by spiking the gyro bias and corrupts the pose (seen on
    # MH_02_easy). The floor is ~1st percentile of per-frame stereo-match counts.
    MIN_KF_MATCHES = 80
    # A dead-vision candidate is merged into the IMU-only gap, but only up to this bound; past
    # it, keep the candidate to cap preintegration drift.
    DROP_MAX_GAP_S = 2 * MAX_GAP_S

    def _stereo_count(frame_idx: int) -> int:
        return len(stereo_matching_result.frames[frame_idx].matches)

    keyframes = [0]

    def _accept_keyframe(candidate: int) -> None:
        """Add `candidate` as a keyframe unless it is dead-vision and merging wouldn't open too wide
        an IMU-only gap. Uses only the candidate's stereo count and the last kept keyframe's
        timestamp.
        """
        elapsed_s = _elapsed_s(stereo_matching_result.frames[candidate].timestamp_ns,
                               stereo_matching_result.frames[keyframes[-1]].timestamp_ns)
        if _stereo_count(candidate) < MIN_KF_MATCHES and elapsed_s <= DROP_MAX_GAP_S:
            return
        keyframes.append(candidate)

    poses: list[Optional[np.ndarray]] = [None] * N
    poses[0] = np.eye(4)
    ref_idx = 0
    ref_covis: Optional[int] = None
    i = 1
    while i < N:
        if on_progress is not None:
            on_progress(i / N)
        ref_fd = feature_detection_result.frames[ref_idx]
        ref_sm = stereo_matching_result.frames[ref_idx]
        cur_fd = feature_detection_result.frames[i]
        try:
            rvec, tvec, num_matches, _ = _run_pnp_step(
                data, ref_sm.points_3d, ref_sm.matches,
                ref_fd.cam0_descriptors[:PROBE_N], cur_fd.cam0_keypoints, cur_fd.cam0_descriptors[:PROBE_N],
            )
        except Exception:
            # Overlap with the reference is gone: anchor a keyframe at the last connected frame
            # (or one past the reference if that is already frame i-1), then re-evaluate i.
            kf = max(i - 1, ref_idx + 1)
            _accept_keyframe(kf)
            if poses[kf] is None:  # this frame never got its own PnP; carry the reference pose
                poses[kf] = poses[ref_idx]
            ref_idx, ref_covis = kf, None
            i = kf + 1
            continue

        step = _rvec_tvec_to_transform(rvec, tvec)
        # poses[ref_idx] is always populated here (seeded at 0; ref_idx only points at
        # visited frames).
        poses[i] = poses[ref_idx] @ step  # type: ignore[operator]

        if ref_covis is None:  # first frame after a keyframe sets the covisibility baseline
            ref_covis = num_matches

        gap = i - ref_idx
        translation = float(np.linalg.norm(tvec))
        rotation = float(np.linalg.norm(rvec))
        elapsed_s = _elapsed_s(stereo_matching_result.frames[i].timestamp_ns,
                               stereo_matching_result.frames[ref_idx].timestamp_ns)

        force = elapsed_s >= MAX_GAP_S
        allow = gap >= MIN_GAP
        weak = num_matches < COVIS_RATIO * ref_covis
        moved = translation > TRANS_THRESH or rotation > ROT_THRESH

        if force or (allow and (weak or moved)):
            _accept_keyframe(i)
            ref_idx, ref_covis = i, None
        i += 1

    # No forced keyframe at the window's last frame (that would need to know where the data
    # ends): MAX_GAP_S and DROP_MAX_GAP_S already bound the tail gap.

    # Forward-fill any frame that never received a pose (rare: skipped by an exception jump).
    last = np.eye(4)
    filled = []
    for p in poses:
        last = p if p is not None else last
        filled.append(last)
    return keyframes, filled


class _KeyframeScanner:
    """Incremental version of _scan_keyframes, fed one frame at a time via add_frame(i).

    Frame i's fd/sm must already be in the result containers passed to __init__; later frames are
    never touched. Returns the frame indices that became keyframes from processing frame i: usually
    0 or 1, occasionally 2 (the exception-retry path can resolve an earlier reference's fate).
    """
    MIN_GAP = 3
    MAX_GAP_S = 0.75
    COVIS_RATIO = 0.6
    TRANS_THRESH = 0.2
    ROT_THRESH = np.deg2rad(10.0)
    PROBE_N = 500
    MIN_KF_MATCHES = 80
    DROP_MAX_GAP_S = 2 * MAX_GAP_S

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        n: int,
    ) -> None:
        self._data = data
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self.keyframes: list[int] = [0]
        # Pre-sized: the exception-retry path can write poses[kf] for a kf that isn't the
        # highest index reached so far.
        self._poses: list[Optional[np.ndarray]] = [None] * n
        self._poses[0] = np.eye(4)
        self._ref_idx = 0
        self._ref_covis: Optional[int] = None

    def _stereo_count(self, frame_idx: int) -> int:
        return len(self._sm_result.frames[frame_idx].matches)

    def _accept_keyframe(self, candidate: int) -> Optional[int]:
        elapsed_s = _elapsed_s(self._sm_result.frames[candidate].timestamp_ns,
                               self._sm_result.frames[self.keyframes[-1]].timestamp_ns)
        if self._stereo_count(candidate) < self.MIN_KF_MATCHES and elapsed_s <= self.DROP_MAX_GAP_S:
            return None
        self.keyframes.append(candidate)
        return candidate

    def add_frame(self, i: int) -> list[int]:
        accepted: list[int] = []
        while True:
            ref_fd = self._fd_result.frames[self._ref_idx]
            ref_sm = self._sm_result.frames[self._ref_idx]
            cur_fd = self._fd_result.frames[i]
            try:
                rvec, tvec, num_matches, _ = _run_pnp_step(
                    self._data, ref_sm.points_3d, ref_sm.matches,
                    ref_fd.cam0_descriptors[:self.PROBE_N], cur_fd.cam0_keypoints,
                    cur_fd.cam0_descriptors[:self.PROBE_N],
                )
            except Exception:
                # Overlap with the reference is gone: anchor a keyframe at the
                # last connected frame (kf is i-1 or i). If i-1, retry frame i
                # against the new reference; if i, nothing more to do.
                kf = max(i - 1, self._ref_idx + 1)
                got = self._accept_keyframe(kf)
                if got is not None:
                    accepted.append(got)
                if self._poses[kf] is None:  # never got its own PnP; carry the reference pose
                    self._poses[kf] = self._poses[self._ref_idx]
                self._ref_idx, self._ref_covis = kf, None
                if kf >= i:
                    return accepted
                continue

            step = _rvec_tvec_to_transform(rvec, tvec)
            # self._poses[self._ref_idx] is always populated by the time it's read here (index 0
            # is seeded in __init__, and _ref_idx only ever points at an already-visited frame).
            self._poses[i] = self._poses[self._ref_idx] @ step  # type: ignore[operator]

            if self._ref_covis is None:  # first frame after a keyframe sets the covisibility baseline
                self._ref_covis = num_matches

            gap = i - self._ref_idx
            translation = float(np.linalg.norm(tvec))
            rotation = float(np.linalg.norm(rvec))
            elapsed_s = _elapsed_s(self._sm_result.frames[i].timestamp_ns,
                                   self._sm_result.frames[self._ref_idx].timestamp_ns)

            force = elapsed_s >= self.MAX_GAP_S
            allow = gap >= self.MIN_GAP
            weak = num_matches < self.COVIS_RATIO * self._ref_covis
            moved = translation > self.TRANS_THRESH or rotation > self.ROT_THRESH

            if force or (allow and (weak or moved)):
                got = self._accept_keyframe(i)
                if got is not None:
                    accepted.append(got)
                self._ref_idx, self._ref_covis = i, None
            return accepted

    @staticmethod
    def _forward_fill(poses: list[Optional[np.ndarray]]) -> list[np.ndarray]:
        last = np.eye(4)
        filled = []
        for p in poses:
            last = p if p is not None else last
            filled.append(last)
        return filled

    def current_poses(self, up_to_frame: int) -> list[np.ndarray]:
        """Forward-filled poses for frames [0, up_to_frame]; safe to call mid-scan."""
        return self._forward_fill(self._poses[:up_to_frame + 1])

    def finalize(self) -> list[np.ndarray]:
        """Call once after every frame in the window has been fed via add_frame: forward-fills
        any frame that never received its own pose (rare -- see add_frame's exception path)."""
        return self._forward_fill(self._poses)


def _run_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    optical_flow_result: OpticalFlowResult,
    imu_samples: list[ImuSample],
    gravity: np.ndarray,
    keyframe_indices: list[int],
    on_progress: Callable[[float, str], None],
    # Run automated loop-closure detection and insert what it finds. Ignored if
    # extra_loop_closures is given.
    enable_loop_closure: bool = False,
    # Manual override / test hook, in frame indices (resolved to nodes internally). noise_mode
    # is "gaussian", "huber" or "dcs"; use "gaussian".
    extra_loop_closures: Optional[list[LoopClosureFactor]] = None,
    # Test-only: if given, filled with {'isam2': isam2}.
    debug_out: Optional[dict] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, list[int]]:
    N = len(stereo_matching_result.frames)
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    # Read-only: otherwise NumPy's temporary elision can run `imu_timestamps_ns - scalar` in
    # place, silently making every timestamp relative and emptying every IMU window.
    imu_timestamps_ns.flags.writeable = False
    imu_lin_accs      = np.array([s.linear_acceleration for s in imu_samples])
    imu_ang_vels      = np.array([s.angular_velocity for s in imu_samples])
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    # Gravity direction in the body frame at frame 0: average the accelerometer over a short
    # window there. A global most-static window would be at a different orientation.
    grav_win = max(1, int(data.imu0_rate_hz * 0.5))
    i0 = int(np.argmin(np.abs(imu_timestamps_ns - cam_timestamps_ns[0])))
    gravity_in_body = imu_lin_accs[i0:i0 + grav_win].mean(axis=0)

    # Progress budget (fractions of the GTSAM phase): setup [0, 0.10], ISAM2 loop [0.10,
    # 0.92], metrics [0.92, 1]. Loop-closure detection runs interleaved with the ISAM2 loop.
    # keyframe_indices maps graph node j -> frame index.
    K = len(keyframe_indices)

    X = lambda i: gtsam.symbol('x', i)
    V = lambda i: gtsam.symbol('v', i)
    B = lambda i: gtsam.symbol('b', i)
    L = lambda i: gtsam.symbol('l', i)

    imu_params = gtsam.PreintegrationParams(gravity)
    imu_params.setGyroscopeCovariance(np.eye(3) * 1e-4)
    # Keep the accelerometer factor from over-dominating the PnP between-factors; too small
    # a covariance lets the IMU dead-reckon (and drift) despite accurate vision constraints.
    imu_params.setAccelerometerCovariance(np.eye(3) * 1e-3)
    imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)

    PRIOR_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    PRIOR_VEL_NOISE  = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    # Loose enough that the optimizer can actually estimate the (constant) IMU bias. A tight
    # prior here pins bias to zero, so the real accelerometer bias double-integrates into drift.
    PRIOR_BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    # Per-step (adjacent-frame) PnP sigmas. A keyframe constraint chains the steps, so sigmas
    # are scaled by sqrt(gap) in the loop.
    PNP_STEP_SIGMAS  = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    # Random-walk noise letting the IMU bias evolve between keyframes: std over dt is
    # density*sqrt(dt) per axis, with the datasheet diffusion (EuRoC imu0/sensor.yaml). Both
    # terms are tight on purpose: a loose gyro sigma let the optimizer absorb pose error by
    # spiking the gyro bias (MH_02_easy ~20 s), and a loose accel sigma let accel bias drift
    # in axes vision can't correct (z on level flight).
    GYRO_BIAS_RW  = 1.9393e-05   # [rad/s^2/sqrt(Hz)]  gyroscope_random_walk from imu0/sensor.yaml
    ACCEL_BIAS_RW = 3.0000e-03   # [m/s^3/sqrt(Hz)]    accelerometer_random_walk from imu0/sensor.yaml
    # Weak prior at the IMU-predicted pose for a keyframe with neither a PnP factor nor any
    # reprojection factor, which would otherwise leave ISAM2 indeterminate (yaw unobservable).
    FALLBACK_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)

    # --- Landmarks --- Persistent 3D points tied to poses by reprojection factors: errors
    # average across observations instead of compounding along a PnP chain.
    MIN_TRACK_LEN   = 3       # observations before a landmark is trusted enough to add
    PNP_FALLBACK_COVIS = 15   # if a keyframe pair shares >= this many landmarks, drop PnP
    PX_SIGMA        = 1.5     # reprojection sigma [px]
    DEPTH_MIN, DEPTH_MAX = 0.3, 40.0
    # cam0/cam1 pinhole calibrations (measurements are pre-undistorted) and body<-cam poses.
    cam0_K = gtsam.Cal3_S2(data.cam0_intrinsics.fx, data.cam0_intrinsics.fy, 0.0,
                           data.cam0_intrinsics.cx, data.cam0_intrinsics.cy)
    cam1_K = gtsam.Cal3_S2(data.cam1_intrinsics.fx, data.cam1_intrinsics.fy, 0.0,
                           data.cam1_intrinsics.cx, data.cam1_intrinsics.cy)
    cam0_pose = gtsam.Pose3(body_T_cam0)
    cam1_pose = gtsam.Pose3(data.cam1_extrinsics)
    # Robust (Huber) pixel noise so a single bad match can't drag a landmark or a pose.
    PX_NOISE = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345),
        gtsam.noiseModel.Isotropic.Sigma(2, PX_SIGMA))
    # Weak prior anchoring each landmark to its triangulated init; regularizes far/low-
    # parallax points that would make ISAM2 indeterminate, negligible next to the pixel
    # factors.
    LM_PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)

    on_progress(0.0, "Building landmark tracks...")
    # Fed one keyframe at a time below (node 0 here, node j+1 in _add_keyframe_to_graph).
    # `tracks` aliases the builder's dict.
    track_builder = _LandmarkTrackBuilder(
        data, feature_detection_result, stereo_matching_result, optical_flow_result,
        DEPTH_MIN, DEPTH_MAX)
    tracks = track_builder.tracks

    # No landmark-starvation keyframe drop: it was non-causal (track maturity isn't known
    # until later keyframes) and FALLBACK_POSE_NOISE already covers a node with no
    # constraints. Removing it improved or left mean ATE flat on MH_02, MH_04, MH_05 and
    # V2_03.

    # Manual extra_loop_closures always override auto-detection; resolved to nodes up front.
    did_auto_detect_loop_closures = enable_loop_closure and extra_loop_closures is None
    loop_closures_by_node: dict[int, list[_LoopClosureEdge]] = {}
    if extra_loop_closures:
        kf_arr = np.array(keyframe_indices)
        for factor in extra_loop_closures:
            from_node = int(np.argmin(np.abs(kf_arr - factor.from_frame)))
            to_node = int(np.argmin(np.abs(kf_arr - factor.to_frame)))
            loop_closures_by_node.setdefault(to_node, []).append(_LoopClosureEdge(
                from_node, to_node, factor.body_relative_pose, factor.rot_sigma, factor.trans_sigma,
                factor.noise_mode))

    # Auto-detection is fed one keyframe at a time from inside the ISAM2 loop below.
    loop_detector = (
        _LoopClosureDetector(data, feature_detection_result, stereo_matching_result)
        if did_auto_detect_loop_closures else None)

    def _add_loop_closure_edges(factors: gtsam.NonlinearFactorGraph, edges: list[_LoopClosureEdge]) -> None:
        for edge in edges:
            delta = _to_gtsam_pose3(edge.body_relative_pose)
            base_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([edge.rot_sigma] * 3 + [edge.trans_sigma] * 3))
            if edge.noise_mode == "huber":
                # Linear loss beyond ~1.35 sigma caps a correct closure's pull
                # after a long blackout (~30% of gaussian's correction). Option
                # only; auto-detection uses gaussian.
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise)
            elif edge.noise_mode == "dcs":
                # c must match this factor's 6 degrees of freedom (a consistent
                # residual has chi-squared ~6); c=1 down-weighted already-correct
                # closures to ~0.29.
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(6.0), base_noise)  # type: ignore[attr-defined]
            else:
                loop_noise = base_noise
            factors.add(gtsam.BetweenFactorPose3(X(edge.from_node), X(edge.to_node), delta, loop_noise))

    # node -> track ids observed there, filled as each node is added.
    nodes_to_tracks: dict[int, list[int]] = {jj: [] for jj in range(K)}
    node_seen: list[set[int]] = [set() for _ in range(K)]

    def _register_node_tracks(node: int, touched: list[int]) -> None:
        nodes_to_tracks[node].extend(touched)
        node_seen[node].update(touched)

    _register_node_tracks(0, track_builder.add_keyframe(0, keyframe_indices[0]))
    if loop_detector is not None:
        loop_detector.add_keyframe(0, keyframe_indices[0])  # never closes a cluster on its own

    inserted_landmarks: set[int] = set()
    added_obs: set[tuple[int, int]] = set()
    n_proj_factors = 0

    def _add_obs_factors(factors: gtsam.NonlinearFactorGraph, tid: int, node: int,
                         uv0: np.ndarray, uv1: np.ndarray) -> None:
        nonlocal n_proj_factors
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv0, PX_NOISE, X(node), L(tid), cam0_K, False, False, cam0_pose))
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv1, PX_NOISE, X(node), L(tid), cam1_K, False, False, cam1_pose))
        added_obs.add((tid, node))
        n_proj_factors += 2

    def _process_node_landmarks(jj: int, est: gtsam.Values,
                                factors: gtsam.NonlinearFactorGraph,
                                values: gtsam.Values) -> int:
        """Add reprojection factors for landmarks observed at node jj (X(jj) already staged).
        Returns how many observations were staged at jj, so the caller can tell whether the new pose
        got a structure constraint.
        """
        n_at_node = 0
        for tid in nodes_to_tracks[jj]:
            obs = tracks[tid]
            if tid in inserted_landmarks:
                if (tid, jj) not in added_obs:
                    o = next(o for o in obs if o.node == jj)
                    _add_obs_factors(factors, tid, jj, o.uv0, o.uv1)
                    n_at_node += 1
                continue
            avail = [o for o in obs if o.node <= jj]
            if len(avail) < MIN_TRACK_LEN:
                continue
            # Initialize the landmark in the nav frame from the first observation's stereo depth.
            first = obs[0]
            T_G_cam0 = est.atPose3(X(first.node)).matrix() @ body_T_cam0
            p_world = (T_G_cam0 @ np.append(first.point_cam0, 1.0))[:3]
            values.insert(L(tid), gtsam.Point3(*p_world))
            factors.add(gtsam.PriorFactorPoint3(L(tid), gtsam.Point3(*p_world), LM_PRIOR_NOISE))
            inserted_landmarks.add(tid)
            for o in avail:
                _add_obs_factors(factors, tid, o.node, o.uv0, o.uv1)
                if o.node == jj:
                    n_at_node += 1
        return n_at_node

    isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

    # Gravity-align the nav frame: rotate body-0 so its measured gravity (the accelerometer
    # reads "up") lands on -gravity.
    R_G_body0 = _align_vectors(gravity_in_body, -gravity)
    pose0 = gtsam.Pose3(gtsam.Rot3(R_G_body0), gtsam.Point3(0.0, 0.0, 0.0))

    f0, v0 = gtsam.NonlinearFactorGraph(), gtsam.Values()
    f0.add(gtsam.PriorFactorPose3(X(0), pose0, PRIOR_POSE_NOISE))
    f0.add(gtsam.PriorFactorVector(V(0), np.zeros(3), PRIOR_VEL_NOISE))
    f0.add(gtsam.PriorFactorConstantBias(B(0), gtsam.imuBias.ConstantBias(), PRIOR_BIAS_NOISE))
    v0.insert(X(0), pose0)
    v0.insert(V(0), np.zeros(3))
    v0.insert(B(0), gtsam.imuBias.ConstantBias())
    isam2.update(f0, v0)

    def _add_keyframe_to_graph(j: int) -> None:
        """Add keyframe j+1: an IMU factor from the preintegrated window since j, a chained-PnP
        between-factor when landmark covisibility is weak, reprojection factors for its landmarks,
        and any loop-closure factors; then commit the update.
        """
        est = isam2.calculateEstimate()
        pose_i = est.atPose3(X(j))
        vel_i  = est.atVector(V(j))
        bias_i = est.atConstantBias(B(j))

        kf_i, kf_next = keyframe_indices[j], keyframe_indices[j + 1]

        # Extend landmark tracks to this node before anything below reads node_seen[j+1] or
        # tracks[tid] for it (strong_covis just below, then _process_node_landmarks).
        _register_node_tracks(j + 1, track_builder.add_keyframe(j + 1, keyframe_indices[j + 1]))
        # Search for a revisit at this keyframe. Returned edges are from a cluster that
        # just closed; their to_node may be earlier than j+1.
        new_loop_edges = loop_detector.add_keyframe(j + 1, keyframe_indices[j + 1]) if loop_detector is not None else []

        new_factors, new_values = gtsam.NonlinearFactorGraph(), gtsam.Values()

        pim = gtsam.PreintegratedImuMeasurements(imu_params, bias_i)
        window = np.where(
            (imu_timestamps_ns >= cam_timestamps_ns[kf_i]) &
            (imu_timestamps_ns <  cam_timestamps_ns[kf_next])
        )[0]
        for k in window:
            dt = (float(imu_timestamps_ns[k + 1] - imu_timestamps_ns[k]) * 1e-9
                  if k + 1 < len(imu_timestamps_ns) else 1.0 / data.imu0_rate_hz)
            pim.integrateMeasurement(imu_lin_accs[k], imu_ang_vels[k], dt)

        new_factors.add(gtsam.ImuFactor(X(j), V(j), X(j + 1), V(j + 1), B(j), pim))
        # Bias random-walk over this interval's duration: sigma = density * sqrt(dt) per axis,
        # ordered [accel(3), gyro(3)] to match imuBias.ConstantBias's tangent.
        dt_kf = float(cam_timestamps_ns[kf_next] - cam_timestamps_ns[kf_i]) * 1e-9
        bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [ACCEL_BIAS_RW * np.sqrt(dt_kf)] * 3 + [GYRO_BIAS_RW * np.sqrt(dt_kf)] * 3))
        new_factors.add(gtsam.BetweenFactorConstantBias(
            B(j), B(j + 1), gtsam.imuBias.ConstantBias(), bias_between_noise))

        nav_j     = pim.predict(gtsam.NavState(pose_i, vel_i), bias_i)
        pose_init = nav_j.pose()
        vel_init  = nav_j.velocity()

        # Chain adjacent-frame PnP steps for the keyframe-to-keyframe constraint (a direct
        # match across the gap would have little overlap); on any failure drop it and let
        # the IMU carry the interval. PnP is only a fallback: when the pair already shares
        # enough inserted landmarks, their reprojection factors constrain the pose and PnP
        # would double-count the pixels. Only inserted (mature) landmarks seen at both
        # ends count.
        strong_covis = sum(
            1 for tid in (node_seen[j] & node_seen[j + 1]) if tid in inserted_landmarks)
        pnp_added = False
        if strong_covis < PNP_FALLBACK_COVIS:
            pnp_cam0 = np.eye(4)
            pnp_ok = True
            for f in range(kf_i, kf_next):
                sm_f = stereo_matching_result.frames[f]
                fd_f, fd_f1 = feature_detection_result.frames[f], feature_detection_result.frames[f + 1]
                try:
                    rvec, tvec, _, _ = _run_pnp_step(
                        data, sm_f.points_3d, sm_f.matches,
                        fd_f.cam0_descriptors, fd_f1.cam0_keypoints, fd_f1.cam0_descriptors,
                    )
                except Exception:
                    pnp_ok = False
                    break
                pnp_cam0 = pnp_cam0 @ _rvec_tvec_to_transform(rvec, tvec)

            if pnp_ok:
                pnp_body = body_T_cam0 @ pnp_cam0 @ cam0_T_body
                pnp_delta = _to_gtsam_pose3(pnp_body)
                pnp_noise = gtsam.noiseModel.Diagonal.Sigmas(PNP_STEP_SIGMAS * np.sqrt(kf_next - kf_i))
                new_factors.add(gtsam.BetweenFactorPose3(X(j), X(j + 1), pnp_delta, pnp_noise))
                pose_init = pose_i.compose(pnp_delta)
                pnp_added = True

        new_values.insert(X(j + 1), pose_init)
        new_values.insert(V(j + 1), vel_init)
        new_values.insert(B(j + 1), bias_i)
        # Landmark inits use poses from `est`, which all predate node j+1.
        n_proj_at_next = _process_node_landmarks(j + 1, est, new_factors, new_values)
        # Guard: a keyframe with neither PnP nor reprojection factors hangs off its IMU
        # factor alone (ISAM2 indeterminate); anchor it with a weak prior.
        if not pnp_added and n_proj_at_next == 0:
            new_factors.add(gtsam.PriorFactorPose3(X(j + 1), pose_init, FALLBACK_POSE_NOISE))

        # Pre-resolved manual edges targeting exactly this node, plus whatever auto-detected edges
        # just became ready above (which may target this node or an earlier one).
        _add_loop_closure_edges(new_factors, loop_closures_by_node.pop(j + 1, []))
        _add_loop_closure_edges(new_factors, new_loop_edges)

        isam2.update(new_factors, new_values)

    for j in range(K - 1):
        on_progress(0.10 + (j / (K - 1)) * (0.92 - 0.10), "Optimizing (ISAM2)...")
        _add_keyframe_to_graph(j)

    if loop_detector is not None:
        # A cluster is only closed by a later candidate, so the last one needs this flush.
        final_edges = loop_detector.flush()
        if final_edges:
            tail_factors = gtsam.NonlinearFactorGraph()
            _add_loop_closure_edges(tail_factors, final_edges)
            isam2.update(tail_factors, gtsam.Values())

    print(f"landmarks: {len(inserted_landmarks)}/{len(tracks)} tracks used, "
          f"{n_proj_factors} reprojection factors")
    on_progress(0.92, "Computing metrics...")
    final = isam2.calculateEstimate()
    poses = [final.atPose3(X(j)).matrix() for j in range(K)]
    velocities = [final.atVector(V(j)) for j in range(K)]
    biases = [final.atConstantBias(B(j)).vector() for j in range(K)]  # [accel(3), gyro(3)]

    # Per-keyframe reprojection RMSE and landmark count: reproject each landmark into every
    # keyframe that saw it, through both cameras.
    inv_Twc0 = [np.linalg.inv(pm @ body_T_cam0) for pm in poses]
    inv_Twc1 = [np.linalg.inv(pm @ data.cam1_extrinsics) for pm in poses]
    sq_px = np.zeros(K)
    n_px = np.zeros(K)
    n_lm = np.zeros(K)
    for tid in inserted_landmarks:
        p = np.append(np.asarray(final.atPoint3(L(tid))), 1.0)
        for o in tracks[tid]:
            seen = False
            for intrin, inv_Twc, uv in (
                (data.cam0_intrinsics, inv_Twc0, o.uv0),
                (data.cam1_intrinsics, inv_Twc1, o.uv1),
            ):
                pc = inv_Twc[o.node] @ p
                if pc[2] <= 1e-6:
                    continue
                u = intrin.fx * pc[0] / pc[2] + intrin.cx
                v = intrin.fy * pc[1] / pc[2] + intrin.cy
                sq_px[o.node] += (u - uv[0]) ** 2 + (v - uv[1]) ** 2
                n_px[o.node] += 1
                seen = True
            if seen:
                n_lm[o.node] += 1
    reprojection_rmse = np.where(n_px > 0, np.sqrt(sq_px / np.maximum(n_px, 1)), np.nan)

    if debug_out is not None:
        debug_out['isam2'] = isam2

    return poses, velocities, biases, reprojection_rmse, n_lm, keyframe_indices


class _GtsamBuilder:
    """Incremental _run_gtsam: feed it one new keyframe at a time via add_keyframe(frame_idx).

    Per keyframe: IMU factor, chained-PnP fallback, landmark reprojection factors, loop-closure
    factors. Requires every frame's fd/sm from the previous keyframe through frame_idx (the PnP
    fallback steps through intermediate frames); later frames are never touched. Does not support
    _run_gtsam's extra_loop_closures/debug_out test hooks.
    """

    def __init__(
        self, data: EuRoCMAVData,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
        optical_flow_result: OpticalFlowResult,
        imu_samples: list[ImuSample],
        gravity: np.ndarray,
        first_keyframe_frame: int,
        enable_loop_closure: bool = False,
    ) -> None:
        self._data = data
        self._fd_result = feature_detection_result
        self._sm_result = stereo_matching_result
        self._of_result = optical_flow_result

        imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
        # See _run_gtsam's identical guard for why this must be read-only.
        imu_timestamps_ns.flags.writeable = False
        self._imu_timestamps_ns = imu_timestamps_ns
        self._imu_lin_accs = np.array([s.linear_acceleration for s in imu_samples])
        self._imu_ang_vels = np.array([s.angular_velocity for s in imu_samples])

        self._body_T_cam0 = data.cam0_extrinsics
        self._cam0_T_body = np.linalg.inv(self._body_T_cam0)

        grav_win = max(1, int(data.imu0_rate_hz * 0.5))
        first_cam_ts = stereo_matching_result.frames[first_keyframe_frame].timestamp_ns
        i0 = int(np.argmin(np.abs(imu_timestamps_ns - first_cam_ts)))
        gravity_in_body = self._imu_lin_accs[i0:i0 + grav_win].mean(axis=0)

        self._X = lambda i: gtsam.symbol('x', i)
        self._V = lambda i: gtsam.symbol('v', i)
        self._B = lambda i: gtsam.symbol('b', i)
        self._L = lambda i: gtsam.symbol('l', i)

        self._imu_params = gtsam.PreintegrationParams(gravity)
        self._imu_params.setGyroscopeCovariance(np.eye(3) * 1e-4)
        self._imu_params.setAccelerometerCovariance(np.eye(3) * 1e-3)
        self._imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)

        self._PRIOR_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self._PRIOR_VEL_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self._PRIOR_BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        self._PNP_STEP_SIGMAS = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
        self._GYRO_BIAS_RW = 1.9393e-05
        self._ACCEL_BIAS_RW = 3.0000e-03
        self._FALLBACK_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)

        self._MIN_TRACK_LEN = 3
        self._PNP_FALLBACK_COVIS = 15
        PX_SIGMA = 1.5
        DEPTH_MIN, DEPTH_MAX = 0.3, 40.0
        self._cam0_K = gtsam.Cal3_S2(data.cam0_intrinsics.fx, data.cam0_intrinsics.fy, 0.0,
                                     data.cam0_intrinsics.cx, data.cam0_intrinsics.cy)
        self._cam1_K = gtsam.Cal3_S2(data.cam1_intrinsics.fx, data.cam1_intrinsics.fy, 0.0,
                                     data.cam1_intrinsics.cx, data.cam1_intrinsics.cy)
        self._cam0_pose = gtsam.Pose3(self._body_T_cam0)
        self._cam1_pose = gtsam.Pose3(data.cam1_extrinsics)
        self._PX_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Isotropic.Sigma(2, PX_SIGMA))
        self._LM_PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)

        self._track_builder = _LandmarkTrackBuilder(
            data, feature_detection_result, stereo_matching_result, optical_flow_result,
            DEPTH_MIN, DEPTH_MAX)
        self._tracks = self._track_builder.tracks
        self._loop_detector = (
            _LoopClosureDetector(data, feature_detection_result, stereo_matching_result)
            if enable_loop_closure else None)

        # node -> track ids observed there; and per-interval covisibility for the PnP gate. Keyed
        # by node index directly (not pre-sized by K, which isn't known until the run ends).
        self._nodes_to_tracks: dict[int, list[int]] = defaultdict(list)
        self._node_seen: dict[int, set[int]] = defaultdict(set)
        self._inserted_landmarks: set[int] = set()
        self._added_obs: set[tuple[int, int]] = set()
        self._n_proj_factors = 0
        # Cached per-node reprojection contributions -- see current_estimate's docstring for why
        # these are cached (and selectively refreshed) rather than rederived in full every call.
        self._sq_px = np.zeros(0)
        self._n_px = np.zeros(0)
        self._n_lm = np.zeros(0)

        self._isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

        R_G_body0 = _align_vectors(gravity_in_body, -gravity)
        pose0 = gtsam.Pose3(gtsam.Rot3(R_G_body0), gtsam.Point3(0.0, 0.0, 0.0))
        f0, v0 = gtsam.NonlinearFactorGraph(), gtsam.Values()
        f0.add(gtsam.PriorFactorPose3(self._X(0), pose0, self._PRIOR_POSE_NOISE))
        f0.add(gtsam.PriorFactorVector(self._V(0), np.zeros(3), self._PRIOR_VEL_NOISE))
        f0.add(gtsam.PriorFactorConstantBias(self._B(0), gtsam.imuBias.ConstantBias(), self._PRIOR_BIAS_NOISE))
        v0.insert(self._X(0), pose0)
        v0.insert(self._V(0), np.zeros(3))
        v0.insert(self._B(0), gtsam.imuBias.ConstantBias())
        self._isam2.update(f0, v0)

        self._register_node_tracks(0, self._track_builder.add_keyframe(0, first_keyframe_frame))
        if self._loop_detector is not None:
            self._loop_detector.add_keyframe(0, first_keyframe_frame)  # never closes a cluster on its own

        self._j = 0  # index of the most recently added node
        self._prev_frame = first_keyframe_frame  # its original frame index

    def _register_node_tracks(self, node: int, touched: list[int]) -> None:
        self._nodes_to_tracks[node].extend(touched)
        self._node_seen[node].update(touched)

    def _add_obs_factors(self, factors: gtsam.NonlinearFactorGraph, tid: int, node: int,
                         uv0: np.ndarray, uv1: np.ndarray) -> None:
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv0, self._PX_NOISE, self._X(node), self._L(tid), self._cam0_K, False, False, self._cam0_pose))
        factors.add(gtsam.GenericProjectionFactorCal3_S2(
            uv1, self._PX_NOISE, self._X(node), self._L(tid), self._cam1_K, False, False, self._cam1_pose))
        self._added_obs.add((tid, node))
        self._n_proj_factors += 2

    def _process_node_landmarks(self, jj: int, est: gtsam.Values,
                                factors: gtsam.NonlinearFactorGraph, values: gtsam.Values) -> int:
        n_at_node = 0
        for tid in self._nodes_to_tracks[jj]:
            obs = self._tracks[tid]
            if tid in self._inserted_landmarks:
                if (tid, jj) not in self._added_obs:
                    o = next(o for o in obs if o.node == jj)
                    self._add_obs_factors(factors, tid, jj, o.uv0, o.uv1)
                    n_at_node += 1
                continue
            avail = [o for o in obs if o.node <= jj]
            if len(avail) < self._MIN_TRACK_LEN:
                continue
            first = obs[0]
            T_G_cam0 = est.atPose3(self._X(first.node)).matrix() @ self._body_T_cam0
            p_world = (T_G_cam0 @ np.append(first.point_cam0, 1.0))[:3]
            values.insert(self._L(tid), gtsam.Point3(*p_world))
            factors.add(gtsam.PriorFactorPoint3(self._L(tid), gtsam.Point3(*p_world), self._LM_PRIOR_NOISE))
            self._inserted_landmarks.add(tid)
            for o in avail:
                self._add_obs_factors(factors, tid, o.node, o.uv0, o.uv1)
                if o.node == jj:
                    n_at_node += 1
        return n_at_node

    def _add_loop_closure_edges(self, factors: gtsam.NonlinearFactorGraph, edges: list[_LoopClosureEdge]) -> None:
        for edge in edges:
            delta = _to_gtsam_pose3(edge.body_relative_pose)
            base_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([edge.rot_sigma] * 3 + [edge.trans_sigma] * 3))
            if edge.noise_mode == "huber":
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise)
            elif edge.noise_mode == "dcs":
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(6.0), base_noise)  # type: ignore[attr-defined]
            else:
                loop_noise = base_noise
            factors.add(gtsam.BetweenFactorPose3(self._X(edge.from_node), self._X(edge.to_node), delta, loop_noise))

    def add_keyframe(self, frame_idx: int) -> None:
        """Add a node for `frame_idx`, linked to the previous node by an IMU factor and, when
        landmark covisibility is weak, a chained-PnP fallback (see _run_gtsam's
        _add_keyframe_to_graph).
        """
        j = self._j
        est = self._isam2.calculateEstimate()
        pose_i = est.atPose3(self._X(j))
        vel_i = est.atVector(self._V(j))
        bias_i = est.atConstantBias(self._B(j))

        kf_i, kf_next = self._prev_frame, frame_idx
        kf_i_ts = self._sm_result.frames[kf_i].timestamp_ns
        kf_next_ts = self._sm_result.frames[kf_next].timestamp_ns

        self._register_node_tracks(j + 1, self._track_builder.add_keyframe(j + 1, frame_idx))
        new_loop_edges = (
            self._loop_detector.add_keyframe(j + 1, frame_idx) if self._loop_detector is not None else [])

        new_factors, new_values = gtsam.NonlinearFactorGraph(), gtsam.Values()

        pim = gtsam.PreintegratedImuMeasurements(self._imu_params, bias_i)
        window = np.where(
            (self._imu_timestamps_ns >= kf_i_ts) & (self._imu_timestamps_ns < kf_next_ts))[0]
        for k in window:
            dt = (float(self._imu_timestamps_ns[k + 1] - self._imu_timestamps_ns[k]) * 1e-9
                  if k + 1 < len(self._imu_timestamps_ns) else 1.0 / self._data.imu0_rate_hz)
            pim.integrateMeasurement(self._imu_lin_accs[k], self._imu_ang_vels[k], dt)

        new_factors.add(gtsam.ImuFactor(self._X(j), self._V(j), self._X(j + 1), self._V(j + 1), self._B(j), pim))
        dt_kf = float(kf_next_ts - kf_i_ts) * 1e-9
        bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [self._ACCEL_BIAS_RW * np.sqrt(dt_kf)] * 3 + [self._GYRO_BIAS_RW * np.sqrt(dt_kf)] * 3))
        new_factors.add(gtsam.BetweenFactorConstantBias(
            self._B(j), self._B(j + 1), gtsam.imuBias.ConstantBias(), bias_between_noise))

        nav_j = pim.predict(gtsam.NavState(pose_i, vel_i), bias_i)
        pose_init = nav_j.pose()
        vel_init = nav_j.velocity()

        strong_covis = sum(
            1 for tid in (self._node_seen[j] & self._node_seen[j + 1]) if tid in self._inserted_landmarks)
        pnp_added = False
        if strong_covis < self._PNP_FALLBACK_COVIS:
            pnp_cam0 = np.eye(4)
            pnp_ok = True
            for f in range(kf_i, kf_next):
                sm_f = self._sm_result.frames[f]
                fd_f, fd_f1 = self._fd_result.frames[f], self._fd_result.frames[f + 1]
                try:
                    rvec, tvec, _, _ = _run_pnp_step(
                        self._data, sm_f.points_3d, sm_f.matches,
                        fd_f.cam0_descriptors, fd_f1.cam0_keypoints, fd_f1.cam0_descriptors,
                    )
                except Exception:
                    pnp_ok = False
                    break
                pnp_cam0 = pnp_cam0 @ _rvec_tvec_to_transform(rvec, tvec)

            if pnp_ok:
                pnp_body = self._body_T_cam0 @ pnp_cam0 @ self._cam0_T_body
                pnp_delta = _to_gtsam_pose3(pnp_body)
                pnp_noise = gtsam.noiseModel.Diagonal.Sigmas(self._PNP_STEP_SIGMAS * np.sqrt(kf_next - kf_i))
                new_factors.add(gtsam.BetweenFactorPose3(self._X(j), self._X(j + 1), pnp_delta, pnp_noise))
                pose_init = pose_i.compose(pnp_delta)
                pnp_added = True

        new_values.insert(self._X(j + 1), pose_init)
        new_values.insert(self._V(j + 1), vel_init)
        new_values.insert(self._B(j + 1), bias_i)
        n_proj_at_next = self._process_node_landmarks(j + 1, est, new_factors, new_values)
        if not pnp_added and n_proj_at_next == 0:
            new_factors.add(gtsam.PriorFactorPose3(self._X(j + 1), pose_init, self._FALLBACK_POSE_NOISE))

        self._add_loop_closure_edges(new_factors, new_loop_edges)

        self._isam2.update(new_factors, new_values)
        self._j += 1
        self._prev_frame = frame_idx

    def _reprojection_for_node(
        self, node: int, final: gtsam.Values, inv_Twc0: list[np.ndarray], inv_Twc1: list[np.ndarray],
    ) -> tuple[float, float, float]:
        """sq_px, n_px and n_lm contributed by every inserted landmark observed at exactly this
        node, grouped by node via self._nodes_to_tracks.
        """
        sq_px = 0.0
        n_px = 0.0
        n_lm = 0.0
        for tid in self._nodes_to_tracks[node]:
            if tid not in self._inserted_landmarks:
                continue
            obs = next((o for o in self._tracks[tid] if o.node == node), None)
            if obs is None:
                continue
            p = np.append(np.asarray(final.atPoint3(self._L(tid))), 1.0)
            seen = False
            for intrin, inv_Twc, uv in (
                (self._data.cam0_intrinsics, inv_Twc0, obs.uv0),
                (self._data.cam1_intrinsics, inv_Twc1, obs.uv1),
            ):
                pc = inv_Twc[node] @ p
                if pc[2] <= 1e-6:
                    continue
                u = intrin.fx * pc[0] / pc[2] + intrin.cx
                v = intrin.fy * pc[1] / pc[2] + intrin.cy
                sq_px += (u - uv[0]) ** 2 + (v - uv[1]) ** 2
                n_px += 1
                seen = True
            if seen:
                n_lm += 1.0
        return sq_px, n_px, n_lm

    def current_estimate(
        self, recompute_window: Optional[int] = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """Poses, velocities, biases and reprojection metrics for every node added so far. Safe to
        call after any keyframe (it doesn't flush a still-open loop-closure cluster).

        recompute_window: if given, only the last N nodes' reprojection metrics are recomputed and
        earlier nodes reuse cached values (ISAM2 keeps revising past nodes, so cached values can go
        stale). 0 skips the loop entirely (cached values only, NaN/0 if never computed): used for
        live previews, since only one UI plot reads these. None recomputes every node exactly;
        finalize() always uses it.
        """
        K = self._j + 1
        final = self._isam2.calculateEstimate()
        poses = [final.atPose3(self._X(j)).matrix() for j in range(K)]
        velocities = [final.atVector(self._V(j)) for j in range(K)]
        biases = [final.atConstantBias(self._B(j)).vector() for j in range(K)]

        inv_Twc0 = [np.linalg.inv(pm @ self._body_T_cam0) for pm in poses]
        inv_Twc1 = [np.linalg.inv(pm @ self._data.cam1_extrinsics) for pm in poses]

        if len(self._sq_px) < K:
            pad = K - len(self._sq_px)
            self._sq_px = np.concatenate([self._sq_px, np.zeros(pad)])
            self._n_px = np.concatenate([self._n_px, np.zeros(pad)])
            self._n_lm = np.concatenate([self._n_lm, np.zeros(pad)])

        recompute_from = 0 if recompute_window is None else max(0, K - recompute_window)
        for node in range(recompute_from, K):
            sq, n, lm = self._reprojection_for_node(node, final, inv_Twc0, inv_Twc1)
            self._sq_px[node] = sq
            self._n_px[node] = n
            self._n_lm[node] = lm

        n_px_K = self._n_px[:K]
        reprojection_rmse = np.where(n_px_K > 0, np.sqrt(self._sq_px[:K] / np.maximum(n_px_K, 1)), np.nan)

        return poses, velocities, biases, reprojection_rmse, self._n_lm[:K].copy()

    def finalize(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        """Call once after the last keyframe: flushes any open loop-closure cluster, then returns
        current_estimate().
        """
        if self._loop_detector is not None:
            final_edges = self._loop_detector.flush()
            if final_edges:
                tail_factors = gtsam.NonlinearFactorGraph()
                self._add_loop_closure_edges(tail_factors, final_edges)
                self._isam2.update(tail_factors, gtsam.Values())

        print(f"landmarks: {len(self._inserted_landmarks)}/{len(self._tracks)} tracks used, "
              f"{self._n_proj_factors} reprojection factors")
        # Every node, exactly -- see current_estimate's docstring on recompute_window=None.
        return self.current_estimate(recompute_window=None)

    def landmark_positions(self) -> np.ndarray:
        """World-frame position of every inserted landmark, for the sparse-map layer. Call once
        after finalize(): it does a full calculateEstimate() plus a pass over the landmark set, too
        costly per live keyframe.
        """
        if not self._inserted_landmarks:
            return np.zeros((0, 3))
        final = self._isam2.calculateEstimate()
        return np.array([final.atPoint3(self._L(tid)) for tid in self._inserted_landmarks])


def _get_full_gt_arrays(data: EuRoCMAVData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth timestamps/positions/rotations for the entire dataset, computed once per run and
    shared across _get_gtsam_result calls. Left unwindowed because the nearest GT sample to a
    boundary keyframe can fall just outside the window.
    """
    gt_timestamps_ns = np.array([s.timestamp_ns for s in data.ground_truth_samples])
    gt_positions = np.array([s.position for s in data.ground_truth_samples])
    gt_rotation_matrices = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in data.ground_truth_samples])
    return gt_timestamps_ns, gt_positions, gt_rotation_matrices


def _get_gtsam_result(
    data: EuRoCMAVData,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    gt_timestamps_ns: np.ndarray,
    gt_positions: np.ndarray,
    gt_rotation_matrices_all: np.ndarray,
    keyframe_indices: list[int],
    poses: list[np.ndarray],
    velocities: list[np.ndarray],
    biases: list[np.ndarray],
    reprojection_rmse: np.ndarray,
    landmark_counts: np.ndarray,
    landmark_positions: np.ndarray,
) -> SlamGtsamResult:
    """Ground-truth alignment and ATE/RPE metrics for an already-computed GTSAM trajectory (see
    _compute). The gt_* arrays come from _get_full_gt_arrays.
    """
    K = len(poses)

    kf_frames = [stereo_matching_result.frames[k] for k in keyframe_indices]
    cam_timestamps_ns = np.array([f.timestamp_ns for f in kf_frames])
    # Anchor to GT at the first GT sample inside the window (after start_s), consistent with
    # how the GT/PnP series are windowed.
    first_gt_sample = next(s for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns)
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt_sample.timestamp_ns))

    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt_sample.quaternion)
    world_T_body_first[:3, 3] = first_gt_sample.position

    T_comp = world_T_body_first @ np.linalg.inv(poses[closest_cam_index])
    world_T_body_poses = np.array([T_comp @ T for T in poses])

    times = np.array([_elapsed_s(f.timestamp_ns, first_timestamp_ns) for f in kf_frames])

    # Per-keyframe position error [m] vs the nearest GT sample; poses are single-point-
    # anchored to GT (T_comp), as overlaid in the view.
    nearest_gt = _nearest_sorted_indices(gt_timestamps_ns, cam_timestamps_ns)
    position_errors = np.linalg.norm(world_T_body_poses[:, :3, 3] - gt_positions[nearest_gt], axis=1)

    # ATE/RPE use the raw poses (before T_comp): T_comp comes from one GT quaternion and can
    # tilt "up" away from gravity, which _yaw_translation_align's assumption depends on.
    raw_positions = np.array([p[:3, 3] for p in poses])
    raw_rotation_matrices = np.array([p[:3, :3] for p in poses])
    gt_positions_at_kf = gt_positions[nearest_gt]
    gt_rotation_matrices_at_kf = gt_rotation_matrices_all[nearest_gt]
    ate_position_errors, ate_rotation_errors = _compute_ate(
        raw_positions, raw_rotation_matrices, gt_positions_at_kf, gt_rotation_matrices_at_kf)
    rpe_translation_errors, rpe_rotation_errors = _compute_rpe(
        times, raw_positions, raw_rotation_matrices, gt_positions_at_kf, gt_rotation_matrices_at_kf, RPE_DELTA_S)

    angular_velocities = []
    linear_accelerations = []
    velocities_np = np.array(velocities)
    for i in range(K - 1):
        dt = times[i + 1] - times[i]
        rotation_matrix = world_T_body_poses[i, :3, :3].T @ world_T_body_poses[i + 1, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        angular_velocities.append(rotation_vector.flatten() / dt)

        acc_world = (velocities_np[i + 1] - velocities_np[i]) / dt
        acc_body = world_T_body_poses[i, :3, :3].T @ acc_world
        linear_accelerations.append(acc_body)

    rotation_matrices = world_T_body_poses[:, :3, :3]
    return SlamGtsamResult(
        times=times,
        positions=world_T_body_poses[:, :3, 3],
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        velocities=velocities_np,
        biases=np.array(biases),
        position_errors=position_errors,
        ate_position_errors=ate_position_errors,
        ate_rotation_errors=ate_rotation_errors,
        rpe_translation_errors=rpe_translation_errors,
        rpe_rotation_errors=rpe_rotation_errors,
        reprojection_rmse=reprojection_rmse,
        landmark_counts=landmark_counts,
        landmark_positions=landmark_positions,
        angular_velocity_times=times[:-1],
        angular_velocities=np.array(angular_velocities),
        linear_accelerations=np.array(linear_accelerations),
    )


def _get_extra_result(
    data: EuRoCMAVData,
    gt_result: SlamGroundTruthResult,
    imu_result: SlamImuResult,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
    gravity: np.ndarray,
) -> SlamExtraResult:

    gt_samples = [s for s in data.ground_truth_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    gt_timestamps_ns = np.array([s.timestamp_ns for s in gt_samples])
    imu_timestamps_ns = np.array([s.timestamp_ns for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns])
    closest_gt_indices = _nearest_sorted_indices(gt_timestamps_ns, imu_timestamps_ns)
    linear_accelerations_in_world = np.array([
        gt_result.rotation_matrices[idx] @ acc
        for idx, acc in zip(closest_gt_indices, imu_result.linear_accelerations)
    ])

    return SlamExtraResult(
        gravity=gravity,
        linear_accelerations_in_world=linear_accelerations_in_world,
    )


def _compute(
    data: EuRoCMAVData,
    start_s: float,
    duration_s: float,
    set_progress: Callable[[float, str], None],
    enable_loop_closure: bool = False,
    on_partial_result: Optional[Callable[[SlamResult], None]] = None,
    frontend_result: Optional[FrontendResult] = None,
) -> SlamResult:
    """Runs the whole SLAM stage as one incremental pass over [start_s, start_s + duration_s]: each
    frame is computed (FrontendFrameComputer), then immediately offered to keyframe selection
    (_KeyframeScanner) and, for any keyframe, the pose graph (_GtsamBuilder), before the next frame
    is touched.

    If on_partial_result is given, it's called after every keyframe with a full SlamResult
    reflecting only what's been processed so far, to render the trajectory as it grows.

    frontend_result: if given, its empty containers are the ones filled frame by frame, so the
    caller can show the frontend's output live without computing it twice.
    """
    first_timestamp_ns = data.cam_timestamps_ns[0]
    min_ts = first_timestamp_ns + int(start_s * 1e9)
    max_ts = min_ts + int(duration_s * 1e9)
    timestamps = [t for t in data.cam_timestamps_ns if min_ts <= t <= max_ts]
    N = len(timestamps)
    # The first/last frame that exists in the window mark its bounds (min_ts/max_ts may fall
    # between frames). Trimming GT/IMU/extra to them makes every series start at start_s.
    min_timestamp_ns = timestamps[0]
    max_timestamp_ns = timestamps[-1]

    gravity = np.array([0.0, 0.0, -9.81])

    # Progress: ground truth [0, 0.03], IMU [0.03, 0.06], frontend + keyframes + GTSAM [0.06,
    # 0.96], finishing [0.97, 1].
    set_progress(0.0, "Loading ground truth...")
    gt_result = _get_ground_truth_result(data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns)
    gt_timestamps_ns, gt_positions_all, gt_rotation_matrices_all = _get_full_gt_arrays(data)

    # Anchor the IMU-integrated orientation to GT at the first GT sample inside the window
    # (i.e. gt_result's first sample, which is gt_rotation_matrices[0]).
    set_progress(0.03, "Integrating IMU...")
    first_gt_timestamp_ns = next(
        s.timestamp_ns for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns
    )
    imu_result = _get_imu_result(
        data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns,
        gt_result.rotation_matrices, first_gt_timestamp_ns,
    )
    # Depends only on gt_result/imu_result/gravity, never on the growing estimate, so it is
    # computed once for the partial snapshots too.
    extra_result = _get_extra_result(data, gt_result, imu_result, min_timestamp_ns, max_timestamp_ns, gravity)

    # Each frame's fd/sm/of is appended to these growing containers, then immediately offered
    # to _KeyframeScanner and _GtsamBuilder. Both read earlier frames from the same
    # containers, so they are shared mutable state.
    set_progress(0.06, "Processing frames...")
    compute_t0 = time.monotonic()
    imu_samples = [s for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]

    frontend = FrontendFrameComputer(data)
    if frontend_result is None:
        frontend_result = FrontendResult(
            feature_detection=FeatureDetectionResult(frames=[], elapsed_s=0.0),
            stereo_matching=StereoMatchingResult(frames=[], elapsed_s=0.0),
            optical_flow=OpticalFlowResult(frames=[], elapsed_s=0.0),
            elapsed_s=0.0)
    feature_detection_result = frontend_result.feature_detection
    stereo_matching_result = frontend_result.stereo_matching
    optical_flow_result = frontend_result.optical_flow

    def _process_frame(ts: int) -> None:
        fd_frame, sm_frame, of_frame = frontend.add_frame(ts)
        feature_detection_result.frames.append(fd_frame)
        stereo_matching_result.frames.append(sm_frame)
        optical_flow_result.frames.append(of_frame)

    def _build_partial_result(up_to_frame: int) -> SlamResult:
        """A full SlamResult reflecting only what's processed through `up_to_frame`: gt/imu/extra
        are the final values (independent of the growing estimate); pnp and gtsam come from the
        scanner/builder's live accessors.
        """
        partial_pnp_poses = scanner.current_poses(up_to_frame)
        partial_pnp_result = _get_pnp_result(
            data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns, partial_pnp_poses)
        # recompute_window=0: reprojection_rmse/landmark_counts are live diagnostics only;
        # they read back NaN/0 until finalize() computes them.
        p_poses, p_vel, p_bias, p_rmse, p_lm = builder.current_estimate(recompute_window=0)
        # landmark_positions is likewise only populated at finalize().
        partial_gtsam_result = _get_gtsam_result(
            data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns,
            gt_timestamps_ns, gt_positions_all, gt_rotation_matrices_all,
            scanner.keyframes, p_poses, p_vel, p_bias, p_rmse, p_lm, np.zeros((0, 3)),
        )
        return SlamResult(
            gt=gt_result, imu=imu_result, pnp=partial_pnp_result, gtsam=partial_gtsam_result,
            extra=extra_result,
        )

    _process_frame(timestamps[0])
    scanner = _KeyframeScanner(data, feature_detection_result, stereo_matching_result, N)
    builder = _GtsamBuilder(
        data, feature_detection_result, stereo_matching_result, optical_flow_result,
        imu_samples, gravity, first_keyframe_frame=0, enable_loop_closure=enable_loop_closure)
    for i in range(1, N):
        set_progress(0.06 + (i / N) * (0.95 - 0.06), "Processing frames...")
        _process_frame(timestamps[i])
        for kf in scanner.add_frame(i):
            builder.add_keyframe(kf)
            if on_partial_result is not None:
                on_partial_result(_build_partial_result(i))
    keyframe_indices = scanner.keyframes
    pnp_poses = scanner.finalize()
    gtsam_poses, velocities, biases, reprojection_rmse, landmark_counts = builder.finalize()
    landmark_positions = builder.landmark_positions()
    # Frontend computation, scanning, and graph-building are now the same combined pass, so they
    # no longer have separately measurable costs -- both diagnostics report the same total.
    compute_elapsed = time.monotonic() - compute_t0
    for r in (feature_detection_result, stereo_matching_result, optical_flow_result, frontend_result):
        r.elapsed_s = compute_elapsed

    pnp_result = _get_pnp_result(
        data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns, pnp_poses)
    pnp_result.elapsed_time = compute_elapsed

    set_progress(0.96, "Aligning to ground truth...")
    gtsam_result = _get_gtsam_result(
        data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns,
        gt_timestamps_ns, gt_positions_all, gt_rotation_matrices_all,
        keyframe_indices, gtsam_poses, velocities, biases, reprojection_rmse, landmark_counts,
        landmark_positions,
    )
    gtsam_result.elapsed_time = compute_elapsed

    set_progress(0.97, "Finishing...")
    set_progress(1.0, "Done")
    return SlamResult(
        gt=gt_result,
        imu=imu_result,
        pnp=pnp_result,
        gtsam=gtsam_result,
        extra=extra_result,
    )


class _SolveCancelled(Exception):
    pass


class SlamSolver:
    """Runs the whole SLAM stage over [start_s, start_s + duration_s] from raw EuRoCMAVData alone.
    It computes the frontend data itself, incrementally (see _compute), and exposes it as
    frontend_result for the diagnostic tabs.
    """

    def __init__(
        self, data: EuRoCMAVData, start_s: float, duration_s: float,
        cancel_event: Optional[threading.Event] = None,
        # On by default: a clear win on one regression-check sequence, a smaller localized
        # regression on another. Detection is O(K^2) brute-force descriptor matching, so
        # frequent interactive re-runs on long sequences may want to opt out.
        enable_loop_closure: bool = True,
    ) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._enable_loop_closure = enable_loop_closure
        self.result: Optional[SlamResult] = None
        # The frontend's per-frame output, filled by _compute and complete once `loading`
        # is False. Frames are appended one list at a time, so a reader should only trust
        # indices below min(len(fd.frames), len(sm.frames), len(of.frames)).
        self.frontend_result = FrontendResult(
            feature_detection=FeatureDetectionResult(frames=[], elapsed_s=0.0),
            stereo_matching=StereoMatchingResult(frames=[], elapsed_s=0.0),
            optical_flow=OpticalFlowResult(frames=[], elapsed_s=0.0),
            elapsed_s=0.0)
        self.loading: bool = True
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.progress_label: str = ""

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        def set_progress(value: float, label: str) -> None:
            if self._cancel_event.is_set():
                raise _SolveCancelled()
            self.progress = value
            self.progress_label = label

        def set_partial_result(result: SlamResult) -> None:
            if self._cancel_event.is_set():
                raise _SolveCancelled()
            # Single-attribute assignment, as in set_progress: the render thread sees
            # either the old SlamResult or the new one, never a half-built one.
            self.result = result

        # Runs on the caller's background thread (see SlamViewModel.start), sharing this
        # process's memory: no spawn, no reimport, no pickling data across a process boundary.
        try:
            self.result = _compute(
                self._data, self._start_s, self._duration_s, set_progress,
                enable_loop_closure=self._enable_loop_closure, on_partial_result=set_partial_result,
                frontend_result=self.frontend_result)
        except _SolveCancelled:
            pass
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.loading = False
