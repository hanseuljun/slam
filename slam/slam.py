from dataclasses import dataclass
import threading
import time
import traceback
from typing import Callable, Optional

import cv2
import gtsam
import numpy as np

from slam.data import EuRoCMAVData, ImuSample
from slam.feature_detection import FeatureDetectionResult
from slam.imu_initialization import ImuInitializationResult
from slam.optical_flow import OpticalFlowFrame, OpticalFlowResult
from slam.stereo_matching import StereoMatchingResult
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
    # ATE: batch (whole-trajectory) yaw+translation alignment, then per-keyframe error. Yaw-only
    # (not full 3D rotation) because gravity makes roll/pitch observable for a VIO system; letting
    # the alignment absorb them would hide real error instead of exposing it. See Zhang &
    # Scaramuzza, "A Tutorial on Quantitative Trajectory Evaluation for VIO", IROS 2018.
    ate_position_errors: np.ndarray  # shape (K,) [m]
    ate_rotation_errors: np.ndarray  # shape (K,) [deg]
    # RPE: relative motion error over a fixed time window, alignment-free by construction (a
    # constant misalignment cancels out of a relative transform), so it isolates local/per-step
    # accuracy from the cumulative drift ATE reports. NaN where no keyframe falls within the
    # window of the trajectory's end.
    rpe_translation_errors: np.ndarray  # shape (K,) [m]
    rpe_rotation_errors: np.ndarray  # shape (K,) [deg]
    reprojection_rmse: np.ndarray  # per-keyframe landmark reprojection RMSE [px], shape (K,)
    landmark_counts: np.ndarray  # per-keyframe number of observed landmarks, shape (K,)
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
    """For each query, the index into sorted_reference whose value is closest -- same result as
    argmin(abs(sorted_reference[:, None] - queries[None, :]), axis=0), but via binary search
    instead of a dense (len(sorted_reference) x len(queries)) broadcast. That broadcast is O(n*m)
    in both time and memory: matching ~29k IMU samples against ~29k ground-truth samples over a
    145s sequence built an 836M-element array (~6.7GB, plus another same-sized array for the abs())
    for a single nearest-timestamp lookup. This is O((n + m) log n) time, O(m) memory.
    """
    idx = np.searchsorted(sorted_reference, queries, side="left")
    idx = np.clip(idx, 1, len(sorted_reference) - 1)
    left, right = sorted_reference[idx - 1], sorted_reference[idx]
    return idx - ((queries - left) <= (right - queries))


def _yaw_translation_align(est_positions: np.ndarray, gt_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rotation-about-z (yaw) + translation aligning est_positions to gt_positions.

    Restricted to yaw rather than a full 3D (Umeyama) rotation: both trajectories are already
    gravity-aligned (nav frame's z is "up"), and for a VIO system gravity makes roll/pitch
    observable, so a full-rotation alignment could quietly absorb a real roll/pitch error instead
    of reporting it. Closed form: for fixed yaw the optimal translation is the centroid offset;
    substituting that in reduces the remaining 1D optimization to maximizing
    A*cos(theta) + B*sin(theta), solved by theta = atan2(B, A).

    Only observable when the trajectory has real horizontal spread: fit purely from positions, so
    over a short or near-stationary window (e.g. EuRoC's static lead-in before takeoff) the
    horizontal signal is too small to pin theta down, and the resulting yaw -- and therefore
    ate_rotation_errors -- can be noisy or biased even though the fit is the true least-squares
    optimum for that window. Not a defect in the estimate; a property of evaluating too short/still
    a stretch. Prefer windows with a few meters of horizontal travel when reading ATE rotation.
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
    """Absolute Trajectory Error: batch-align the whole estimated trajectory to ground truth
    once (yaw + translation), then report per-keyframe position/rotation error after that single
    alignment -- comparable to how other VIO/SLAM systems' ATE is usually reported."""
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
    """Relative Pose Error: for each keyframe, compare the estimated relative motion over the
    next delta_s seconds to ground truth's relative motion over the same interval. Needs no
    alignment -- a constant misalignment cancels out of a relative (inv(T_i) @ T_j) transform --
    so unlike ATE this isolates local/per-step accuracy from cumulative drift."""
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
    times = np.array([(s.timestamp_ns - first_timestamp_ns) / 1e9 for s in samples])
    positions = np.array([s.position for s in samples])
    rotation_matrices = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in samples])

    angular_velocities = []
    for j in range(len(samples) - 1):
        rotation = rotation_matrices[j].T @ rotation_matrices[j + 1]
        rotation_vector, _ = cv2.Rodrigues(rotation)
        dt = (samples[j + 1].timestamp_ns - samples[j].timestamp_ns) / 1e9
        angular_velocity = rotation_vector.flatten() / dt
        angular_velocities.append(angular_velocity)
    angular_velocities = np.array(angular_velocities)

    return SlamGroundTruthResult(
        times=times,
        positions=positions,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocity_times=np.array([(s.timestamp_ns - first_timestamp_ns) / 1e9 for s in samples[:-1]]),
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
    times = np.array([(s.timestamp_ns - first_timestamp_ns) / 1e9 for s in samples])
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

    # inversing the pose from cv2.solvePnPRansac as they are the inverse of
    # what the rest of the code expects.
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


# Sanity bound on a verified closure's recovered translation magnitude: no real revisit inside a
# single EuRoC room traverses farther than this. Needed because verification (num_matches,
# reprojection_error) does not by itself catch a degenerate PnP/RANSAC solution -- an
# instrumented sweep on V2_03_difficult found ~1.1% of otherwise-plausible-looking verified pairs
# come back with translation magnitudes in the millions of meters (obvious RANSAC garbage from a
# bad point configuration), and neither of the other two signals flagged them.
LOOP_CLOSURE_MAX_TRANSLATION_M = 20.0

# Noise for an auto-detected closure inserted into the real pose graph (see _run_gtsam's
# enable_loop_closure). Plain Gaussian, not Huber or DCS: both were tested and found to cap or
# outright zero a *correct* closure's pull once the prior disagreement is many multiples of
# sigma, which it always is right after a long blackout -- see the Loop Closure plan's Phase 2/3
# writeup. False-positive protection lives entirely in _find_loop_closures' verification gate
# (match count + degeneracy check), not in this noise model.
LOOP_CLOSURE_ROT_SIGMA = 0.02   # rad
LOOP_CLOSURE_TRANS_SIGMA = 0.05  # m


def _find_loop_closures(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    keyframe_indices: list[int],
    min_temporal_gap_s: float = 10.0,
    min_matches: int = 200,
) -> list[LoopClosureCandidate]:
    """For every keyframe, check whether it revisits an earlier one: plain ORB descriptor match
    count as the place-recognition signal (no bag-of-words dependency needed at this scale --
    validated on ~320 keyframes in seconds), then PnP/RANSAC geometric verification via the same
    _run_pnp_step already used for temporal matching elsewhere in this module.

    Only the single best-scoring earlier candidate is kept per query, and same-event clusters are
    then consolidated (see _consolidate_loop_closure_clusters) into one representative each -- a
    stay-put stretch has every one of its keyframes independently "rediscover" the same historical
    match, and inserting all of them as separate factors overcounts what is really one correlated
    observation as if it were N independent ones. A regression check on MH_02_easy and
    MH_04_difficult found this artificially over-stiffens that part of the graph and perturbs
    other, untouched regions through the shared IMU/bias chain. A candidate is only returned if it
    clears min_matches (an instrumented precision/recall sweep on V2_03_difficult found ~84-92%
    precision, measured as correct relative-pose recovery rather than raw position proximity, at
    150-300) *and* the translation-magnitude sanity check above.
    """
    K = len(keyframe_indices)
    first_ts = data.cam_timestamps_ns[0]
    kf_times = np.array([
        (stereo_matching_result.frames[k].timestamp_ns - first_ts) / 1e9 for k in keyframe_indices])
    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates: list[LoopClosureCandidate] = []
    for q in range(K):
        q_frame = keyframe_indices[q]
        q_desc = feature_detection_result.frames[q_frame].cam0_descriptors
        if q_desc is None or len(q_desc) == 0:
            continue
        earlier = [c for c in range(q) if kf_times[q] - kf_times[c] >= min_temporal_gap_s]
        if not earlier:
            continue

        best_score, best_c = -1, -1
        for c in earlier:
            c_desc = feature_detection_result.frames[keyframe_indices[c]].cam0_descriptors
            if c_desc is None or len(c_desc) == 0:
                continue
            raw = bf.knnMatch(c_desc, q_desc, k=2)
            score = sum(1 for pair in raw if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance)
            if score > best_score:
                best_score, best_c = score, c
        if best_score < min_matches:
            continue

        c_frame = keyframe_indices[best_c]
        c_sm = stereo_matching_result.frames[c_frame]
        c_fd = feature_detection_result.frames[c_frame]
        q_fd = feature_detection_result.frames[q_frame]
        if not c_sm.matches:
            continue
        try:
            rvec, tvec, num_matches, reproj_err = _run_pnp_step(
                data, c_sm.points_3d, c_sm.matches,
                c_fd.cam0_descriptors, q_fd.cam0_keypoints, q_fd.cam0_descriptors)
        except Exception:
            continue
        if num_matches < min_matches or float(np.linalg.norm(tvec)) > LOOP_CLOSURE_MAX_TRANSLATION_M:
            continue

        pnp_cam0 = np.eye(4)
        pnp_cam0[:3, :3], _ = cv2.Rodrigues(rvec)
        pnp_cam0[:3, 3] = tvec.flatten()
        rel_pose_body = body_T_cam0 @ pnp_cam0 @ cam0_T_body
        candidates.append(LoopClosureCandidate(c_frame, q_frame, rel_pose_body, num_matches, float(reproj_err)))

    return _consolidate_loop_closure_clusters(data, stereo_matching_result, candidates)


# Candidates this close together in *both* query time and anchor time are treated as the same
# revisit event rather than independent evidence -- chained (each candidate compared to the
# previous one already placed in the cluster, not to the cluster's first member) so a slowly
# drifting match target during a hover/settle still merges into one cluster even though its first
# and last anchors may differ by more than this on their own.
LOOP_CLOSURE_CLUSTER_GAP_S = 2.0


def _consolidate_loop_closure_clusters(
    data: EuRoCMAVData,
    stereo_matching_result: StereoMatchingResult,
    candidates: list[LoopClosureCandidate],
) -> list[LoopClosureCandidate]:
    """Collapse a run of candidates that are really one revisit event (e.g. every keyframe during
    a stay-put stretch independently matching the same historical frame) into a single
    representative -- the one with the most temporal matches. Keeps genuinely separate revisits
    to a similar place, since those won't be adjacent in query time to begin with.

    Improved, not resolved: re-checked against all three regression-check datasets after adding
    this. MH_04_difficult's regression (from the original per-query, no-consolidation version) is
    essentially gone (~12% worse than baseline in its known-sensitive ~46s region -> ~1% or
    better). MH_02_easy is mixed -- some points improved, but its worst regression point barely
    moved and the overall final-trajectory error got worse than the unconsolidated version (still
    better than no closures at all, just not as good). And on V2_03_difficult itself (94 closures
    -> 15), the correction is measurably weaker during the transition into the corrected region,
    though still a >95% reduction from baseline overall. So some of what this collapses down was
    real corrective signal, not pure redundancy -- collapsing every same-event cluster to exactly
    one representative is too aggressive in at least one direction. LOOP_CLOSURE_CLUSTER_GAP_S=2.0
    and "keep only the single best match per cluster" are both first guesses, not tuned values;
    likely next step is keeping a small spread of representatives per cluster instead of one.
    """
    if not candidates:
        return []
    first_ts = data.cam_timestamps_ns[0]

    def t(frame_idx: int) -> float:
        return (stereo_matching_result.frames[frame_idx].timestamp_ns - first_ts) / 1e9

    ordered = sorted(candidates, key=lambda c: c.to_frame)
    consolidated: list[LoopClosureCandidate] = []
    cluster = [ordered[0]]
    for c in ordered[1:]:
        prev = cluster[-1]
        same_event = (abs(t(c.to_frame) - t(prev.to_frame)) <= LOOP_CLOSURE_CLUSTER_GAP_S
                      and abs(t(c.from_frame) - t(prev.from_frame)) <= LOOP_CLOSURE_CLUSTER_GAP_S)
        if not same_event:
            consolidated.append(max(cluster, key=lambda x: x.num_matches))
            cluster = []
        cluster.append(c)
    consolidated.append(max(cluster, key=lambda x: x.num_matches))
    return consolidated


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
    # Anchor the trajectory to GT at the first GT sample inside the window (after start_s),
    # not the dataset's first sample -- otherwise the windowed poses get aligned to a GT pose
    # from before the window even begins.
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
        (stereo_matching_result.frames[i].timestamp_ns - first_timestamp_ns) / 1e9
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
    """A single sighting of one landmark at one keyframe. uv0/uv1 are undistorted pixel
    coordinates in cam0/cam1 (matching a Cal3_S2 pinhole model); point_cam0 is the
    stereo-triangulated 3D point in the cam0 frame at that observation, used only to seed the
    landmark's initial 3D guess -- the shared position estimate itself lives in the factor graph.
    """
    node: int  # graph-node index (into keyframe_indices) that observed the landmark
    uv0: np.ndarray
    uv1: np.ndarray
    point_cam0: np.ndarray


# Match a keyframe's stereo-matched cam0 keypoint to the optical-flow track currently sitting on
# it, if any. A track is seeded from an ORB keypoint (optical_flow.py) and drifts sub-pixel from
# there via KLT, so it rarely lands exactly on a keypoint re-detected fresh at a later frame --
# but a real corner keeps getting re-detected close to where the track still is, so a small pixel
# radius is enough to identify it without any descriptor matching. ORB keypoints can still cluster
# tightly enough that two of them both fall within this radius of the same track (common on richly
# textured regions), so _snap_obs_to_tracks additionally enforces one obs per track. 3.0 was too
# tight on V1_03_difficult's fast-rotation segments (KLT drift between keyframes up to MAX_GAP_S=0.75s
# (15 frames at the nominal 20Hz cam rate) apart exceeded it, dropping real correspondences and starving keyframes of landmarks --
# fewer, sparser tracks hurt rotation accuracy more than the snap collisions they avoided). Kept
# comfortably under optical_flow.MIN_TRACK_SEPARATION_PX (8px) so it still can't straddle two
# distinct live tracks.
LANDMARK_SNAP_PX = 6.0


def _snap_obs_to_tracks(fd, sm, optical_flow_frame: OpticalFlowFrame) -> dict[int, int]:
    """obs index (into sm.matches) -> optical-flow track id, for the nearest live track within
    LANDMARK_SNAP_PX, if any. Each track claims at most one obs (its closest) -- ORB keypoints can
    cluster tightly enough that two of them both land within the snap radius of the same track;
    without this a track could pick up two observations at the same keyframe, which breaks the
    one-observation-per-node assumption the rest of _build_landmark_tracks and _run_gtsam rely on.
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


def _build_landmark_tracks(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    optical_flow_result: OpticalFlowResult,
    keyframe_indices: list[int],
    min_track_len: int,
    depth_min: float,
    depth_max: float,
) -> dict[int, list[LandmarkObservation]]:
    """Chain stereo-matched features across keyframes into persistent landmark tracks.

    Only cam0-cam1 stereo inliers are tracked, so every observation carries a metric depth for
    initialization; observations are pre-undistorted so they match a Cal3_S2 pinhole model.
    Correspondence across keyframes comes from optical flow's already-tracked ids (a track id is
    valid correspondence through every intermediate frame, not just the two keyframes being
    linked) instead of re-matching ORB descriptors keyframe-to-keyframe -- the same fast-rotation
    failure mode OpticalFlowSolver's docstring documents for that approach.
    """
    K0 = data.cam0_intrinsics.to_matrix()
    K1 = data.cam1_intrinsics.to_matrix()
    dist0 = np.array([data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
                      data.cam0_intrinsics.p1, data.cam0_intrinsics.p2])
    dist1 = np.array([data.cam1_intrinsics.k1, data.cam1_intrinsics.k2,
                      data.cam1_intrinsics.p1, data.cam1_intrinsics.p2])

    # Per node: this keyframe's stereo-inlier observations, and which live optical-flow track (if
    # any) each one corresponds to.
    node_obs: list[list[LandmarkObservation]] = []
    node_obs_flow_tid: list[list[Optional[int]]] = []
    for jj, frame in enumerate(keyframe_indices):
        sm = stereo_matching_result.frames[frame]
        fd = feature_detection_result.frames[frame]
        if not sm.matches:
            node_obs.append([])
            node_obs_flow_tid.append([])
            continue
        q_idx = [m.queryIdx for m in sm.matches]
        t_idx = [m.trainIdx for m in sm.matches]
        uv0 = np.array([fd.cam0_keypoints[i].pt for i in q_idx], dtype=np.float64)
        uv1 = np.array([fd.cam1_keypoints[i].pt for i in t_idx], dtype=np.float64)
        uv0u = cv2.undistortPoints(uv0.reshape(-1, 1, 2), K0, dist0, P=K0).reshape(-1, 2)
        uv1u = cv2.undistortPoints(uv1.reshape(-1, 1, 2), K1, dist1, P=K1).reshape(-1, 2)
        pts = sm.points_3d.T  # (M, 3), column i <-> sm.matches[i]
        node_obs.append([LandmarkObservation(jj, uv0u[i], uv1u[i], pts[i]) for i in range(len(sm.matches))])

        obs_to_flow_tid = _snap_obs_to_tracks(fd, sm, optical_flow_result.frames[frame])
        node_obs_flow_tid.append([obs_to_flow_tid.get(i) for i in range(len(sm.matches))])

    tracks: dict[int, list[LandmarkObservation]] = {}
    next_tid = 0
    prev_flow_tid_to_tid: dict[int, int] = {}  # optical-flow track id (at node jj-1) -> our track id
    for jj in range(len(keyframe_indices)):
        cur_flow_tid_to_tid: dict[int, int] = {}
        for landmark_obs, flow_tid in zip(node_obs[jj], node_obs_flow_tid[jj]):
            if flow_tid is None:
                continue
            tid = prev_flow_tid_to_tid.get(flow_tid)
            if tid is None:
                tid = next_tid
                next_tid += 1
                tracks[tid] = []
            tracks[tid].append(landmark_obs)
            cur_flow_tid_to_tid[flow_tid] = tid
        prev_flow_tid_to_tid = cur_flow_tid_to_tid

    # Keep only tracks long enough to bundle-adjust and whose first (init) depth is sane.
    kept: dict[int, list[LandmarkObservation]] = {}
    for tid, obs in tracks.items():
        if len(obs) < min_track_len:
            continue
        z0 = float(obs[0].point_cam0[2])
        if not (depth_min < z0 < depth_max):
            continue
        kept[tid] = obs
    return kept


def _scan_keyframes(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    N: int,
    on_progress: Optional[Callable[[float], None]] = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Single per-frame PnP scan that yields BOTH the factor-graph nodes and a per-frame PnP
    trajectory, so the ref->frame ORB matching (the SLAM stage's dominant cost) is done once
    rather than separately for keyframe selection and the PnP diagnostic.

    Adaptive keyframing: a single PnP from the current reference keyframe to frame i yields both
    signals we need at once -- the temporal-match count (covisibility with the reference, which
    decays as we move away) and the relative pose (motion since the reference). We open a new
    keyframe when overlap drops below a fraction of the post-keyframe baseline, when enough
    translation/rotation has accrued, or when a hard frame cap is hit -- all clamped by a min gap
    so we never place near-duplicate nodes. Compared to a fixed stride this keeps large,
    well-conditioned baselines when the camera moves fast and avoids zero-parallax nodes when it
    hovers. The chained relative poses are also returned as a per-frame cam0 dead-reckoning
    trajectory for the PnP diagnostic view.
    """
    MIN_GAP = 3                     # never place keyframes closer than this (avoid duplicate nodes)
    # Time-based, not frame-count: a frame-count cap silently doubles its effective time span
    # whenever the camera's actual frame rate drops below the nominal 20Hz it was tuned for (e.g.
    # cam0 dropping to ~10Hz during V2_03_difficult's ~65-75s exposure glitch), which is exactly
    # when this bound on IMU-only preintegration drift matters most. 0.75s = 15 frames at 20Hz,
    # matching the original tuning.
    MAX_GAP_S = 0.75                # force a keyframe at least this often (bound IMU preint. drift)
    COVIS_RATIO = 0.6               # new keyframe once covisibility falls below this * baseline
    TRANS_THRESH = 0.2              # ... or once translation since the reference exceeds this [m]
    ROT_THRESH = np.deg2rad(10.0)   # ... or rotation exceeds this [rad]
    # This probe only needs a covisibility *count* and a rough relative pose to decide where to
    # place nodes -- not a precise reconstruction. ORB brute-force matching is O(features^2) and
    # dominates the whole SLAM stage, so match only the strongest PROBE_N descriptors (ORB returns
    # them response-ordered). The covisibility ratio is scale-free (baseline and current are both
    # capped the same way), so placement is essentially unchanged at a fraction of the cost.
    PROBE_N = 500
    # Never anchor a graph node on a "dead-vision" frame -- one with too few stereo matches to
    # triangulate/observe landmarks (motion blur, low texture). Such a keyframe gets no landmark
    # reprojection factors, so its pose leans entirely on a noisy chained PnP + the IMU factor;
    # ISAM2 then reconciles their disagreement by spiking the gyro bias, corrupting the pose and
    # injecting a permanent trajectory offset (see the ~20 s failure on MH_02_easy). The floor is
    # ~1st percentile of per-frame stereo-match counts, so it only rejects the genuinely starved
    # tail, not ordinary low-overlap keyframes. Dead nodes are dropped after selection (below).
    MIN_KF_MATCHES = 80

    def _stereo_count(frame_idx: int) -> int:
        return len(stereo_matching_result.frames[frame_idx].matches)

    keyframes = [0]
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
            keyframes.append(kf)
            if poses[kf] is None:  # this frame never got its own PnP; carry the reference pose
                poses[kf] = poses[ref_idx]
            ref_idx, ref_covis = kf, None
            i = kf + 1
            continue

        # Chain the reference->i transform onto the reference pose for the per-frame trajectory.
        step = np.eye(4)
        step[:3, :3], _ = cv2.Rodrigues(rvec)
        step[:3, 3] = tvec.flatten()
        # poses[ref_idx] is always populated by the time it's read here (poses[0] is seeded
        # above, and ref_idx only ever points at an already-visited frame) -- Pylance can't
        # prove that invariant from the list's `Optional[np.ndarray]` element type.
        poses[i] = poses[ref_idx] @ step  # type: ignore[operator]

        if ref_covis is None:  # first frame after a keyframe sets the covisibility baseline
            ref_covis = num_matches

        gap = i - ref_idx
        translation = float(np.linalg.norm(tvec))
        rotation = float(np.linalg.norm(rvec))
        elapsed_s = (stereo_matching_result.frames[i].timestamp_ns
                     - stereo_matching_result.frames[ref_idx].timestamp_ns) / 1e9

        force = elapsed_s >= MAX_GAP_S
        allow = gap >= MIN_GAP
        weak = num_matches < COVIS_RATIO * ref_covis
        moved = translation > TRANS_THRESH or rotation > ROT_THRESH

        if force or (allow and (weak or moved)):
            keyframes.append(i)
            ref_idx, ref_covis = i, None
        i += 1

    if keyframes[-1] != N - 1:
        keyframes.append(N - 1)

    # Drop interior keyframes that landed on dead vision -- rather than pin a pose on a starved
    # frame, merge its neighbours so the IMU factor carries the gap (extend the IMU-only interval).
    # Keep the frame-0 anchor and the terminal node. Bound the merged gap so removing a run of dead
    # frames can't open an unbounded IMU-only stretch; if it would, keep that node as the least-bad
    # option to cap preintegration drift. Time-based (see MAX_GAP_S above) for the same reason.
    DROP_MAX_GAP_S = 2 * MAX_GAP_S
    kept = [keyframes[0]]
    for k in keyframes[1:-1]:
        elapsed_s = (stereo_matching_result.frames[k].timestamp_ns
                     - stereo_matching_result.frames[kept[-1]].timestamp_ns) / 1e9
        if _stereo_count(k) < MIN_KF_MATCHES and elapsed_s <= DROP_MAX_GAP_S:
            continue
        kept.append(k)
    kept.append(keyframes[-1])
    keyframes = kept
    # Forward-fill any frame that never received a pose (rare: skipped by an exception jump).
    last = np.eye(4)
    filled = []
    for p in poses:
        last = p if p is not None else last
        filled.append(last)
    return keyframes, filled


def _run_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    optical_flow_result: OpticalFlowResult,
    imu_samples: list[ImuSample],
    gravity: np.ndarray,
    keyframe_indices: list[int],
    on_progress: Callable[[float, str], None],
    # Run automated loop-closure detection (_find_loop_closures) and insert whatever it finds into
    # this graph, using LOOP_CLOSURE_ROT_SIGMA/TRANS_SIGMA with plain Gaussian noise. Off by
    # default: _find_loop_closures is O(K^2) brute-force descriptor matching, and the "keep one
    # representative per revisit event" consolidation it does is validated-but-not-tuned (see
    # _consolidate_loop_closure_clusters' docstring) -- it clearly helps on a real blackout and
    # was regression-checked clean on one other sequence, but a second sequence still showed a
    # real, if smaller, localized regression. Ignored if extra_loop_closures below is given
    # explicitly (manual override always wins over auto-detection).
    enable_loop_closure: bool = False,
    # Manual override / test hook (see tmp/investigate/v2_03_loop_closure_phase*.py): each tuple
    # is (from_frame_idx, to_frame_idx, body_relative_pose_4x4, rot_sigma_rad, trans_sigma_m,
    # noise_mode). Frame indices (not node indices) so callers don't need to know how
    # MIN_KF_LANDMARKS re-keying below will renumber nodes -- resolved to nodes internally once
    # keyframe_indices is final. noise_mode in {"gaussian", "huber", "dcs"}: use "gaussian" --
    # Huber's linear (not quadratic) loss caps a *correct* closure's pull once the prior
    # disagreement is many multiples of sigma (which it always is, right after a long blackout),
    # and DCS (Dynamic Covariance Scaling) turned out worse, not better: as a *redescending*
    # estimator its influence doesn't just cap for large residuals, it vanishes -- it can't tell
    # a huge-but-correct residual from a huge-and-wrong one and discards both. False-positive
    # protection belongs in verification (_find_loop_closures' match-count + degeneracy gate),
    # not in the inserted factor's own robustness.
    # Defaults to a no-op: omitting both this and enable_loop_closure leaves _run_gtsam's output
    # byte-for-byte unchanged.
    extra_loop_closures: Optional[list[tuple[int, int, np.ndarray, float, float, str]]] = None,
    # Test-only introspection hook (see v2_03_loop_closure_phase2b.py): if provided, filled with
    # {'isam2': isam2} so a caller can run its own batch re-optimization over the full factor
    # graph afterward, without _run_gtsam's return signature changing for existing callers.
    debug_out: Optional[dict] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray, list[int]]:
    N = len(stereo_matching_result.frames)
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    # Mark read-only: these absolute-ns timestamps are a lookup table we only ever *read* (via
    # `imu_timestamps_ns - cam_timestamps_ns[0]` etc.). Without this, NumPy's temporary-elision
    # optimization can execute that subtraction in place -- when this array is large (long IMU
    # prefix) and has refcount 1, `a - scalar` mutates `a` and returns it -- silently turning
    # every timestamp relative, emptying every IMU window, and producing zero-dt (degenerate)
    # ImuFactors that make ISAM2 indeterminate. Read-only arrays are never elided.
    imu_timestamps_ns.flags.writeable = False
    imu_lin_accs      = np.array([s.linear_acceleration for s in imu_samples])
    imu_ang_vels      = np.array([s.angular_velocity for s in imu_samples])
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    # Estimate the gravity direction in the body frame *at frame 0* by averaging the
    # accelerometer over a short window there (at rest it senses specific force, "up").
    # A global most-static window would measure gravity at a different body orientation,
    # because the body is generally rotated differently by the time it settles; integrating
    # the gyro across that gap drifts too much to correct for it reliably.
    grav_win = max(1, int(data.imu0_rate_hz * 0.5))
    i0 = int(np.argmin(np.abs(imu_timestamps_ns - cam_timestamps_ns[0])))
    gravity_in_body = imu_lin_accs[i0:i0 + grav_win].mean(axis=0)

    # Progress budget across this stage's sub-steps (fractions of the GTSAM phase):
    #   landmark tracks    [0.00, 0.10]
    #   ISAM2 forward loop [0.10, 0.92]
    #   reprojection metrics [0.92, 1.00]
    # keyframe_indices (from the shared _scan_keyframes pass) maps graph node j -> original frame
    # index; gaps between them vary since nodes are chosen adaptively (covisibility + motion).
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
    # Per-step (adjacent-frame) PnP sigmas. A keyframe constraint chains the intermediate steps,
    # so its error accumulates like a random walk -> scale the sigmas by sqrt(gap) per interval,
    # where gap is that interval's frame count. Built inside the loop since gaps now vary.
    PNP_STEP_SIGMAS  = np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05])
    # Random-walk noise letting the (per-keyframe) IMU bias evolve between keyframes instead of
    # being pinned to a single constant. The accumulated std over an interval of duration dt is
    # (diffusion_density * sqrt(dt)) per axis, so it is built inside the loop (keyframe gaps vary).
    # Both terms use the IMU datasheet's physical diffusion (EuRoC ADIS16448, imu0/sensor.yaml).
    # The gyro term was tightened first -- the old blanket sigma=0.1 permitted ~1 rad/s jumps per
    # keyframe and let the optimizer absorb a pose/rotation error by spiking the gyro bias (the
    # ~20 s MH_02_easy failure), forcing that residual into the (vision-correctable) pose instead.
    # The accel term used to stay at the same loose blanket 0.1 (with no sqrt(dt) scaling at all)
    # to "soak up unmodeled gravity/scale error" -- but that let accel bias free-walk in axes vision
    # can't correct (e.g. z on a near-level flight segment, where there's almost no vertical
    # parallax to check it against), producing a near-linear position drift over tens of seconds.
    # Tightened to the same physical-diffusion treatment as gyro.
    GYRO_BIAS_RW  = 1.9393e-05   # [rad/s^2/sqrt(Hz)]  gyroscope_random_walk from imu0/sensor.yaml
    ACCEL_BIAS_RW = 3.0000e-03   # [m/s^3/sqrt(Hz)]    accelerometer_random_walk from imu0/sensor.yaml
    # Fallback regularizer for a keyframe that ends up with neither a PnP between-factor nor any
    # landmark reprojection factor -- e.g. fast motion where feature tracks break and the chained
    # PnP fails. Such a pose would hang off only its IMU factor (yaw about gravity unobservable),
    # making ISAM2's system indeterminate. A weak prior at the IMU-predicted pose pins the free
    # directions without fighting real constraints; it is far looser than PnP/reprojection noise.
    FALLBACK_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)

    # --- Landmarks (rung 2: explicit structure / bundle adjustment) ---------------------
    # Persistent 3D landmarks tied to poses by reprojection factors. A point seen across
    # several keyframes becomes one shared variable with many rigid constraints (errors
    # average), instead of a chain of noisy relative PnP poses (errors compound).
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
    # Weak prior anchoring each landmark to its stereo-triangulated init. Far / low-parallax
    # points are barely constrained along the viewing ray and would make ISAM2's system
    # indeterminate; this regularizes them. It is orders of magnitude looser than the pixel
    # factors, so it is negligible for well-observed landmarks.
    LM_PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)

    on_progress(0.0, "Building landmark tracks...")
    tracks = _build_landmark_tracks(
        data, feature_detection_result, stereo_matching_result, optical_flow_result, keyframe_indices,
        MIN_TRACK_LEN, DEPTH_MIN, DEPTH_MAX)

    # MIN_KF_MATCHES (in _scan_keyframes) only rejects a keyframe with too few *stereo*
    # (cam0-cam1, same-instant) matches. A keyframe can clear that easily -- plenty of L/R
    # overlap -- and still land almost no landmark *tracks*, if temporal (frame-to-frame)
    # matching is what's failing: a lighting swing or glare patch that changes the scene's
    # appearance between keyframes without touching stereo overlap at all. Such a node gets
    # too few reprojection factors to pin its pose, so ISAM2 reconciles the shortfall against
    # the IMU factor by pulling the accel bias instead (see ACCEL_BIAS_RW above) -- the
    # accel-bias runaway + position drift seen on MH_04_difficult ~46s.
    # Floor is ~5th percentile of per-keyframe landmark counts on a clean run, so it only
    # rejects the genuinely track-starved tail, not ordinary low-structure keyframes.
    # Lowered from 100 -> 60: on V2_02_medium's ~74-78s fast vertical maneuver, 100 was
    # dropping every keyframe in a multi-second, uniformly-mediocre (60-97 landmark) stretch
    # except the one forced to survive by DROP_MAX_GAP -- but that survivor's own tracks
    # depended on the just-dropped neighbours, so it came out with 0 landmarks (worse than
    # if the gate had left the stretch alone). Not a full fix -- the gate is non-causal (it
    # decides drops from pre-drop counts, never re-checking whether a survivor still clears
    # MIN_TRACK_LEN after its neighbours are gone) -- but 60 keeps this specific stretch intact.
    MIN_KF_LANDMARKS = 60
    # Time-based, not frame-count, for the same reason as _scan_keyframes' MAX_GAP_S: a
    # frame-count cap silently doubles its effective time span whenever the camera's actual frame
    # rate drops below the nominal 20Hz it was tuned for (e.g. cam0 dropping to ~10Hz during
    # V2_03_difficult's ~65-75s exposure glitch). 1.5s = 30 frames at 20Hz, matching the original
    # tuning.
    DROP_MAX_GAP_S = 1.5
    landmark_counts_per_node = [0] * K
    for obs in tracks.values():
        for o in obs:
            landmark_counts_per_node[o.node] += 1
    if any(c < MIN_KF_LANDMARKS for c in landmark_counts_per_node[1:-1]):
        kept_positions = [0]
        for pos in range(1, K - 1):
            elapsed_s = (cam_timestamps_ns[keyframe_indices[pos]]
                         - cam_timestamps_ns[keyframe_indices[kept_positions[-1]]]) / 1e9
            if (landmark_counts_per_node[pos] < MIN_KF_LANDMARKS
                    and elapsed_s <= DROP_MAX_GAP_S):
                continue
            kept_positions.append(pos)
        kept_positions.append(K - 1)
        if len(kept_positions) != K:
            # Re-key the *already-built* tracks onto the surviving nodes instead of asking
            # _build_landmark_tracks to re-derive temporal matches over the now-wider gaps
            # between them: that would re-run frame-to-frame descriptor matching across a
            # bigger baseline exactly where appearance is already unstable, trading one
            # starved node for worse tracks at every surviving neighbour (verified: rebuilding
            # made position error worse, not better -- 0.62m -> 1.15m by t=50s on this dataset).
            old_to_new = {old: new for new, old in enumerate(kept_positions)}
            remapped_tracks = {}
            for tid, obs in tracks.items():
                filtered = [LandmarkObservation(old_to_new[o.node], o.uv0, o.uv1, o.point_cam0)
                            for o in obs if o.node in old_to_new]
                if len(filtered) >= MIN_TRACK_LEN:
                    remapped_tracks[tid] = filtered
            tracks = remapped_tracks
            keyframe_indices = [keyframe_indices[p] for p in kept_positions]
            K = len(keyframe_indices)

    # Auto-detect closures against the *final* (post-re-keying) keyframe set, so a node dropped
    # above for landmark starvation can't be picked as an endpoint. Manual extra_loop_closures
    # (if given) always overrides auto-detection rather than adding to it -- keeps the test/
    # override path from Phase 2/3 exact and predictable.
    if enable_loop_closure and extra_loop_closures is None:
        detected = _find_loop_closures(data, feature_detection_result, stereo_matching_result, keyframe_indices)
        extra_loop_closures = [
            (c.from_frame, c.to_frame, c.body_relative_pose,
             LOOP_CLOSURE_ROT_SIGMA, LOOP_CLOSURE_TRANS_SIGMA, "gaussian")
            for c in detected
        ]

    # Resolve loop-closure endpoints to final node indices now that re-keying (above) has settled
    # keyframe_indices -- keyed by the *to* node, since that's when the factor becomes addable
    # (its *from* node, always earlier, is already in the graph by then).
    loop_closures_by_node: dict[int, list[tuple[int, np.ndarray, float, float, str]]] = {}
    if extra_loop_closures:
        kf_arr = np.array(keyframe_indices)
        for from_frame, to_frame, rel_pose, rot_sigma, trans_sigma, noise_mode in extra_loop_closures:
            from_node = int(np.argmin(np.abs(kf_arr - from_frame)))
            to_node = int(np.argmin(np.abs(kf_arr - to_frame)))
            loop_closures_by_node.setdefault(to_node, []).append(
                (from_node, rel_pose, rot_sigma, trans_sigma, noise_mode))

    # node -> track ids observed there; and per-interval covisibility for the PnP gate.
    nodes_to_tracks: dict[int, list[int]] = {jj: [] for jj in range(K)}
    node_seen: list[set[int]] = [set() for _ in range(K)]
    for tid, obs in tracks.items():
        for o in obs:
            nodes_to_tracks[o.node].append(tid)
            node_seen[o.node].add(tid)
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

        Returns how many landmark observations were staged *at node jj* -- i.e. how many
        projection factors now reference X(jj). The caller uses this to tell whether the new
        pose picked up any structure constraint (see the underconstrained-pose guard below).
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

    # Anchor the GTSAM navigation frame G to be gravity-aligned: rotate body-frame-0
    # so its measured gravity lands on the nav-frame gravity axis. At rest the
    # accelerometer reads "up" (specific force), so R_G_body0 @ gravity_in_body = -gravity.
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

    for j in range(K - 1):
        on_progress(0.10 + (j / (K - 1)) * (0.92 - 0.10), "Optimizing (ISAM2)...")
        est = isam2.calculateEstimate()
        pose_i = est.atPose3(X(j))
        vel_i  = est.atVector(V(j))
        bias_i = est.atConstantBias(B(j))

        kf_i, kf_next = keyframe_indices[j], keyframe_indices[j + 1]

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

        # Build the keyframe->keyframe PnP constraint by chaining the intermediate
        # adjacent-frame PnP steps. Each step is between neighbouring frames, where feature
        # overlap is high, so it is well-conditioned; a single direct match across the full
        # ~10-frame gap would have little overlap and fail often. If any step fails, drop the
        # whole constraint for this interval and let the IMU factor carry it.
        #
        # PnP is now a *fallback*: when this keyframe pair already shares plenty of *inserted*
        # landmarks, their reprojection factors constrain the relative pose directly, so adding
        # PnP on top would double-count the same pixels. Only count landmarks that are already
        # mature (inserted) and seen at both endpoints -- those are the ones that actually
        # contribute factors linking X(j) and X(j+1) after this update. A track that is merely
        # covisible but not yet mature contributes nothing here, so PnP must still carry it,
        # otherwise the new pose would be left underconstrained (indeterminate system).
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
                step = np.eye(4)
                step[:3, :3], _ = cv2.Rodrigues(rvec)
                step[:3, 3] = tvec.flatten()
                pnp_cam0 = pnp_cam0 @ step

            if pnp_ok:
                pnp_body = body_T_cam0 @ pnp_cam0 @ cam0_T_body
                pnp_delta = gtsam.Pose3(gtsam.Rot3(pnp_body[:3, :3]), gtsam.Point3(*pnp_body[:3, 3]))
                pnp_noise = gtsam.noiseModel.Diagonal.Sigmas(PNP_STEP_SIGMAS * np.sqrt(kf_next - kf_i))
                new_factors.add(gtsam.BetweenFactorPose3(X(j), X(j + 1), pnp_delta, pnp_noise))
                pose_init = pose_i.compose(pnp_delta)
                pnp_added = True

        new_values.insert(X(j + 1), pose_init)
        new_values.insert(V(j + 1), vel_init)
        new_values.insert(B(j + 1), bias_i)
        # Reprojection factors for landmarks observed at the new keyframe. X(j+1) is staged in
        # new_values (valid within this same update); landmark inits use poses from `est`, all
        # of which predate node j+1, so they are already in the estimate.
        n_proj_at_next = _process_node_landmarks(j + 1, est, new_factors, new_values)
        # Guard: never add a keyframe without a relative constraint. If neither the PnP fallback
        # nor any landmark reprojection factor touched X(j+1), it would hang off only its IMU
        # factor and make ISAM2 indeterminate. Anchor it with a weak prior at the IMU prediction.
        if not pnp_added and n_proj_at_next == 0:
            new_factors.add(gtsam.PriorFactorPose3(X(j + 1), pose_init, FALLBACK_POSE_NOISE))

        for from_node, rel_pose, rot_sigma, trans_sigma, noise_mode in loop_closures_by_node.get(j + 1, []):
            delta = gtsam.Pose3(gtsam.Rot3(rel_pose[:3, :3]), gtsam.Point3(*rel_pose[:3, 3]))
            base_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([rot_sigma] * 3 + [trans_sigma] * 3))
            if noise_mode == "huber":
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise)
            elif noise_mode == "dcs":
                # DCS's scale factor is min(1, 2c / (c + ||r||^2)) where ||r||^2 is the whitened
                # (chi-squared) residual -- so c must be calibrated to this factor's degrees of
                # freedom (6, for a Pose3 BetweenFactor), not left at the textbook-example c=1.
                # A genuinely-consistent 6-DOF residual has ||r||^2 around 6 on average; c=1
                # measured this down to a ~0.29 scale for an *already-correct* closure (verified
                # empirically: it made a 94-closure test on V2_03_difficult behave identically to
                # having no closures at all). c=6 keeps full trust up to the expected value and
                # only meaningfully down-weights past the ~12.6 (95th-percentile chi2_6) mark --
                # i.e. genuine outliers, not closures merely large-but-correct after a long
                # blackout.
                loop_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.DCS.Create(6.0), base_noise)  # type: ignore[attr-defined]
            else:
                loop_noise = base_noise
            new_factors.add(gtsam.BetweenFactorPose3(X(from_node), X(j + 1), delta, loop_noise))

        isam2.update(new_factors, new_values)

    print(f"landmarks: {len(inserted_landmarks)}/{len(tracks)} tracks used, "
          f"{n_proj_factors} reprojection factors")
    on_progress(0.92, "Computing metrics...")
    final = isam2.calculateEstimate()
    poses = [final.atPose3(X(j)).matrix() for j in range(K)]
    velocities = [final.atVector(V(j)) for j in range(K)]
    biases = [final.atConstantBias(B(j)).vector() for j in range(K)]  # [accel(3), gyro(3)]

    # Per-keyframe landmark-quality metrics: reprojection RMSE [px] and how many landmarks were
    # observed there. Reproject each final landmark into every keyframe that saw it, through
    # both cameras, and compare to the (undistorted) measurement.
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


def _get_gtsam_result(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    optical_flow_result: OpticalFlowResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
    gravity: np.ndarray,
    keyframe_indices: list[int],
    on_progress: Callable[[float, str], None],
    enable_loop_closure: bool = False,
) -> SlamGtsamResult:
    # Window the IMU to [min, max]; only this range is ever integrated. Leaving the whole t=0
    # prefix in (as `<= max` alone did) needlessly grows the array -- and, at larger start_s,
    # pushes it past NumPy's temporary-elision size threshold (see the read-only guard in
    # _run_gtsam), so keeping it windowed is both cheaper and safer.
    imu_samples = [s for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    # _run_gtsam gets [0.00, 0.95] of this stage; the trajectory alignment below takes the rest.
    poses, velocities, biases, reprojection_rmse, landmark_counts, keyframe_indices = _run_gtsam(
        data, feature_detection_result, stereo_matching_result, optical_flow_result, imu_samples, gravity,
        keyframe_indices, on_progress=lambda p, lbl: on_progress(p * 0.95, lbl),
        enable_loop_closure=enable_loop_closure)
    on_progress(0.96, "Aligning to ground truth...")
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

    times = np.array([(f.timestamp_ns - first_timestamp_ns) / 1e9 for f in kf_frames])

    # Per-keyframe position error vs the nearest ground-truth sample [m]. Poses are already
    # anchored to GT at closest_cam_index (T_comp), so this is a single-point-aligned error,
    # consistent with how positions are overlaid against GT in the view.
    gt_timestamps_ns = np.array([s.timestamp_ns for s in data.ground_truth_samples])
    gt_positions = np.array([s.position for s in data.ground_truth_samples])
    gt_rotation_matrices_all = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in data.ground_truth_samples])
    nearest_gt = _nearest_sorted_indices(gt_timestamps_ns, cam_timestamps_ns)
    position_errors = np.linalg.norm(world_T_body_poses[:, :3, 3] - gt_positions[nearest_gt], axis=1)

    # ATE / RPE use the *raw*, un-anchored poses (before T_comp): T_comp is a full 6-DOF
    # transform derived from one GT sample's quaternion, so it can rotate the "up" axis away from
    # true gravity-up -- the raw poses are still exactly the gravity-aligned nav frame _run_gtsam
    # built, which _yaw_translation_align's roll/pitch-observable assumption depends on. RPE would
    # be unaffected either way (a constant transform cancels out of a relative pose), but ATE's
    # yaw-only alignment would not be consistent applied to the already-anchored poses.
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
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    optical_flow_result: OpticalFlowResult,
    set_progress: Callable[[float, str], None],
    enable_loop_closure: bool = False,
) -> SlamResult:
    first_timestamp_ns = data.cam_timestamps_ns[0]
    # SLAM runs on the frames sliced to the config's [start_s, start_s + duration_s] window,
    # so the first stereo frame marks the window start. Trimming GT/IMU/extra to the same
    # lower bound makes every series' time axis begin at start_s (times stay relative to the
    # dataset start), matching the PnP/GTSAM series that are already windowed.
    min_timestamp_ns = stereo_matching_result.frames[0].timestamp_ns
    max_timestamp_ns = stereo_matching_result.frames[-1].timestamp_ns

    gravity = np.array([0.0, 0.0, -9.81])

    # Progress budget across the whole solver so every step advances the bar:
    #   ground truth [0.00, 0.03]  IMU [0.03, 0.06]  PnP [0.06, 0.40]
    #   GTSAM [0.40, 0.97]  extra/finishing [0.97, 1.00]
    set_progress(0.0, "Loading ground truth...")
    gt_result = _get_ground_truth_result(data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns)

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

    # Single per-frame PnP scan shared by the PnP diagnostic and GTSAM node selection: it returns
    # both the keyframe indices (GTSAM nodes) and the per-frame cam0 dead-reckoning trajectory,
    # so the expensive ref->frame ORB matching happens once instead of twice.
    set_progress(0.06, "Selecting keyframes...")
    pnp_t0 = time.monotonic()
    keyframe_indices, pnp_poses = _scan_keyframes(
        data, feature_detection_result, stereo_matching_result, len(stereo_matching_result.frames),
        on_progress=lambda p: set_progress(0.06 + p * (0.45 - 0.06), "Selecting keyframes..."),
    )
    pnp_result = _get_pnp_result(
        data, stereo_matching_result, first_timestamp_ns, min_timestamp_ns, pnp_poses)
    pnp_result.elapsed_time = time.monotonic() - pnp_t0

    gtsam_t0 = time.monotonic()
    gtsam_result = _get_gtsam_result(
        data, feature_detection_result, stereo_matching_result, optical_flow_result,
        first_timestamp_ns, min_timestamp_ns, max_timestamp_ns,
        gravity=gravity, keyframe_indices=keyframe_indices,
        on_progress=lambda p, lbl: set_progress(0.45 + p * (0.97 - 0.45), lbl),
        enable_loop_closure=enable_loop_closure,
    )
    gtsam_result.elapsed_time = time.monotonic() - gtsam_t0

    set_progress(0.97, "Finishing...")
    extra_result = _get_extra_result(data, gt_result, imu_result, min_timestamp_ns, max_timestamp_ns, gravity)
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
    def __init__(
        self, data: EuRoCMAVData, feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult, optical_flow_result: OpticalFlowResult,
        cancel_event: Optional[threading.Event] = None,
        # Off by default: _find_loop_closures is O(K^2) brute-force descriptor matching (fine for
        # an offline research run, not yet acceptable as an always-on interactive-tool cost), and
        # its candidate-consolidation step is validated-but-not-tuned (see
        # _consolidate_loop_closure_clusters' docstring) -- clear win on one regression-check
        # sequence, real if smaller localized regression on another. Opt in explicitly.
        enable_loop_closure: bool = False,
    ) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self._stereo_matching_result = stereo_matching_result
        self._optical_flow_result = optical_flow_result
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self._enable_loop_closure = enable_loop_closure
        self.result: Optional[SlamResult] = None
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

        # Runs on the caller's background thread (see SlamViewModel.start), sharing this
        # process's memory: no spawn, no reimport, no pickling data across a process boundary.
        try:
            self.result = _compute(
                self._data, self._feature_detection_result, self._stereo_matching_result,
                self._optical_flow_result, set_progress,
                enable_loop_closure=self._enable_loop_closure)
        except _SolveCancelled:
            pass
        except Exception:
            self.error = traceback.format_exc()
        finally:
            self.loading = False
