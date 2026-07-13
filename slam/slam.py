import copyreg
from dataclasses import dataclass
import multiprocessing as mp
import time
import traceback
from typing import Any, Callable, Optional

import cv2
import gtsam
import numpy as np

from slam.data import EuRoCMAVData, ImuSample
from slam.feature_detection import FeatureDetectionResult
from slam.imu_initialization import ImuInitializationResult
from slam.stereo_matching import StereoMatchingResult
from slam.util import quaternion_to_rotation_matrix


def _pickle_keypoint(kp: cv2.KeyPoint):
    return cv2.KeyPoint, (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)


def _pickle_dmatch(dm: cv2.DMatch):
    return cv2.DMatch, (dm.queryIdx, dm.trainIdx, dm.imgIdx, dm.distance)


copyreg.pickle(cv2.KeyPoint, _pickle_keypoint)
copyreg.pickle(cv2.DMatch, _pickle_dmatch)


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

    inlier_object_points = object_points[inliers.flatten()]
    inlier_image_points = image_points[inliers.flatten()]
    projected, _ = cv2.projectPoints(inlier_object_points, rotation_vector, translation_vector, intrinsics_matrix, dist_coeffs)
    reprojection_error = np.mean(np.linalg.norm(inlier_image_points - projected.reshape(-1, 2), axis=1))

    # inversing the pose from cv2.solvePnPRansac as they are the inverse of
    # what the rest of the code expects.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rotation_vector = -rotation_vector
    translation_vector = -rotation_matrix.T @ translation_vector

    return rotation_vector, translation_vector, len(temporal_good_matches), reprojection_error


def _run_pnp(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    on_progress: Callable[[float], None],
) -> list[np.ndarray]:
    keyframe_indices: list[int] = [0]
    keyframe_num_temporal_matches: Optional[int] = None
    pnp_poses: list[np.ndarray] = [np.eye(4)]
    n = len(stereo_matching_result.frames)
    for i in range(1, n):
        on_progress(i / n)
        try:
            kf_idx = keyframe_indices[-1]
            kf_fd = feature_detection_result.frames[kf_idx]
            kf_sm = stereo_matching_result.frames[kf_idx]
            cf_fd = feature_detection_result.frames[i]
            rotation_vector, translation_vector, num_temporal_matches, _ = _run_pnp_step(
                data,
                kf_sm.points_3d, kf_sm.matches,
                kf_fd.cam0_descriptors,
                cf_fd.cam0_keypoints, cf_fd.cam0_descriptors,
            )
        except Exception as e:
            print(f"_run_pnp_step failed at i={i}: {e}")
            continue
        if keyframe_num_temporal_matches is None:
            keyframe_num_temporal_matches = num_temporal_matches
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation_vector.flatten()
        pnp_poses.append(pnp_poses[keyframe_indices[-1]] @ transform)
        if num_temporal_matches < keyframe_num_temporal_matches / 2:
            keyframe_indices.append(i)
            keyframe_num_temporal_matches = None
    return pnp_poses


def _get_pnp_result(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    on_progress: Callable[[float], None],
) -> SlamPnpResult:
    pnp_poses = _run_pnp(
        data, feature_detection_result, stereo_matching_result, on_progress,
    )

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


def _build_landmark_tracks(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    keyframe_indices: list[int],
    min_track_len: int,
    depth_min: float,
    depth_max: float,
    fund_ransac_px: float,
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """Chain stereo-matched features across keyframes into persistent landmark tracks.

    Returns {track_id: [(node, uv0_undistorted, uv1_undistorted, point_cam0), ...]} where
    ``node`` is the graph-node index (into keyframe_indices) that observed the landmark.
    Only cam0-cam1 stereo inliers are tracked, so every observation carries a metric depth
    for initialization; observations are pre-undistorted so they match a Cal3_S2 pinhole model.
    """
    K0 = data.cam0_intrinsics.to_matrix()
    K1 = data.cam1_intrinsics.to_matrix()
    dist0 = np.array([data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
                      data.cam0_intrinsics.p1, data.cam0_intrinsics.p2])
    dist1 = np.array([data.cam1_intrinsics.k1, data.cam1_intrinsics.k2,
                      data.cam1_intrinsics.p1, data.cam1_intrinsics.p2])

    # Per node: the stereo-inlier descriptors (for matching) and observation tuples.
    node_descs: list[np.ndarray] = []
    node_obs: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    for frame in keyframe_indices:
        sm = stereo_matching_result.frames[frame]
        fd = feature_detection_result.frames[frame]
        if not sm.matches:
            node_descs.append(np.zeros((0, 32), dtype=np.uint8))
            node_obs.append([])
            continue
        q_idx = [m.queryIdx for m in sm.matches]
        t_idx = [m.trainIdx for m in sm.matches]
        descs = fd.cam0_descriptors[q_idx]
        uv0 = np.array([fd.cam0_keypoints[i].pt for i in q_idx], dtype=np.float64)
        uv1 = np.array([fd.cam1_keypoints[i].pt for i in t_idx], dtype=np.float64)
        uv0u = cv2.undistortPoints(uv0.reshape(-1, 1, 2), K0, dist0, P=K0).reshape(-1, 2)
        uv1u = cv2.undistortPoints(uv1.reshape(-1, 1, 2), K1, dist1, P=K1).reshape(-1, 2)
        pts = sm.points_3d.T  # (M, 3), column i <-> sm.matches[i]
        node_descs.append(descs)
        node_obs.append([(uv0u[i], uv1u[i], pts[i]) for i in range(len(sm.matches))])

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    tracks: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    next_tid = 0
    prev_obs_to_tid: dict[int, int] = {}  # obs-index at node jj-1 -> track id
    for jj in range(len(keyframe_indices)):
        cur_obs_to_tid: dict[int, int] = {}
        if jj > 0 and len(node_descs[jj - 1]) and len(node_descs[jj]):
            raw = bf.knnMatch(node_descs[jj - 1], node_descs[jj], k=2)
            good = [pair[0] for pair in raw
                    if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
            # Geometric gate: keep only matches consistent with a two-view epipolar geometry
            # (fundamental-matrix RANSAC on the undistorted cam0 points). Descriptor + ratio
            # matching alone leaves gross mismatches that would otherwise seed bad landmarks
            # and inflate reprojection error; the epipolar constraint removes them without
            # needing any pose estimate.
            if len(good) >= 8:
                pts_prev = np.array([node_obs[jj - 1][m.queryIdx][0] for m in good])
                pts_cur = np.array([node_obs[jj][m.trainIdx][0] for m in good])
                _, mask = cv2.findFundamentalMat(
                    pts_prev, pts_cur, cv2.FM_RANSAC, fund_ransac_px, 0.99)
                if mask is not None:
                    good = [m for m, keep in zip(good, mask.ravel()) if keep]
            for m in good:
                p, c = m.queryIdx, m.trainIdx
                if c in cur_obs_to_tid:
                    continue
                tid = prev_obs_to_tid.get(p)
                if tid is None:
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = [(jj - 1, *node_obs[jj - 1][p])]
                cur_obs_to_tid[c] = tid
                tracks[tid].append((jj, *node_obs[jj][c]))
        prev_obs_to_tid = cur_obs_to_tid

    # Keep only tracks long enough to bundle-adjust and whose first (init) depth is sane.
    kept: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for tid, obs in tracks.items():
        if len(obs) < min_track_len:
            continue
        z0 = float(obs[0][3][2])
        if not (depth_min < z0 < depth_max):
            continue
        kept[tid] = obs
    return kept


def _select_keyframes(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    N: int,
) -> list[int]:
    """Adaptively choose which frames become factor-graph nodes.

    A single PnP from the current reference keyframe to frame i yields both signals we need at
    once: the temporal-match count (covisibility with the reference, which decays as we move
    away) and the relative pose (motion since the reference). We open a new keyframe when overlap
    drops below a fraction of the post-keyframe baseline, when enough translation/rotation has
    accrued, or when a hard frame cap is hit -- all clamped by a min gap so we never place
    near-duplicate nodes. Compared to a fixed stride this keeps large, well-conditioned baselines
    when the camera moves fast and avoids zero-parallax nodes when it hovers.
    """
    MIN_GAP = 3                     # never place keyframes closer than this (avoid duplicate nodes)
    MAX_GAP = 15                    # force a keyframe at least this often (bound IMU preint. drift)
    COVIS_RATIO = 0.6               # new keyframe once covisibility falls below this * baseline
    TRANS_THRESH = 0.2              # ... or once translation since the reference exceeds this [m]
    ROT_THRESH = np.deg2rad(10.0)   # ... or rotation exceeds this [rad]

    keyframes = [0]
    ref_idx = 0
    ref_covis: Optional[int] = None
    i = 1
    while i < N:
        ref_fd = feature_detection_result.frames[ref_idx]
        ref_sm = stereo_matching_result.frames[ref_idx]
        cur_fd = feature_detection_result.frames[i]
        try:
            rvec, tvec, num_matches, _ = _run_pnp_step(
                data, ref_sm.points_3d, ref_sm.matches,
                ref_fd.cam0_descriptors, cur_fd.cam0_keypoints, cur_fd.cam0_descriptors,
            )
        except Exception:
            # Overlap with the reference is gone: anchor a keyframe at the last connected frame
            # (or one past the reference if that is already frame i-1), then re-evaluate i.
            kf = max(i - 1, ref_idx + 1)
            keyframes.append(kf)
            ref_idx, ref_covis = kf, None
            i = kf + 1
            continue

        if ref_covis is None:  # first frame after a keyframe sets the covisibility baseline
            ref_covis = num_matches

        gap = i - ref_idx
        translation = float(np.linalg.norm(tvec))
        rotation = float(np.linalg.norm(rvec))

        force = gap >= MAX_GAP
        allow = gap >= MIN_GAP
        weak = num_matches < COVIS_RATIO * ref_covis
        moved = translation > TRANS_THRESH or rotation > ROT_THRESH

        if force or (allow and (weak or moved)):
            keyframes.append(i)
            ref_idx, ref_covis = i, None
        i += 1

    if keyframes[-1] != N - 1:
        keyframes.append(N - 1)
    return keyframes


def _run_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    imu_samples: list[ImuSample],
    gravity: np.ndarray,
    on_progress: Callable[[float], None],
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

    # Choose the factor-graph nodes adaptively (covisibility + motion, clamped by min/max frame
    # gaps) instead of a fixed stride, so keyframes stay well-conditioned across varying motion.
    # keyframe_indices maps graph node j -> original frame index; gaps between them now vary.
    keyframe_indices = _select_keyframes(data, feature_detection_result, stereo_matching_result, N)
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
    # Random-walk noise letting the (per-keyframe) IMU bias evolve between keyframes instead
    # of being pinned to a single constant, so real time-varying bias doesn't leak into drift.
    BIAS_BETWEEN_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
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
    FUND_RANSAC_PX  = 2.0     # epipolar (fundamental-matrix RANSAC) inlier threshold [px]
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

    tracks = _build_landmark_tracks(
        data, feature_detection_result, stereo_matching_result, keyframe_indices,
        MIN_TRACK_LEN, DEPTH_MIN, DEPTH_MAX, FUND_RANSAC_PX)
    # node -> track ids observed there; and per-interval covisibility for the PnP gate.
    nodes_to_tracks: dict[int, list[int]] = {jj: [] for jj in range(K)}
    node_seen: list[set[int]] = [set() for _ in range(K)]
    for tid, obs in tracks.items():
        for o in obs:
            nodes_to_tracks[o[0]].append(tid)
            node_seen[o[0]].add(tid)
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
                    o = next(o for o in obs if o[0] == jj)
                    _add_obs_factors(factors, tid, jj, o[1], o[2])
                    n_at_node += 1
                continue
            avail = [o for o in obs if o[0] <= jj]
            if len(avail) < MIN_TRACK_LEN:
                continue
            # Initialize the landmark in the nav frame from the first observation's stereo depth.
            first = obs[0]
            T_G_cam0 = est.atPose3(X(first[0])).matrix() @ body_T_cam0
            p_world = (T_G_cam0 @ np.append(first[3], 1.0))[:3]
            values.insert(L(tid), gtsam.Point3(*p_world))
            factors.add(gtsam.PriorFactorPoint3(L(tid), gtsam.Point3(*p_world), LM_PRIOR_NOISE))
            inserted_landmarks.add(tid)
            for o in avail:
                _add_obs_factors(factors, tid, o[0], o[1], o[2])
                if o[0] == jj:
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
        on_progress(j / (K - 1))
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
        new_factors.add(gtsam.BetweenFactorConstantBias(
            B(j), B(j + 1), gtsam.imuBias.ConstantBias(), BIAS_BETWEEN_NOISE))

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
        isam2.update(new_factors, new_values)

    print(f"landmarks: {len(inserted_landmarks)}/{len(tracks)} tracks used, "
          f"{n_proj_factors} reprojection factors")
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
        for node, uv0, uv1, _pt in tracks[tid]:
            seen = False
            for intrin, inv_Twc, uv in (
                (data.cam0_intrinsics, inv_Twc0, uv0),
                (data.cam1_intrinsics, inv_Twc1, uv1),
            ):
                pc = inv_Twc[node] @ p
                if pc[2] <= 1e-6:
                    continue
                u = intrin.fx * pc[0] / pc[2] + intrin.cx
                v = intrin.fy * pc[1] / pc[2] + intrin.cy
                sq_px[node] += (u - uv[0]) ** 2 + (v - uv[1]) ** 2
                n_px[node] += 1
                seen = True
            if seen:
                n_lm[node] += 1
    reprojection_rmse = np.where(n_px > 0, np.sqrt(sq_px / np.maximum(n_px, 1)), np.nan)

    return poses, velocities, biases, reprojection_rmse, n_lm, keyframe_indices


def _get_gtsam_result(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    min_timestamp_ns: int,
    max_timestamp_ns: int,
    gravity: np.ndarray,
    on_progress: Callable[[float], None],
) -> SlamGtsamResult:
    # Window the IMU to [min, max]; only this range is ever integrated. Leaving the whole t=0
    # prefix in (as `<= max` alone did) needlessly grows the array -- and, at larger start_s,
    # pushes it past NumPy's temporary-elision size threshold (see the read-only guard in
    # _run_gtsam), so keeping it windowed is both cheaper and safer.
    imu_samples = [s for s in data.imu_samples if min_timestamp_ns <= s.timestamp_ns <= max_timestamp_ns]
    poses, velocities, biases, reprojection_rmse, landmark_counts, keyframe_indices = _run_gtsam(
        data, feature_detection_result, stereo_matching_result, imu_samples, gravity, on_progress)
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
    nearest_gt = np.argmin(np.abs(cam_timestamps_ns[:, None] - gt_timestamps_ns[None, :]), axis=1)
    position_errors = np.linalg.norm(world_T_body_poses[:, :3, 3] - gt_positions[nearest_gt], axis=1)

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
    closest_gt_indices = np.argmin(np.abs(gt_timestamps_ns[:, None] - imu_timestamps_ns[None, :]), axis=0)
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
    set_progress: Callable[[float, str], None],
) -> SlamResult:
    first_timestamp_ns = data.cam_timestamps_ns[0]
    # SLAM runs on the frames sliced to the config's [start_s, start_s + duration_s] window,
    # so the first stereo frame marks the window start. Trimming GT/IMU/extra to the same
    # lower bound makes every series' time axis begin at start_s (times stay relative to the
    # dataset start), matching the PnP/GTSAM series that are already windowed.
    min_timestamp_ns = stereo_matching_result.frames[0].timestamp_ns
    max_timestamp_ns = stereo_matching_result.frames[-1].timestamp_ns

    gravity = np.array([0.0, 0.0, -9.81])

    gt_result = _get_ground_truth_result(data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns)

    # Anchor the IMU-integrated orientation to GT at the first GT sample inside the window
    # (i.e. gt_result's first sample, which is gt_rotation_matrices[0]).
    first_gt_timestamp_ns = next(
        s.timestamp_ns for s in data.ground_truth_samples if s.timestamp_ns >= min_timestamp_ns
    )
    imu_result = _get_imu_result(
        data, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns,
        gt_result.rotation_matrices, first_gt_timestamp_ns,
    )

    set_progress(0.0, "Running PnP...")
    pnp_t0 = time.monotonic()
    pnp_result = _get_pnp_result(
        data, feature_detection_result, stereo_matching_result, first_timestamp_ns, min_timestamp_ns,
        on_progress=lambda p: set_progress(p / 2.0, "Running PnP..."),
    )
    pnp_result.elapsed_time = time.monotonic() - pnp_t0

    set_progress(2.0 / 4.0, "Running GTSAM optimization...")
    gtsam_t0 = time.monotonic()
    gtsam_result = _get_gtsam_result(
        data, feature_detection_result, stereo_matching_result, first_timestamp_ns, min_timestamp_ns, max_timestamp_ns,
        gravity=gravity,
        on_progress=lambda p: set_progress(2.0 / 4.0 + p * (0.95 - 2.0 / 4.0), "Running GTSAM optimization..."),
    )
    gtsam_result.elapsed_time = time.monotonic() - gtsam_t0

    set_progress(0.95, "Finishing...")
    extra_result = _get_extra_result(data, gt_result, imu_result, min_timestamp_ns, max_timestamp_ns, gravity)
    return SlamResult(
        gt=gt_result,
        imu=imu_result,
        pnp=pnp_result,
        gtsam=gtsam_result,
        extra=extra_result,
    )


def _worker_entry(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    progress_val: Any,
    label_arr: Any,
    result_queue: Any,
) -> None:
    def set_progress(value: float, label: str) -> None:
        progress_val.value = value
        label_arr.value = label.encode('utf-8')[:255]

    try:
        result = _compute(data, feature_detection_result, stereo_matching_result, set_progress)
        result_queue.put(('ok', result))
    except Exception:
        result_queue.put(('err', traceback.format_exc()))


class SlamSolver:
    def __init__(self, data: EuRoCMAVData, feature_detection_result: FeatureDetectionResult, stereo_matching_result: StereoMatchingResult) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self._stereo_matching_result = stereo_matching_result
        self.result: Optional[SlamResult] = None
        self.loading: bool = True
        self.error: Optional[str] = None
        self._progress_val = mp.Value('d', 0.0)
        self._label_arr = mp.Array('c', 256)
        self._result_queue = mp.Queue()

    @property
    def progress(self) -> float:
        return self._progress_val.value

    @property
    def progress_label(self) -> str:
        return self._label_arr.value.decode('utf-8', errors='replace')

    def run(self) -> None:
        process = mp.Process(
            target=_worker_entry,
            args=(
                self._data, self._feature_detection_result, self._stereo_matching_result,
                self._progress_val, self._label_arr, self._result_queue,
            ),
            daemon=True,
        )
        process.start()
        try:
            kind, value = self._result_queue.get()
            if kind == 'ok':
                self.result = value
            else:
                self.error = value
        except Exception as e:
            self.error = str(e)
        finally:
            self.loading = False
