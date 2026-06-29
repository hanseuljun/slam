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
    gravities: np.ndarray


@dataclass
class SlamImuResult:
    times: np.ndarray
    attitudes: np.ndarray
    rotation_matrices: np.ndarray
    angular_velocities: np.ndarray
    linear_accelerations: np.ndarray
    linear_accelerations_in_world: np.ndarray


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
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray
    linear_accelerations: np.ndarray
    elapsed_time: float = 0.0


@dataclass
class SlamResult:
    gt: SlamGroundTruthResult
    imu: SlamImuResult
    pnp: SlamPnpResult
    gtsam: SlamGtsamResult



def _mats_to_rvecs(rotation_matrices: np.ndarray) -> np.ndarray:
    return np.array([cv2.Rodrigues(R)[0].flatten() for R in rotation_matrices])


def _get_ground_truth_result(
    data: EuRoCMAVData,
    first_timestamp_ns: int,
    max_timestamp_ns: int,
) -> SlamGroundTruthResult:
    samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_timestamp_ns]
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

    g_world = np.array([-9.81, 0.0, 0.0])
    gravities = rotation_matrices.transpose(0, 2, 1) @ g_world

    return SlamGroundTruthResult(
        times=times,
        positions=positions,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocity_times=np.array([(s.timestamp_ns - first_timestamp_ns) / 1e9 for s in samples[:-1]]),
        angular_velocities=angular_velocities,
        gravities=gravities,
    )


def _get_imu_result(
    data: EuRoCMAVData,
    first_timestamp_ns: int,
    max_timestamp_ns: int,
    gt_rotation_matrices: np.ndarray,
) -> SlamImuResult:
    samples = [s for s in data.imu_samples if s.timestamp_ns <= max_timestamp_ns]
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
    index_closest_to_first_gt_sample = np.argmin(np.abs(timestamps_ns - data.ground_truth_samples[0].timestamp_ns))
    compensation_rotation_matrix = gt_rotation_matrices[0] @ rotation_matrices_list[index_closest_to_first_gt_sample].T
    for i in range(len(rotation_matrices_list)):
        rotation_matrices_list[i] = compensation_rotation_matrix @ rotation_matrices_list[i]

    rotation_matrices = np.array(rotation_matrices_list)

    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_timestamp_ns]
    gt_timestamps_ns = np.array([s.timestamp_ns for s in gt_samples])
    closest_gt_indices = np.argmin(np.abs(gt_timestamps_ns[:, None] - timestamps_ns[None, :]), axis=0)
    linear_accelerations_in_world = np.array([
        gt_rotation_matrices[idx] @ acc
        for idx, acc in zip(closest_gt_indices, linear_accelerations)
    ])

    return SlamImuResult(
        times=times,
        attitudes=_mats_to_rvecs(rotation_matrices),
        rotation_matrices=rotation_matrices,
        angular_velocities=angular_velocities,
        linear_accelerations=linear_accelerations,
        linear_accelerations_in_world=linear_accelerations_in_world,
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
    on_progress: Callable[[float], None],
) -> SlamPnpResult:
    pnp_poses = _run_pnp(
        data, feature_detection_result, stereo_matching_result, on_progress,
    )

    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames])
    first_gt_sample = data.ground_truth_samples[0]
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


def _run_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    imu_samples: list[ImuSample],
    on_progress: Callable[[float], None],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    N = len(stereo_matching_result.frames)
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    imu_lin_accs      = np.array([s.linear_acceleration for s in imu_samples])
    imu_ang_vels      = np.array([s.angular_velocity for s in imu_samples])
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)
    cam0_R_body = cam0_T_body[:3, :3]
    imu_lin_accs = (cam0_R_body @ imu_lin_accs.T).T
    imu_ang_vels = (cam0_R_body @ imu_ang_vels.T).T

    X = lambda i: gtsam.symbol('x', i)
    V = lambda i: gtsam.symbol('v', i)
    B = gtsam.symbol('b', 0)

    imu_params = gtsam.PreintegrationParams(cam0_R_body @ np.array([-9.81, 0.0, 0.0]))
    # imu_params.setGyroscopeCovariance(np.eye(3) * 1e-4)
    imu_params.setGyroscopeCovariance(np.eye(3) * 1)
    # imu_params.setAccelerometerCovariance(np.eye(3) * 1e-3)
    imu_params.setAccelerometerCovariance(np.eye(3) * 1)
    # imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)
    imu_params.setIntegrationCovariance(np.eye(3) * 1)

    PRIOR_POSE_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    PRIOR_VEL_NOISE  = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    PRIOR_BIAS_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
    PNP_NOISE        = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.01, 0.01, 0.01, 0.05, 0.05, 0.05]))

    isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

    f0, v0 = gtsam.NonlinearFactorGraph(), gtsam.Values()
    f0.add(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(), PRIOR_POSE_NOISE))
    f0.add(gtsam.PriorFactorVector(V(0), np.zeros(3), PRIOR_VEL_NOISE))
    f0.add(gtsam.PriorFactorConstantBias(B, gtsam.imuBias.ConstantBias(), PRIOR_BIAS_NOISE))
    v0.insert(X(0), gtsam.Pose3())
    v0.insert(V(0), np.zeros(3))
    v0.insert(B, gtsam.imuBias.ConstantBias())
    isam2.update(f0, v0)

    for i in range(N - 1):
        on_progress(i / (N - 1))
        est = isam2.calculateEstimate()
        pose_i = est.atPose3(X(i))
        vel_i  = est.atVector(V(i))
        bias_i = est.atConstantBias(B)

        fd_i, fd_next = feature_detection_result.frames[i], feature_detection_result.frames[i + 1]
        sm_i = stereo_matching_result.frames[i]

        new_factors, new_values = gtsam.NonlinearFactorGraph(), gtsam.Values()

        pim = gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias())
        window = np.where(
            (imu_timestamps_ns >= cam_timestamps_ns[i]) &
            (imu_timestamps_ns <  cam_timestamps_ns[i + 1])
        )[0]
        for k in window:
            dt = (float(imu_timestamps_ns[k + 1] - imu_timestamps_ns[k]) * 1e-9
                  if k + 1 < len(imu_timestamps_ns) else 1.0 / data.imu0_rate_hz)
            pim.integrateMeasurement(imu_lin_accs[k], imu_ang_vels[k], dt)

        new_factors.add(gtsam.ImuFactor(X(i), V(i), X(i + 1), V(i + 1), B, pim))

        nav_j     = pim.predict(gtsam.NavState(pose_i, vel_i), bias_i)
        pose_init = nav_j.pose()
        vel_init  = nav_j.velocity()

        try:
            pnp_rotation_vector, pnp_translation_vector, _, _ = _run_pnp_step(
                data, sm_i.points_3d, sm_i.matches,
                fd_i.cam0_descriptors, fd_next.cam0_keypoints, fd_next.cam0_descriptors,
            )
            pnp_rotation_matrix, _ = cv2.Rodrigues(pnp_rotation_vector)
            pnp_delta = gtsam.Pose3(gtsam.Rot3(pnp_rotation_matrix), gtsam.Point3(*pnp_translation_vector.flatten()))
            new_factors.add(gtsam.BetweenFactorPose3(X(i), X(i + 1), pnp_delta, PNP_NOISE))
            pose_init = pose_i.compose(pnp_delta)
        except Exception:
            pass

        new_values.insert(X(i + 1), pose_init)
        new_values.insert(V(i + 1), vel_init)
        isam2.update(new_factors, new_values)

    final = isam2.calculateEstimate()
    poses = [final.atPose3(X(i)).matrix() for i in range(N)]
    velocities = [final.atVector(V(i)) for i in range(N)]
    return poses, velocities


def _get_gtsam_result(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    first_timestamp_ns: int,
    max_timestamp_ns: int,
    on_progress: Callable[[float], None],
) -> SlamGtsamResult:
    imu_samples = [s for s in data.imu_samples if s.timestamp_ns <= max_timestamp_ns]
    poses, velocities = _run_gtsam(data, feature_detection_result, stereo_matching_result, imu_samples, on_progress)
    N = len(poses)

    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])
    first_gt_sample = data.ground_truth_samples[0]
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt_sample.timestamp_ns))

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt_sample.quaternion)
    world_T_body_first[:3, 3] = first_gt_sample.position

    gtsam_body_poses = [body_T_cam0 @ T @ cam0_T_body for T in poses]
    T_comp = world_T_body_first @ np.linalg.inv(gtsam_body_poses[closest_cam_index])
    world_T_body_poses = np.array([T_comp @ T for T in gtsam_body_poses])

    times = np.array([(f.timestamp_ns - first_timestamp_ns) / 1e9 for f in stereo_matching_result.frames[:N]])

    angular_velocities = []
    linear_accelerations = []
    velocities_np = np.array(velocities)
    for i in range(N - 1):
        rotation_matrix = world_T_body_poses[i, :3, :3].T @ world_T_body_poses[i + 1, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        angular_velocities.append(rotation_vector.flatten() * data.cam0_rate_hz)

        dt = times[i + 1] - times[i]
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
        angular_velocity_times=times[:-1],
        angular_velocities=np.array(angular_velocities),
        linear_accelerations=np.array(linear_accelerations),
    )


def _compute(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    set_progress: Callable[[float, str], None],
) -> SlamResult:
    first_timestamp_ns = data.cam_timestamps_ns[0]
    max_timestamp_ns = stereo_matching_result.frames[-1].timestamp_ns

    gt_result = _get_ground_truth_result(data, first_timestamp_ns, max_timestamp_ns)

    imu_result = _get_imu_result(data, first_timestamp_ns, max_timestamp_ns, gt_result.rotation_matrices)

    set_progress(0.0, "Running PnP...")
    pnp_t0 = time.monotonic()
    pnp_result = _get_pnp_result(
        data, feature_detection_result, stereo_matching_result, first_timestamp_ns,
        on_progress=lambda p: set_progress(p / 2.0, "Running PnP..."),
    )
    pnp_result.elapsed_time = time.monotonic() - pnp_t0

    set_progress(2.0 / 4.0, "Running GTSAM optimization...")
    gtsam_t0 = time.monotonic()
    gtsam_result = _get_gtsam_result(
        data, feature_detection_result, stereo_matching_result, first_timestamp_ns, max_timestamp_ns,
        on_progress=lambda p: set_progress(2.0 / 4.0 + p * (0.95 - 2.0 / 4.0), "Running GTSAM optimization..."),
    )
    gtsam_result.elapsed_time = time.monotonic() - gtsam_t0

    set_progress(0.95, "Finishing...")
    return SlamResult(
        gt=gt_result,
        imu=imu_result,
        pnp=pnp_result,
        gtsam=gtsam_result,
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
