from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import gtsam
import numpy as np
from scipy.optimize import least_squares

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult
from slam.stereo_matching import StereoMatchingResult
from slam.util import quaternion_to_rotation_matrix


@dataclass
class SlamGroundTruthResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamImuResult:
    attitude_times: np.ndarray
    attitudes: np.ndarray
    times: np.ndarray
    angular_velocities: np.ndarray
    angular_velocities_at_cam_times: np.ndarray


@dataclass
class SlamPnpResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamScipyResult:
    attitudes: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamGtsamResult:
    attitudes: np.ndarray  # (N, 3, 3)
    positions: np.ndarray  # (N, 3)


@dataclass
class SlamResult:
    gt: SlamGroundTruthResult
    imu: SlamImuResult
    pnp: SlamPnpResult
    scipy: SlamScipyResult
    gtsam: SlamGtsamResult



def _solve_pnp(
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

    K0 = data.cam0_intrinsics.to_matrix()
    dist_coeffs = np.array([
        data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1, data.cam0_intrinsics.p2,
    ])

    success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, K0, dist_coeffs)
    if not success:
        raise RuntimeError(f"cv2.solvePnPRansac failed. len(object_points): {len(object_points)}")

    inlier_object_points = object_points[inliers.flatten()]
    inlier_image_points = image_points[inliers.flatten()]
    projected, _ = cv2.projectPoints(inlier_object_points, rvec, tvec, K0, dist_coeffs)
    reprojection_error = np.mean(np.linalg.norm(inlier_image_points - projected.reshape(-1, 2), axis=1))

    return rvec, tvec, len(temporal_good_matches), reprojection_error


def _run_pnp(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    on_progress: Callable[[float], None],
) -> tuple[np.ndarray, np.ndarray]:
    keyframe_indices: list[int] = [0]
    keyframe_num_temporal_matches: Optional[int] = None
    pnp_poses: list[np.ndarray] = [np.eye(4)]
    pnp_angular_velocities: list[np.ndarray] = []
    n = len(stereo_matching_result.frames)
    for i in range(1, n):
        on_progress(i / n)
        try:
            kf_idx = keyframe_indices[-1]
            kf_fd = feature_detection_result.frames[kf_idx]
            kf_sm = stereo_matching_result.frames[kf_idx]
            cf_fd = feature_detection_result.frames[i]
            rvec, tvec, num_temporal_matches, _ = _solve_pnp(
                data,
                kf_sm.points_3d, kf_sm.matches,
                kf_fd.cam0_descriptors,
                cf_fd.cam0_keypoints, cf_fd.cam0_descriptors,
            )
        except Exception as e:
            print(f"_solve_pnp failed at i={i}: {e}")
            continue
        if keyframe_num_temporal_matches is None:
            keyframe_num_temporal_matches = num_temporal_matches
        body_T_cam0 = data.cam0_extrinsics
        rvec = body_T_cam0[:3, :3] @ rvec
        tvec = body_T_cam0[:3, :3] @ tvec
        pnp_angular_velocities.append(rvec.flatten() * data.cam0_rate_hz)
        R_body, _ = cv2.Rodrigues(rvec)
        pose_relative = np.eye(4)
        pose_relative[:3, :3] = R_body
        pose_relative[:3, 3] = tvec.flatten()
        pnp_poses.append(pnp_poses[keyframe_indices[-1]] @ pose_relative)
        if num_temporal_matches < keyframe_num_temporal_matches / 2:
            keyframe_indices.append(i)
            keyframe_num_temporal_matches = None
    return np.array(pnp_poses), np.array(pnp_angular_velocities)


def _optimize_angular_velocities(
    imu_angular_velocities_at_cam_times: np.ndarray,
    pnp_attitudes: np.ndarray,
    cam_rate_hz: int,
) -> SlamScipyResult:
    N = len(pnp_attitudes)
    x0 = imu_angular_velocities_at_cam_times[:-1].flatten()

    def accumulate(x: np.ndarray) -> np.ndarray:
        omegas = x.reshape(N - 1, 3)
        attitudes = [pnp_attitudes[0]]
        for omega in omegas:
            R, _ = cv2.Rodrigues(omega / cam_rate_hz)
            attitudes.append(attitudes[-1] @ R)
        return np.array(attitudes)

    def residuals(x: np.ndarray) -> np.ndarray:
        omegas = x.reshape(N - 1, 3)
        optimized_attitudes = accumulate(x)
        omega_res = (omegas - imu_angular_velocities_at_cam_times[:-1]).flatten()
        attitude_res = (optimized_attitudes[1:] - pnp_attitudes[1:]).flatten()
        return np.concatenate([omega_res, attitude_res])

    result = least_squares(residuals, x0, method='lm')
    return SlamScipyResult(
        attitudes=accumulate(result.x),
        angular_velocities=result.x.reshape(N - 1, 3),
    )


def _optimize_with_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    pnp_poses_in_world: np.ndarray,
    imu_angular_velocities_at_cam_times: np.ndarray,
) -> SlamGtsamResult:
    from gtsam.symbol_shorthand import L, P

    N = len(pnp_poses_in_world)
    intr = data.cam0_intrinsics
    K = gtsam.Cal3_S2(intr.fx, intr.fy, 0.0, intr.cx, intr.cy)
    K_mat = intr.to_matrix()
    dist_coeffs = np.array([intr.k1, intr.k2, intr.p1, intr.p2])

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    for i, T in enumerate(pnp_poses_in_world):
        initial.insert(P(i), gtsam.Pose3(gtsam.Rot3(T[:3, :3]), T[:3, 3]))

    graph.add(gtsam.PriorFactorPose3(
        P(0), initial.atPose3(P(0)),
        gtsam.noiseModel.Isotropic.Sigma(6, 1e-3),
    ))

    # IMU between factors: constrain rotation, leave translation unconstrained
    imu_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01, 1e6, 1e6, 1e6])
    )
    for i in range(N - 1):
        R_imu, _ = cv2.Rodrigues(imu_angular_velocities_at_cam_times[i] / data.cam0_rate_hz)
        graph.add(gtsam.BetweenFactorPose3(
            P(i), P(i + 1),
            gtsam.Pose3(gtsam.Rot3(R_imu), np.zeros(3)),
            imu_noise,
        ))

    # Projection factors between temporally adjacent frames
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    proj_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    lm_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)
    lm_id = 0

    for k in range(N - 1):
        sm_k = stereo_matching_result.frames[k]
        fd_k = feature_detection_result.frames[k]
        fd_k1 = feature_detection_result.frames[k + 1]

        if len(sm_k.matches) < 2 or len(fd_k1.cam0_descriptors) < 2:
            continue

        # Descriptors of frame k's stereo-matched cam0 keypoints, ordered by sm_k.points_3d columns
        stereo_desc_k = fd_k.cam0_descriptors[[m.queryIdx for m in sm_k.matches]]
        raw_matches = bf.knnMatch(stereo_desc_k, fd_k1.cam0_descriptors, k=2)
        good = [ms[0] for ms in raw_matches if len(ms) == 2 and ms[0].distance < 0.75 * ms[1].distance]

        T_k = pnp_poses_in_world[k]
        for m in good:
            p_cam = sm_k.points_3d[:, m.queryIdx]
            p_world = T_k[:3, :3] @ p_cam + T_k[:3, 3]

            pt = np.array([[fd_k1.cam0_keypoints[m.trainIdx].pt]], dtype=np.float64)
            pt_u = cv2.undistortPoints(pt, K_mat, dist_coeffs, P=K_mat).reshape(2)

            lk = L(lm_id)
            lm_id += 1
            initial.insert(lk, p_world)
            graph.add(gtsam.PriorFactorPoint3(lk, p_world, lm_noise))
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                pt_u, proj_noise, P(k + 1), lk, K,
            ))

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    world_T_cam0_poses = np.array([result.atPose3(P(i)).matrix() for i in range(N)])
    cam0_T_body = np.linalg.inv(data.cam0_extrinsics)
    world_T_body_poses = world_T_cam0_poses @ cam0_T_body
    return SlamGtsamResult(attitudes=world_T_body_poses[:, :3, :3], positions=world_T_body_poses[:, :3, 3])


def _compute(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    set_progress: Callable[[float, str], None],
) -> SlamResult:
    first_ts = data.cam_timestamps_ns[0]
    max_ts = stereo_matching_result.frames[-1].timestamp_ns

    body_T_cam0 = data.cam0_extrinsics

    first_gt = data.ground_truth_samples[0]
    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt.quaternion)
    world_T_body_first[:3, 3] = first_gt.position
    first_gt_pose = world_T_body_first

    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_ts]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - first_ts) / 1e9 for s in gt_samples])
    gt_attitudes = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in gt_samples])

    gt_angular_velocities = []
    gt_angular_velocity_times = []
    for j in range(len(gt_samples) - 1):
        R0 = quaternion_to_rotation_matrix(gt_samples[j].quaternion)
        R1 = quaternion_to_rotation_matrix(gt_samples[j + 1].quaternion)
        R_rel = R0.T @ R1
        rvec_gt, _ = cv2.Rodrigues(R_rel)
        dt = (gt_samples[j + 1].timestamp_ns - gt_samples[j].timestamp_ns) / 1e9
        gt_angular_velocities.append(rvec_gt.flatten() / dt)
        gt_angular_velocity_times.append((gt_samples[j].timestamp_ns - first_ts) / 1e9)
    gt_angular_velocities = np.array(gt_angular_velocities)
    gt_angular_velocity_times = np.array(gt_angular_velocity_times)

    imu_samples = [s for s in data.imu_samples if s.timestamp_ns <= max_ts]
    imu_angular_velocities = np.array([s.angular_velocity for s in imu_samples])
    imu_rotations = imu_angular_velocities / data.imu0_rate_hz
    imu_attitudes = [np.eye(3)]
    for rot in imu_rotations:
        R, _ = cv2.Rodrigues(rot)
        imu_attitudes.append(imu_attitudes[-1] @ R)
    imu_attitudes = np.array(imu_attitudes)
    imu_times = np.array([(s.timestamp_ns - first_ts) / 1e9 for s in imu_samples])
    imu_attitude_times = np.concatenate([[0.0], imu_times])

    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    closest_imu_index = np.argmin(np.abs(imu_timestamps_ns - first_gt.timestamp_ns))
    R_gt = first_gt_pose[:3, :3]
    R_leica = data.leica_extrinsics[:3, :3]
    imu_attitudes_in_world = np.array([
        R_gt @ R_leica @ imu_attitudes[closest_imu_index + 1].T @ att @ R_leica.T
        for att in imu_attitudes
    ])

    set_progress(0.0, "Running PnP...")
    pnp_poses_without_initial, pnp_angular_velocities_from_rvec = _run_pnp(
        data, feature_detection_result, stereo_matching_result,
        on_progress=lambda p: set_progress(p / 4.0, "Running PnP..."),
    )

    set_progress(1.0 / 4.0, "Transforming poses...")
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames])
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt.timestamp_ns))

    # pnp_poses_in_world = np.array([
    #     first_gt_pose @ data.leica_extrinsics
    #     @ np.linalg.inv(data.cam0_extrinsics @ pnp_poses_without_initial[closest_cam_index])
    #     @ data.cam0_extrinsics @ T
    #     @ np.linalg.inv(data.leica_extrinsics)
    #     for T in pnp_poses_without_initial
    # ])

    world_T_cam0_first = world_T_body_first @ body_T_cam0
    pnp_poses_in_world = np.array([
        world_T_cam0_first @ T
        for T in pnp_poses_without_initial
    ])

    cam0_T_body = np.linalg.inv(body_T_cam0)
    pnp_world_T_body = pnp_poses_in_world @ cam0_T_body

    pnp_times = np.array([
        (stereo_matching_result.frames[i].timestamp_ns - first_ts) / 1e9
        for i in range(len(pnp_poses_in_world))
    ])
    pnp_positions_in_world = pnp_world_T_body[:, :3, 3]
    pnp_attitudes = pnp_world_T_body[:, :3, :3]
    pnp_angular_velocity_times = np.array([
        (stereo_matching_result.frames[i].timestamp_ns - first_ts) / 1e9
        for i in range(len(pnp_angular_velocities_from_rvec))
    ])

    set_progress(2.0 / 4.0, "Optimizing angular velocities...")
    pnp_cam_timestamps_ns = np.array([
        stereo_matching_result.frames[i].timestamp_ns
        for i in range(len(pnp_poses_without_initial))
    ])
    nearest_imu_indices = np.array([
        np.argmin(np.abs(imu_timestamps_ns - ts)) for ts in pnp_cam_timestamps_ns
    ])
    imu_angular_velocities_at_cam_times = imu_angular_velocities[nearest_imu_indices]
    optimization = _optimize_angular_velocities(
        imu_angular_velocities_at_cam_times, pnp_attitudes, data.cam0_rate_hz
    )

    set_progress(3.0 / 4.0, "Running GTSAM optimization...")
    gtsam_result = _optimize_with_gtsam(
        data, feature_detection_result, stereo_matching_result,
        pnp_poses_in_world, imu_angular_velocities_at_cam_times,
    )

    set_progress(0.95, "Finishing...")
    return SlamResult(
        gt=SlamGroundTruthResult(
            times=gt_times,
            positions=gt_positions,
            attitudes=gt_attitudes,
            angular_velocity_times=gt_angular_velocity_times,
            angular_velocities=gt_angular_velocities,
        ),
        imu=SlamImuResult(
            attitude_times=imu_attitude_times,
            attitudes=imu_attitudes_in_world,
            times=imu_times,
            angular_velocities=imu_angular_velocities,
            angular_velocities_at_cam_times=imu_angular_velocities_at_cam_times,
        ),
        pnp=SlamPnpResult(
            times=pnp_times,
            positions=pnp_positions_in_world,
            attitudes=pnp_attitudes,
            angular_velocity_times=pnp_angular_velocity_times,
            angular_velocities=pnp_angular_velocities_from_rvec,
        ),
        scipy=optimization,
        gtsam=gtsam_result,
    )


class SlamSolver:
    def __init__(self, data: EuRoCMAVData, feature_detection_result: FeatureDetectionResult, stereo_matching_result: StereoMatchingResult) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self._stereo_matching_result = stereo_matching_result
        self.result: Optional[SlamResult] = None
        self.loading: bool = True
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.progress_label: str = ""

    def _set_progress(self, value: float, label: str) -> None:
        self.progress = value
        self.progress_label = label

    def run(self) -> None:
        try:
            self.result = _compute(
                self._data, self._feature_detection_result,
                self._stereo_matching_result, self._set_progress,
            )
        except Exception as e:
            self.error = str(e)
        finally:
            self.loading = False
