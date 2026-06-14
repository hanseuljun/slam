import copyreg
from dataclasses import dataclass
import multiprocessing as mp
import traceback
from typing import Any, Callable, Optional

import cv2
import gtsam
import numpy as np

from slam.data import EuRoCMAVData
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
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamImuResult:
    times: np.ndarray
    attitudes: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamPnpResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamGtsamResult:
    times: np.ndarray
    positions: np.ndarray
    attitudes: np.ndarray
    angular_velocity_times: np.ndarray
    angular_velocities: np.ndarray


@dataclass
class SlamResult:
    gt: SlamGroundTruthResult
    imu: SlamImuResult
    pnp: SlamPnpResult
    gtsam: SlamGtsamResult



def _get_ground_truth_result(
    data: EuRoCMAVData,
    first_ts: int,
    max_ts: int,
) -> SlamGroundTruthResult:
    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_ts]
    gt_times = np.array([(s.timestamp_ns - first_ts) / 1e9 for s in gt_samples])
    gt_positions = np.array([s.position for s in gt_samples])
    gt_attitudes = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in gt_samples])

    gt_angular_velocities = []
    for j in range(len(gt_samples) - 1):
        gt_rotation = gt_attitudes[j].T @ gt_attitudes[j + 1]
        gt_rotation_vector, _ = cv2.Rodrigues(gt_rotation)
        dt = (gt_samples[j + 1].timestamp_ns - gt_samples[j].timestamp_ns) / 1e9
        gt_angular_velocity = gt_rotation_vector.flatten() / dt
        gt_angular_velocities.append(gt_angular_velocity)
    gt_angular_velocities = np.array(gt_angular_velocities)

    return SlamGroundTruthResult(
        times=gt_times,
        positions=gt_positions,
        attitudes=gt_attitudes,
        angular_velocity_times=np.array([(s.timestamp_ns - first_ts) / 1e9 for s in gt_samples[:-1]]),
        angular_velocities=gt_angular_velocities,
    )


def _get_imu_result(
    data: EuRoCMAVData,
    first_ts: int,
    max_ts: int,
    first_gt_sample_timestamp_ns: int,
    gt_attitudes: np.ndarray,
) -> tuple[SlamImuResult, list]:
    imu_samples = [s for s in data.imu_samples if s.timestamp_ns <= max_ts]
    imu_times = np.array([(s.timestamp_ns - first_ts) / 1e9 for s in imu_samples])
    imu_angular_velocities = np.array([s.angular_velocity for s in imu_samples])
    imu_attitudes = []
    prev_attitude = np.eye(3)
    for angular_velocity in imu_angular_velocities:
        imu_rotation_matrix, _ = cv2.Rodrigues(angular_velocity / data.imu0_rate_hz)
        prev_attitude = prev_attitude @ imu_rotation_matrix
        imu_attitudes.append(prev_attitude)

    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    imu_index_closest_to_first_gt_sample = np.argmin(np.abs(imu_timestamps_ns - first_gt_sample_timestamp_ns))
    imu_compensation_rotation_matrix = gt_attitudes[0] @ imu_attitudes[imu_index_closest_to_first_gt_sample].T
    for i in range(len(imu_attitudes)):
        imu_attitudes[i] = imu_compensation_rotation_matrix @ imu_attitudes[i]

    return SlamImuResult(
        times=imu_times,
        attitudes=np.array(imu_attitudes),
        angular_velocities=imu_angular_velocities,
    ), imu_samples


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
    first_ts: int,
    first_gt_sample,
    on_progress: Callable[[float], None],
) -> SlamPnpResult:
    pnp_poses = _run_pnp(
        data, feature_detection_result, stereo_matching_result, on_progress,
    )

    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames])
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
        (stereo_matching_result.frames[i].timestamp_ns - first_ts) / 1e9
        for i in range(len(pnp_world_T_body))
    ])

    angular_velocities = []
    for i in range(len(pnp_world_T_body) - 1):
        rotation_matrix = pnp_world_T_body[i, :3, :3].T @ pnp_world_T_body[i + 1, :3, :3]
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
        angular_velocities.append(rotation_vector.flatten() * data.cam0_rate_hz)

    return SlamPnpResult(
        times=pnp_times,
        positions=pnp_world_T_body[:, :3, 3],
        attitudes=pnp_world_T_body[:, :3, :3],
        angular_velocity_times=pnp_times[:-1],
        angular_velocities=np.array(angular_velocities),
    )


def _run_gtsam(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    imu_samples: list,
) -> list[np.ndarray]:
    N = len(stereo_matching_result.frames)
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    imu_angular_velocities = np.array([s.angular_velocity for s in imu_samples])
    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])
    nearest_imu_indices = np.array([np.argmin(np.abs(imu_timestamps_ns - ts)) for ts in cam_timestamps_ns])
    imu_angular_velocities_at_cam_times = imu_angular_velocities[nearest_imu_indices]

    PRIOR_NOISE = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
    ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 1e9, 1e9, 1e9]))
    PROJECTION_NOISE = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

    K = gtsam.Cal3_S2(
        data.cam0_intrinsics.fx, data.cam0_intrinsics.fy, 0.0,
        data.cam0_intrinsics.cx, data.cam0_intrinsics.cy,
    )

    isam2 = gtsam.ISAM2(gtsam.ISAM2Params())

    frame0_factors = gtsam.NonlinearFactorGraph()
    frame0_values = gtsam.Values()
    frame0_factors.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), PRIOR_NOISE))
    frame0_values.insert(0, gtsam.Pose3())
    isam2.update(frame0_factors, frame0_values)

    landmark_id = 0
    landmark_map = {}  # (frame_i, cam0_queryIdx) -> l_key
    added_observations = set()  # (l_key, pose_key) pairs already in graph

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    for i in range(N - 1):
        estimate = isam2.calculateEstimate()
        pose_i = estimate.atPose3(i)

        rotation_vector = imu_angular_velocities_at_cam_times[i] / data.cam0_rate_hz
        delta = gtsam.Pose3(gtsam.Rot3.Expmap(rotation_vector), gtsam.Point3(0.0, 0.0, 0.0))

        new_factors = gtsam.NonlinearFactorGraph()
        new_values = gtsam.Values()

        new_factors.add(gtsam.BetweenFactorPose3(i, i + 1, delta, ODOMETRY_NOISE))
        new_values.insert(i + 1, pose_i.compose(delta))

        fd_i = feature_detection_result.frames[i]
        fd_next = feature_detection_result.frames[i + 1]
        sm_i = stereo_matching_result.frames[i]
        cam0_idx_to_3d = {m.queryIdx: k for k, m in enumerate(sm_i.matches)}

        raw_matches = bf.knnMatch(fd_i.cam0_descriptors, fd_next.cam0_descriptors, k=2)
        good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

        for m in good_matches:
            if m.queryIdx not in cam0_idx_to_3d:
                continue

            p_3d = sm_i.points_3d[:, cam0_idx_to_3d[m.queryIdx]]

            key_id = (i, m.queryIdx)
            if key_id not in landmark_map:
                l_key = gtsam.symbol('l', landmark_id)
                landmark_map[key_id] = l_key
                p_world = pose_i.transformFrom(gtsam.Point3(p_3d[0], p_3d[1], p_3d[2]))
                new_values.insert(l_key, p_world)
                landmark_id += 1

            l_key = landmark_map[key_id]

            for pose_key, kp in [(i, fd_i.cam0_keypoints[m.queryIdx].pt), (i + 1, fd_next.cam0_keypoints[m.trainIdx].pt)]:
                obs = (l_key, pose_key)
                if obs not in added_observations:
                    new_factors.add(gtsam.GenericProjectionFactorCal3_S2(
                        np.array(kp), PROJECTION_NOISE, pose_key, l_key, K
                    ))
                    added_observations.add(obs)

        isam2.update(new_factors, new_values)

    final_estimate = isam2.calculateEstimate()
    return [final_estimate.atPose3(i).matrix() for i in range(N)]


def _get_gtsam_result(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    imu_samples: list,
    first_ts: int,
    first_gt_sample,
) -> SlamGtsamResult:
    poses = _run_gtsam(data, feature_detection_result, stereo_matching_result, imu_samples)
    N = len(poses)

    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames[:N]])
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt_sample.timestamp_ns))

    body_T_cam0 = data.cam0_extrinsics
    cam0_T_body = np.linalg.inv(body_T_cam0)

    world_T_body_first = np.eye(4)
    world_T_body_first[:3, :3] = quaternion_to_rotation_matrix(first_gt_sample.quaternion)
    world_T_body_first[:3, 3] = first_gt_sample.position

    gtsam_body_poses = [body_T_cam0 @ T @ cam0_T_body for T in poses]
    T_comp = world_T_body_first @ np.linalg.inv(gtsam_body_poses[closest_cam_index])
    world_T_body_poses = np.array([T_comp @ T for T in gtsam_body_poses])

    times = np.array([(f.timestamp_ns - first_ts) / 1e9 for f in stereo_matching_result.frames[:N]])

    angular_velocities = []
    for i in range(N - 1):
        R_rel = world_T_body_poses[i, :3, :3].T @ world_T_body_poses[i + 1, :3, :3]
        rvec, _ = cv2.Rodrigues(R_rel)
        angular_velocities.append(rvec.flatten() * data.cam0_rate_hz)

    return SlamGtsamResult(
        times=times,
        positions=world_T_body_poses[:, :3, 3],
        attitudes=world_T_body_poses[:, :3, :3],
        angular_velocity_times=times[:-1],
        angular_velocities=np.array(angular_velocities),
    )


def _compute(
    data: EuRoCMAVData,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    set_progress: Callable[[float, str], None],
) -> SlamResult:
    first_ts = data.cam_timestamps_ns[0]
    max_ts = stereo_matching_result.frames[-1].timestamp_ns
    first_gt_sample = data.ground_truth_samples[0]

    gt_result = _get_ground_truth_result(data, first_ts, max_ts)

    imu_result, imu_samples = _get_imu_result(data, first_ts, max_ts, first_gt_sample.timestamp_ns, gt_result.attitudes)

    set_progress(0.0, "Running PnP...")
    pnp_result = _get_pnp_result(
        data, feature_detection_result, stereo_matching_result, first_ts, first_gt_sample,
        on_progress=lambda p: set_progress(p / 4.0, "Running PnP..."),
    )

    set_progress(2.0 / 4.0, "Running GTSAM optimization...")
    gtsam_result = _get_gtsam_result(
        data, feature_detection_result, stereo_matching_result, imu_samples, first_ts, first_gt_sample,
    )

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
