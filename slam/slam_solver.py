import threading
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from slam.data import DataFolder
from slam.feature_detection import FeatureDetectionResult
from slam.solve import solve_stereo_pnp
from slam.stereo_matching import StereoMatchingResult


def _quaternion_to_rotation_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _optimize_angular_velocities(
    imu_angular_velocities_at_cam_times: np.ndarray,
    pnp_attitudes: np.ndarray,
    cam_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    return result.x.reshape(N - 1, 3), accumulate(result.x)


class _Cancelled(Exception):
    pass


def _run_pnp(
    data: DataFolder,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    on_progress: Callable[[float], None],
    stop_event: threading.Event,
) -> tuple[np.ndarray, np.ndarray]:
    keyframe_indices: list[int] = [0]
    keyframe_num_temporal_matches: Optional[int] = None
    pnp_poses: list[np.ndarray] = [np.eye(4)]
    pnp_angular_velocities: list[np.ndarray] = []
    n = len(stereo_matching_result.frames)
    for i in range(1, n):
        if stop_event.is_set():
            raise _Cancelled
        on_progress(i / n)
        try:
            kf_idx = keyframe_indices[-1]
            kf_fd = feature_detection_result.frames[kf_idx]
            kf_sm = stereo_matching_result.frames[kf_idx]
            cf_fd = feature_detection_result.frames[i]
            rvec, tvec, num_temporal_matches, _ = solve_stereo_pnp(
                data,
                kf_fd.cam0_descriptors,
                kf_sm.matches, kf_sm.points_3d,
                cf_fd.cam0_keypoints, cf_fd.cam0_descriptors,
            )
        except Exception as e:
            print(f"solve_stereo_pnp failed at i={i}: {e}")
            continue
        if keyframe_num_temporal_matches is None:
            keyframe_num_temporal_matches = num_temporal_matches
        M = np.linalg.inv(data.cam0_extrinsics)
        rvec = M[:3, :3] @ rvec
        tvec = M[:3, :3] @ tvec
        pnp_angular_velocities.append(rvec.flatten() * data.cam0_rate_hz)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        pnp_poses.append(pnp_poses[keyframe_indices[-1]] @ T)
        if num_temporal_matches < keyframe_num_temporal_matches / 2:
            keyframe_indices.append(i)
            keyframe_num_temporal_matches = None
    return np.array(pnp_poses), np.array(pnp_angular_velocities)


@dataclass
class SlamResults:
    pnp_times: np.ndarray
    pnp_positions: np.ndarray
    pnp_attitudes: np.ndarray
    pnp_angular_velocity_times: np.ndarray
    pnp_angular_velocities: np.ndarray
    imu_attitude_times: np.ndarray
    imu_attitudes: np.ndarray
    imu_times: np.ndarray
    imu_angular_velocities: np.ndarray
    imu_angular_velocities_at_cam_times: np.ndarray
    gt_times: np.ndarray
    gt_positions: np.ndarray
    gt_attitudes: np.ndarray
    gt_angular_velocity_times: np.ndarray
    gt_angular_velocities: np.ndarray
    optimized_attitudes: np.ndarray
    optimized_angular_velocities: np.ndarray


def _compute_plots(
    data: DataFolder,
    feature_detection_result: FeatureDetectionResult,
    stereo_matching_result: StereoMatchingResult,
    set_progress: Callable[[float, str], None],
    stop_event: threading.Event,
) -> SlamResults:
    first_ts = data.cam_timestamps_ns[0]
    max_ts = stereo_matching_result.frames[-1].timestamp_ns

    set_progress(0.0, "Running PnP...")
    pnp_poses_without_initial, pnp_angular_velocities_from_rvec = _run_pnp(
        data, feature_detection_result, stereo_matching_result,
        on_progress=lambda p: set_progress(p * 0.8, "Running PnP..."),
        stop_event=stop_event,
    )

    set_progress(0.8, "Transforming poses...")
    first_gt = data.ground_truth_samples[0]
    first_gt_pose = np.eye(4)
    first_gt_pose[:3, :3] = _quaternion_to_rotation_matrix(first_gt.quaternion)
    first_gt_pose[:3, 3] = first_gt.position

    cam_timestamps_ns = np.array([f.timestamp_ns for f in stereo_matching_result.frames])
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt.timestamp_ns))

    pnp_poses_in_world = np.array([
        first_gt_pose @ data.leica_extrinsics
        @ np.linalg.inv(data.cam0_extrinsics @ pnp_poses_without_initial[closest_cam_index])
        @ data.cam0_extrinsics @ T
        @ np.linalg.inv(data.leica_extrinsics)
        for T in pnp_poses_without_initial
    ])

    pnp_times = np.array([
        (stereo_matching_result.frames[i].timestamp_ns - first_ts) / 1e9
        for i in range(len(pnp_poses_in_world))
    ])
    pnp_positions_in_world = pnp_poses_in_world[:, :3, 3]
    pnp_attitudes = pnp_poses_in_world[:, :3, :3]
    pnp_angular_velocity_times = np.array([
        (stereo_matching_result.frames[i].timestamp_ns - first_ts) / 1e9
        for i in range(len(pnp_angular_velocities_from_rvec))
    ])

    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_ts]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - first_ts) / 1e9 for s in gt_samples])
    gt_attitudes = np.array([_quaternion_to_rotation_matrix(s.quaternion) for s in gt_samples])

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
    pnp_cam_timestamps_ns = np.array([
        stereo_matching_result.frames[i].timestamp_ns
        for i in range(len(pnp_poses_without_initial))
    ])
    nearest_imu_indices = np.array([
        np.argmin(np.abs(imu_timestamps_ns - ts)) for ts in pnp_cam_timestamps_ns
    ])
    imu_angular_velocities_at_cam_times = imu_angular_velocities[nearest_imu_indices]

    set_progress(0.85, "Optimizing angular velocities...")
    optimized_angular_velocities, optimized_attitudes = _optimize_angular_velocities(
        imu_angular_velocities_at_cam_times, pnp_attitudes, data.cam0_rate_hz
    )

    closest_imu_index = np.argmin(np.abs(imu_timestamps_ns - first_gt.timestamp_ns))
    R_gt = first_gt_pose[:3, :3]
    R_leica = data.leica_extrinsics[:3, :3]
    imu_attitudes_in_world = np.array([
        R_gt @ R_leica @ imu_attitudes[closest_imu_index + 1].T @ att @ R_leica.T
        for att in imu_attitudes
    ])

    gt_angular_velocities = []
    gt_angular_velocity_times = []
    for j in range(len(gt_samples) - 1):
        R0 = _quaternion_to_rotation_matrix(gt_samples[j].quaternion)
        R1 = _quaternion_to_rotation_matrix(gt_samples[j + 1].quaternion)
        R_rel = R0.T @ R1
        rvec_gt, _ = cv2.Rodrigues(R_rel)
        dt = (gt_samples[j + 1].timestamp_ns - gt_samples[j].timestamp_ns) / 1e9
        gt_angular_velocities.append(rvec_gt.flatten() / dt)
        gt_angular_velocity_times.append((gt_samples[j].timestamp_ns - first_ts) / 1e9)
    gt_angular_velocities = np.array(gt_angular_velocities)
    gt_angular_velocity_times = np.array(gt_angular_velocity_times)

    set_progress(0.95, "Finishing...")
    return SlamResults(
        pnp_times=pnp_times,
        pnp_positions=pnp_positions_in_world,
        pnp_attitudes=pnp_attitudes,
        pnp_angular_velocity_times=pnp_angular_velocity_times,
        pnp_angular_velocities=pnp_angular_velocities_from_rvec,
        imu_attitude_times=imu_attitude_times,
        imu_attitudes=imu_attitudes_in_world,
        imu_times=imu_times,
        imu_angular_velocities=imu_angular_velocities,
        imu_angular_velocities_at_cam_times=imu_angular_velocities_at_cam_times,
        gt_times=gt_times,
        gt_positions=gt_positions,
        gt_attitudes=gt_attitudes,
        gt_angular_velocity_times=gt_angular_velocity_times,
        gt_angular_velocities=gt_angular_velocities,
        optimized_attitudes=optimized_attitudes,
        optimized_angular_velocities=optimized_angular_velocities,
    )


class SlamSolver:
    def __init__(self, data: DataFolder, feature_detection_result: FeatureDetectionResult, stereo_matching_result: StereoMatchingResult) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self._stereo_matching_result = stereo_matching_result
        self.plots: Optional[SlamResults] = None
        self.loading: bool = False
        self.error: Optional[str] = None
        self.progress: float = 0.0
        self.progress_label: str = ""
        self._started: bool = False
        self._stop_event: threading.Event = threading.Event()

    def _set_progress(self, value: float, label: str) -> None:
        self.progress = value
        self.progress_label = label

    def _compute(self, stop_event: threading.Event) -> None:
        try:
            self.plots = _compute_plots(self._data, self._feature_detection_result, self._stereo_matching_result, self._set_progress, stop_event)
        except _Cancelled:
            pass
        except Exception as e:
            self.error = str(e)
        finally:
            self.loading = False

    def _start_thread(self) -> None:
        self._stop_event = threading.Event()
        threading.Thread(target=self._compute, args=(self._stop_event,), daemon=True).start()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.loading = True
        self._start_thread()
