from dataclasses import dataclass

import numpy as np

from slam.data import EuRoCMAVData
from slam.util import quaternion_to_rotation_matrix


@dataclass
class ImuInitializationResult:
    times: np.ndarray
    lin_acc_norms: np.ndarray
    ang_vel_norms: np.ndarray
    static_start_s: float
    static_end_s: float
    gravity_in_body: np.ndarray   # shape (3,)
    gravity_in_world: np.ndarray  # shape (3,)


class ImuInitializationSolver:
    def __init__(self, data: EuRoCMAVData, window_s: float = 1.0) -> None:
        self._data = data
        self._window_s = window_s

    def run(self) -> ImuInitializationResult:
        samples = self._data.imu_samples
        first_ts_ns = samples[0].timestamp_ns
        timestamps_ns = np.array([s.timestamp_ns for s in samples])
        times = (timestamps_ns - first_ts_ns) / 1e9

        lin_accs = np.array([s.linear_acceleration for s in samples])
        ang_vels = np.array([s.angular_velocity for s in samples])

        lin_acc_norms = np.linalg.norm(lin_accs, axis=1)
        ang_vel_norms = np.linalg.norm(ang_vels, axis=1)

        window = int(self._data.imu0_rate_hz * self._window_s)
        n = len(samples)

        best_score = np.inf
        best_start = 0
        for i in range(n - window):
            score = np.var(ang_vel_norms[i:i + window]) + np.var(lin_acc_norms[i:i + window])
            if score < best_score:
                best_score = score
                best_start = i
        best_end = best_start + window

        gravity_in_body = np.mean(lin_accs[best_start:best_end], axis=0)

        gt_timestamps_ns = np.array([s.timestamp_ns for s in self._data.ground_truth_samples])
        closest_gt_idx = int(np.argmin(np.abs(gt_timestamps_ns - timestamps_ns[best_start])))
        gt_sample = self._data.ground_truth_samples[closest_gt_idx]
        R_world_body = quaternion_to_rotation_matrix(gt_sample.quaternion)
        gravity_in_world = R_world_body @ gravity_in_body

        return ImuInitializationResult(
            times=times,
            lin_acc_norms=lin_acc_norms,
            ang_vel_norms=ang_vel_norms,
            static_start_s=float(times[best_start]),
            static_end_s=float(times[best_end]),
            gravity_in_body=gravity_in_body,
            gravity_in_world=gravity_in_world,
        )
