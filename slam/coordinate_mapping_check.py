from dataclasses import dataclass

import cv2
import numpy as np

from slam.data import DataFolder, GroundTruthSample
from slam.feature_detection import FeatureDetectionResult
from slam.stereo_matching import StereoMatchingResult
from slam.util import quaternion_to_rotation_matrix


def _get_closest_ground_truth_pose(
    ground_truth_samples: list[GroundTruthSample],
    gt_timestamps: np.ndarray,
    timestamp_ns: int,
) -> np.ndarray:
    idx = int(np.argmin(np.abs(gt_timestamps - timestamp_ns)))
    sample = ground_truth_samples[idx]
    pose = np.eye(4)
    pose[:3, :3] = quaternion_to_rotation_matrix(sample.quaternion)
    pose[:3, 3] = np.array(sample.position)
    return pose


@dataclass
class CoordinateMappingCheckFrame:
    timestamp_ns: int
    projection_errors: list[float]
    matches: list  # list[cv2.DMatch]
    projected_points: list[tuple[float, float]]  # projected positions in frame k+1 image space


@dataclass
class CoordinateMappingCheckResult:
    frames: list[CoordinateMappingCheckFrame]
    times: np.ndarray  # seconds since first camera frame


class CoordinateMappingChecker:
    def __init__(
        self,
        data: DataFolder,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
    ) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self._stereo_matching_result = stereo_matching_result
        self.progress: float = 0.0

    def run(self) -> CoordinateMappingCheckResult:
        data = self._data
        fd_result = self._feature_detection_result
        sm_result = self._stereo_matching_result

        K0 = data.cam0_intrinsics.to_matrix()
        dist_coeffs = np.array([
            data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
            data.cam0_intrinsics.p1, data.cam0_intrinsics.p2,
        ])

        gt_timestamps = np.array([s.timestamp_ns for s in data.ground_truth_samples])

        def gt_world_T_cam0(timestamp_ns: int) -> np.ndarray:
            pose = _get_closest_ground_truth_pose(data.ground_truth_samples, gt_timestamps, timestamp_ns)
            return pose

        n = len(sm_result.frames)
        first_ts = data.cam_timestamps_ns[0]
        frames: list[CoordinateMappingCheckFrame] = []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        for k in range(n - 1):
            self.progress = k / (n - 1)

            sm_k = sm_result.frames[k]
            sm_k1 = sm_result.frames[k + 1]
            fd_k = fd_result.frames[k]
            fd_k1 = fd_result.frames[k + 1]

            if len(sm_k.matches) < 2 or len(fd_k1.cam0_descriptors) < 2:
                frames.append(CoordinateMappingCheckFrame(
                    timestamp_ns=sm_k.timestamp_ns,
                    projection_errors=[],
                    matches=[],
                    projected_points=[],
                ))
                continue

            world_T_cam0_k = gt_world_T_cam0(sm_k.timestamp_ns)
            world_T_cam0_k1 = gt_world_T_cam0(sm_k1.timestamp_ns)
            cam0_T_world_k1 = np.linalg.inv(world_T_cam0_k1)

            stereo_desc_k = fd_k.cam0_descriptors[[m.queryIdx for m in sm_k.matches]]
            raw_matches = bf.knnMatch(stereo_desc_k, fd_k1.cam0_descriptors, k=2)
            good = [ms[0] for ms in raw_matches if len(ms) == 2 and ms[0].distance < 0.75 * ms[1].distance]

            if not good:
                frames.append(CoordinateMappingCheckFrame(
                    timestamp_ns=sm_k.timestamp_ns,
                    projection_errors=[],
                    matches=[],
                    projected_points=[],
                ))
                continue

            errors = []
            valid_matches = []
            projected_points = []
            for m in good:
                p_cam0_k = sm_k.points_3d[:, m.queryIdx]
                p_world = world_T_cam0_k[:3, :3] @ p_cam0_k + world_T_cam0_k[:3, 3]
                p_cam0_k1 = cam0_T_world_k1[:3, :3] @ p_world + cam0_T_world_k1[:3, 3]

                if p_cam0_k1[2] <= 0:
                    continue

                projected, _ = cv2.projectPoints(
                    np.array([p_cam0_k1], dtype=np.float64),
                    np.zeros(3), np.zeros(3), K0, dist_coeffs,
                )
                projected_pt = projected.reshape(2)
                actual_pt = np.array(fd_k1.cam0_keypoints[m.trainIdx].pt)
                errors.append(float(np.linalg.norm(projected_pt - actual_pt)))
                valid_matches.append(m)
                projected_points.append((float(projected_pt[0]), float(projected_pt[1])))

            frames.append(CoordinateMappingCheckFrame(
                timestamp_ns=sm_k.timestamp_ns,
                projection_errors=errors,
                matches=valid_matches,
                projected_points=projected_points,
            ))

        self.progress = 1.0
        times = np.array([(f.timestamp_ns - first_ts) / 1e9 for f in frames])
        return CoordinateMappingCheckResult(frames=frames, times=times)
