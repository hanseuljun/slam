from dataclasses import dataclass
from typing import Optional

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
    icp_transform: Optional[np.ndarray]  # 4x4 rigid transform (world space) minimizing error between matched points
    icp_projection_errors: list[float]  # per-match projection errors when reprojecting via icp_transform


@dataclass
class CoordinateMappingCheckResult:
    frames: list[CoordinateMappingCheckFrame]
    times: np.ndarray  # seconds since first camera frame


def _solve_icp_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Return 4x4 rigid transform T minimizing sum ||T @ src - dst||^2 (Kabsch/SVD method)."""
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)
    H = (src_pts - mu_src).T @ (dst_pts - mu_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_dst - R @ mu_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


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
            return np.linalg.inv(data.cam0_extrinsics) @ pose

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
                    icp_transform=None,
                    icp_projection_errors=[],
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
                    icp_transform=None,
                    icp_projection_errors=[],
                ))
                continue

            # Build lookup: cam0 keypoint index in frame k+1 → 3D point in cam0_k1 space
            k1_kp_to_3d = {m.queryIdx: sm_k1.points_3d[:, i] for i, m in enumerate(sm_k1.matches)}

            errors = []
            valid_matches = []
            projected_points = []
            icp_src: list[np.ndarray] = []
            icp_dst: list[np.ndarray] = []
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

                if m.trainIdx in k1_kp_to_3d:
                    _MIN_DEPTH, _MAX_DEPTH, _MAX_WORLD_DIST = 0.1, 50.0, 0.5
                    p_stereo_k1 = k1_kp_to_3d[m.trainIdx]
                    if (_MIN_DEPTH < p_cam0_k[2] < _MAX_DEPTH
                            and _MIN_DEPTH < p_stereo_k1[2] < _MAX_DEPTH):
                        p_world_k = world_T_cam0_k[:3, :3] @ p_cam0_k + world_T_cam0_k[:3, 3]
                        p_world_k1 = world_T_cam0_k1[:3, :3] @ p_stereo_k1 + world_T_cam0_k1[:3, 3]
                        if np.linalg.norm(p_world_k - p_world_k1) < _MAX_WORLD_DIST:
                            icp_src.append(p_world_k)
                            icp_dst.append(p_world_k1)

            icp_transform = None
            if len(icp_src) >= 3:
                icp_transform = _solve_icp_transform(np.array(icp_src), np.array(icp_dst))

            icp_projection_errors: list[float] = []
            if icp_transform is not None:
                for m in valid_matches:
                    p_cam0_k = sm_k.points_3d[:, m.queryIdx]
                    p_world_k = world_T_cam0_k[:3, :3] @ p_cam0_k + world_T_cam0_k[:3, 3]
                    p_world_k1_icp = icp_transform[:3, :3] @ p_world_k + icp_transform[:3, 3]
                    p_cam0_k1_icp = cam0_T_world_k1[:3, :3] @ p_world_k1_icp + cam0_T_world_k1[:3, 3]
                    if p_cam0_k1_icp[2] <= 0:
                        continue
                    projected_icp, _ = cv2.projectPoints(
                        np.array([p_cam0_k1_icp], dtype=np.float64),
                        np.zeros(3), np.zeros(3), K0, dist_coeffs,
                    )
                    projected_icp_pt = projected_icp.reshape(2)
                    actual_pt = np.array(fd_k1.cam0_keypoints[m.trainIdx].pt)
                    icp_projection_errors.append(float(np.linalg.norm(projected_icp_pt - actual_pt)))

            frames.append(CoordinateMappingCheckFrame(
                timestamp_ns=sm_k.timestamp_ns,
                projection_errors=errors,
                matches=valid_matches,
                projected_points=projected_points,
                icp_transform=icp_transform,
                icp_projection_errors=icp_projection_errors,
            ))

        self.progress = 1.0
        times = np.array([(f.timestamp_ns - first_ts) / 1e9 for f in frames])
        return CoordinateMappingCheckResult(frames=frames, times=times)
