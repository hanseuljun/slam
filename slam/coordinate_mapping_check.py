import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from slam.data import EuRoCMAVData
from slam.optical_flow import OpticalFlowResult
from slam.util import quaternion_to_rotation_matrix



@dataclass
class CoordinateMappingCheckFrame:
    timestamp_ns: int
    gt_sample_index: int
    gt_transform: np.ndarray  # 4x4 cam0_k1_T_cam0_k from ground truth (relative pose, k → k+1)
    projection_errors: list[float]
    track_ids: list[int]  # optical-flow tracks checked (alive in both frames, with a 3D point at frame k)
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
        data: EuRoCMAVData,
        optical_flow_result: OpticalFlowResult,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._data = data
        self._optical_flow_result = optical_flow_result
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress: float = 0.0

    def run(self) -> CoordinateMappingCheckResult:
        data = self._data
        of_result = self._optical_flow_result

        K0 = data.cam0_intrinsics.to_matrix()
        dist_coeffs = np.array([
            data.cam0_intrinsics.k1, data.cam0_intrinsics.k2,
            data.cam0_intrinsics.p1, data.cam0_intrinsics.p2,
        ])

        gt_timestamps = np.array([s.timestamp_ns for s in data.ground_truth_samples])

        def gt_sample_index(timestamp_ns: int) -> int:
            return int(np.argmin(np.abs(gt_timestamps - timestamp_ns)))


        def gt_world_T_cam0(idx: int) -> np.ndarray:
            sample = data.ground_truth_samples[idx]
            world_T_body = np.eye(4)
            world_T_body[:3, :3] = quaternion_to_rotation_matrix(sample.quaternion)
            world_T_body[:3, 3] = np.array(sample.position)
            body_T_cam0 = data.cam0_extrinsics
            return world_T_body @ body_T_cam0

        n = len(of_result.frames)
        first_ts = data.cam_timestamps_ns[0]
        frames: list[CoordinateMappingCheckFrame] = []

        for k in range(n - 1):
            if self._cancel_event.is_set():
                break
            self.progress = k / (n - 1)

            of_k = of_result.frames[k]
            of_k1 = of_result.frames[k + 1]

            gt_idx_k = gt_sample_index(of_k.timestamp_ns)
            gt_idx_k1 = gt_sample_index(of_k1.timestamp_ns)
            world_T_cam0_k = gt_world_T_cam0(gt_idx_k)
            world_T_cam0_k1 = gt_world_T_cam0(gt_idx_k1)
            cam0_T_world_k1 = np.linalg.inv(world_T_cam0_k1)
            gt_transform = cam0_T_world_k1 @ world_T_cam0_k

            tids = [tid for tid in of_k.stereo_point_cam0 if tid in of_k1.track_uv]
            if not tids:
                frames.append(CoordinateMappingCheckFrame(
                    timestamp_ns=of_k.timestamp_ns,
                    gt_sample_index=gt_idx_k,
                    gt_transform=gt_transform,
                    projection_errors=[],
                    track_ids=[],
                    projected_points=[],
                    icp_transform=None,
                    icp_projection_errors=[],
                ))
                continue

            errors = []
            valid_tids: list[int] = []
            projected_points = []
            icp_src: list[np.ndarray] = []
            icp_dst: list[np.ndarray] = []
            for tid in tids:
                p_cam0_k = of_k.stereo_point_cam0[tid]
                p_world = world_T_cam0_k[:3, :3] @ p_cam0_k + world_T_cam0_k[:3, 3]
                p_cam0_k1 = cam0_T_world_k1[:3, :3] @ p_world + cam0_T_world_k1[:3, 3]

                _MIN_DEPTH, _MAX_DEPTH = 0.1, 50.0
                if not (_MIN_DEPTH < p_cam0_k[2] < _MAX_DEPTH) or p_cam0_k1[2] <= 0:
                    continue

                projected, _ = cv2.projectPoints(
                    np.array([p_cam0_k1], dtype=np.float64),
                    np.zeros(3), np.zeros(3), K0, dist_coeffs,
                )
                projected_pt = projected.reshape(2)
                actual_pt = np.array(of_k1.track_uv[tid])
                errors.append(float(np.linalg.norm(projected_pt - actual_pt)))
                valid_tids.append(tid)
                projected_points.append((float(projected_pt[0]), float(projected_pt[1])))

                if tid in of_k1.stereo_point_cam0:
                    _MIN_DEPTH, _MAX_DEPTH, _MAX_WORLD_DIST = 0.1, 50.0, 0.5
                    p_stereo_k1 = of_k1.stereo_point_cam0[tid]
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
                for tid in valid_tids:
                    p_cam0_k = of_k.stereo_point_cam0[tid]
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
                    actual_pt = np.array(of_k1.track_uv[tid])
                    icp_projection_errors.append(float(np.linalg.norm(projected_icp_pt - actual_pt)))

            frames.append(CoordinateMappingCheckFrame(
                timestamp_ns=of_k.timestamp_ns,
                gt_sample_index=gt_idx_k,
                gt_transform=gt_transform,
                projection_errors=errors,
                track_ids=valid_tids,
                projected_points=projected_points,
                icp_transform=icp_transform,
                icp_projection_errors=icp_projection_errors,
            ))

        self.progress = 1.0
        times = np.array([(f.timestamp_ns - first_ts) / 1e9 for f in frames])
        return CoordinateMappingCheckResult(frames=frames, times=times)
