import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult


# Reject a stereo match if its (undistorted) points sit more than this far, in pixels, from the
# epipolar geometry implied by the *known* stereo calibration (Sampson distance). Tight because
# F is exact here -- it is derived from the fixed extrinsics, not estimated from noisy matches.
EPIPOLAR_SAMPSON_PX = 1.5


def _skew(t: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ])


@dataclass
class StereoMatchingFrame:
    timestamp_ns: int
    matches: list  # list[cv2.DMatch]
    points_3d: np.ndarray  # shape (3, N)


@dataclass
class StereoMatchingResult:
    frames: list[StereoMatchingFrame]
    elapsed_s: float


class StereoMatchingSolver:
    def __init__(self, data: EuRoCMAVData, feature_detection_result: FeatureDetectionResult) -> None:
        self._data = data
        self._feature_detection_result = feature_detection_result
        self.progress: float = 0.0

    def _process_frame(self, i: int) -> StereoMatchingFrame:
        fd = self._feature_detection_result.frames[i]
        data = self._data

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = bf.knnMatch(fd.cam0_descriptors, fd.cam1_descriptors, k=2)
        good_matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

        if not good_matches:
            return StereoMatchingFrame(
                timestamp_ns=fd.timestamp_ns,
                matches=[],
                points_3d=np.zeros((3, 0)),
            )

        K0 = data.cam0_intrinsics.to_matrix()
        K1 = data.cam1_intrinsics.to_matrix()
        dist_coeffs0 = np.array([
            data.cam0_intrinsics.k1,
            data.cam0_intrinsics.k2,
            data.cam0_intrinsics.p1,
            data.cam0_intrinsics.p2,
        ])
        dist_coeffs1 = np.array([
            data.cam1_intrinsics.k1,
            data.cam1_intrinsics.k2,
            data.cam1_intrinsics.p1,
            data.cam1_intrinsics.p2,
        ])

        points0 = np.array([fd.cam0_keypoints[m.queryIdx].pt for m in good_matches])
        points1 = np.array([fd.cam1_keypoints[m.trainIdx].pt for m in good_matches])

        points0 = cv2.undistortPoints(points0, K0, dist_coeffs0, P=K0).reshape(-1, 2)
        points1 = cv2.undistortPoints(points1, K1, dist_coeffs1, P=K1).reshape(-1, 2)

        T_cam0_to_cam1 = np.linalg.inv(data.cam1_extrinsics) @ data.cam0_extrinsics
        R = T_cam0_to_cam1[:3, :3]
        t = T_cam0_to_cam1[:3, 3]

        # Epipolar gate. A correct stereo match must satisfy x1^T F x0 = 0. F is built exactly
        # from the known extrinsics (F = K1^-T [t]x R K0^-1), so instead of estimating it we
        # score each match by its Sampson distance -- an approximate pixel distance to the
        # epipolar geometry -- and drop matches beyond EPIPOLAR_SAMPSON_PX. The vast majority of
        # nonsense pairs (repeated texture that fools the descriptor ratio test) land far off
        # their epipolar line and are removed before they can triangulate into garbage 3D points.
        F = np.linalg.inv(K1).T @ _skew(t) @ R @ np.linalg.inv(K0)
        p0h = np.hstack([points0, np.ones((len(points0), 1))])
        p1h = np.hstack([points1, np.ones((len(points1), 1))])
        Fp0 = p0h @ F.T      # epipolar line in cam1 for each cam0 point
        Ftp1 = p1h @ F       # epipolar line in cam0 for each cam1 point
        num = np.sum(p1h * Fp0, axis=1)  # x1^T F x0
        denom = Fp0[:, 0] ** 2 + Fp0[:, 1] ** 2 + Ftp1[:, 0] ** 2 + Ftp1[:, 1] ** 2
        sampson_sq = np.where(denom > 1e-12, num ** 2 / np.maximum(denom, 1e-12), np.inf)
        keep = sampson_sq < EPIPOLAR_SAMPSON_PX ** 2

        good_matches = [m for m, k in zip(good_matches, keep) if k]
        points0 = points0[keep]
        points1 = points1[keep]

        if not good_matches:
            return StereoMatchingFrame(
                timestamp_ns=fd.timestamp_ns,
                matches=[],
                points_3d=np.zeros((3, 0)),
            )

        P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K1 @ T_cam0_to_cam1[:3, :]

        points_4d = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]

        return StereoMatchingFrame(
            timestamp_ns=fd.timestamp_ns,
            matches=good_matches,
            points_3d=points_3d,
        )

    def run(self) -> StereoMatchingResult:
        n = len(self._feature_detection_result.frames)
        frames: list[StereoMatchingFrame | None] = [None] * n
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_index = {
                executor.submit(self._process_frame, i): i
                for i in range(n)
            }
            completed = 0
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                frames[i] = future.result()
                completed += 1
                self.progress = completed / n
        elapsed_s = time.monotonic() - t0
        return StereoMatchingResult(frames=frames, elapsed_s=elapsed_s)
