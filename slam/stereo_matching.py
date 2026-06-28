import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult


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

        P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        T_cam0_to_cam1 = np.linalg.inv(data.cam1_extrinsics) @ data.cam0_extrinsics
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
