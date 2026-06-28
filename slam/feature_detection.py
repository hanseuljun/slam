import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from slam.data import EuRoCMAVData


@dataclass
class FeatureDetectionFrame:
    timestamp_ns: int
    cam0_keypoints: list  # list[cv2.KeyPoint]
    cam0_descriptors: np.ndarray  # shape (N, 32), ORB binary descriptors
    cam1_keypoints: list  # list[cv2.KeyPoint]
    cam1_descriptors: np.ndarray  # shape (N, 32), ORB binary descriptors


@dataclass
class FeatureDetectionResult:
    frames: list[FeatureDetectionFrame]
    elapsed_s: float


class FeatureDetectionSolver:
    def __init__(self, data: EuRoCMAVData, start_s: float = 0.0, duration_s: float = 5.0) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self.progress: float = 0.0

    def _process_frame(self, ts: int) -> FeatureDetectionFrame:
        orb = cv2.ORB_create(nfeatures=2000)
        cam0_img = cv2.imread(str(self._data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        cam1_img = cv2.imread(str(self._data.get_cam1_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        cam0_keypoints, cam0_descriptors = orb.detectAndCompute(cam0_img, None)
        cam1_keypoints, cam1_descriptors = orb.detectAndCompute(cam1_img, None)
        return FeatureDetectionFrame(
            timestamp_ns=ts,
            cam0_keypoints=list(cam0_keypoints),
            cam0_descriptors=cam0_descriptors,
            cam1_keypoints=list(cam1_keypoints),
            cam1_descriptors=cam1_descriptors,
        )

    def run(self) -> FeatureDetectionResult:
        first_ts = self._data.cam_timestamps_ns[0]
        min_ts = first_ts + int(self._start_s * 1e9)
        max_ts = min_ts + int(self._duration_s * 1e9)
        timestamps = [t for t in self._data.cam_timestamps_ns if min_ts <= t <= max_ts]
        n = len(timestamps)
        frames: list[FeatureDetectionFrame | None] = [None] * n
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_index = {
                executor.submit(self._process_frame, ts): i
                for i, ts in enumerate(timestamps)
            }
            completed = 0
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                frames[i] = future.result()
                completed += 1
                self.progress = completed / n
        elapsed_s = time.monotonic() - t0
        return FeatureDetectionResult(frames=frames, elapsed_s=elapsed_s)
