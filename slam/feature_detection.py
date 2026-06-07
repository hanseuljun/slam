import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from slam.data import DataFolder


@dataclass
class FeatureDetectionFrame:
    timestamp_ns: int
    keypoints: list  # list[cv2.KeyPoint]
    descriptors: np.ndarray  # shape (N, 32), ORB binary descriptors


@dataclass
class FeatureDetectionResult:
    frames: list[FeatureDetectionFrame]


class FeatureDetectionSolver:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self.progress: float = 0.0

    def _process_frame(self, ts: int) -> FeatureDetectionFrame:
        orb = cv2.ORB_create(nfeatures=2000)
        img = cv2.imread(str(self._data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        return FeatureDetectionFrame(
            timestamp_ns=ts,
            keypoints=list(keypoints),
            descriptors=descriptors,
        )

    def run(self) -> FeatureDetectionResult:
        timestamps = self._data.cam_timestamps_ns
        n = len(timestamps)
        frames: list[FeatureDetectionFrame | None] = [None] * n
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
        return FeatureDetectionResult(frames=frames)
