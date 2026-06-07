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


def detect_features(data: DataFolder) -> FeatureDetectionResult:
    orb = cv2.ORB_create(nfeatures=2000)
    frames = []
    for ts in data.cam_timestamps_ns:
        img = cv2.imread(str(data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(img, None)
        frames.append(FeatureDetectionFrame(
            timestamp_ns=ts,
            keypoints=list(keypoints),
            descriptors=descriptors,
        ))
    return FeatureDetectionResult(frames=frames)
