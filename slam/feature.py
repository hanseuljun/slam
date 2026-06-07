from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureDetectionResult:
    keypoints: list  # list[cv2.KeyPoint]
    descriptors: np.ndarray  # shape (N, 32), ORB binary descriptors


def detect_features(image: np.ndarray) -> FeatureDetectionResult:
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return FeatureDetectionResult(keypoints=list(keypoints), descriptors=descriptors)
