import threading
from dataclasses import dataclass
from typing import Optional

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


def detect_features_for_frame(
    data: EuRoCMAVData, ts: int,
    cam0_img: Optional[np.ndarray] = None, cam1_img: Optional[np.ndarray] = None,
) -> FeatureDetectionFrame:
    """ORB detect+compute on both cameras for a single frame: a pure function of that frame's
    images, no cross-frame state. Callers may pass pre-loaded images, since the same ones are needed
    for stereo matching and optical flow.
    """
    # FAST_SCORE (reuses the FAST corner strength) instead of the default HARRIS_SCORE, which
    # needs a separate Harris pass per keypoint, as in ORB-SLAM.
    #
    # Separate cam0/cam1 ORB instances so the two detectAndCompute calls can run concurrently
    # (Feature2D isn't documented as thread-safe).
    orb0 = cv2.ORB.create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
    orb1 = cv2.ORB.create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
    if cam0_img is None:
        cam0_img = cv2.imread(str(data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
    if cam1_img is None:
        cam1_img = cv2.imread(str(data.get_cam1_image_path(ts)), cv2.IMREAD_GRAYSCALE)
    # cv2's stub types `mask` as non-Optional, but passing None is the standard idiom.
    #
    # Run cam1's detectAndCompute on a second thread while this one runs cam0's; cv2's native
    # calls release the GIL, so this overlaps real work.
    cam1_out: list = [None, None]

    def _detect_cam1() -> None:
        cam1_out[0], cam1_out[1] = orb1.detectAndCompute(cam1_img, None)  # type: ignore[call-overload]

    cam1_thread = threading.Thread(target=_detect_cam1)
    cam1_thread.start()
    cam0_keypoints, cam0_descriptors = orb0.detectAndCompute(cam0_img, None)  # type: ignore[call-overload]
    cam1_thread.join()
    cam1_keypoints, cam1_descriptors = cam1_out
    return FeatureDetectionFrame(
        timestamp_ns=ts,
        cam0_keypoints=list(cam0_keypoints),
        cam0_descriptors=cam0_descriptors,
        cam1_keypoints=list(cam1_keypoints),
        cam1_descriptors=cam1_descriptors,
    )
