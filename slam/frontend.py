from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionFrame, detect_features_for_frame
from slam.optical_flow import OpticalFlowFrame, OpticalFlowResult, OpticalFlowTracker
from slam.stereo_matching import StereoGeometry, StereoMatchingFrame, match_and_triangulate_stereo


@dataclass
class KeyframeFeatures:
    """ORB features and the ORB stereo match of one keyframe, for loop closure."""
    fd: FeatureDetectionFrame
    sm: StereoMatchingFrame


@dataclass
class FrontendResult:
    optical_flow: OpticalFlowResult  # every frame
    keyframe_features: dict[int, KeyframeFeatures] = field(default_factory=dict)  # frame index -> ORB
    elapsed_s: float = 0.0


class FrontendFrameComputer:
    """The per-frame frontend, driven by slam.py's frame loop: KLT tracking with stereo depth on
    every frame (OpticalFlowTracker). ORB is much costlier and is only needed for loop closure, so
    it is computed separately, for keyframes only (compute_keyframe_features).
    """

    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data
        self._tracker = OpticalFlowTracker(data)
        self._geom = StereoGeometry(data)
        self._pool = ThreadPoolExecutor(max_workers=1)

    def add_frame(self, ts: int) -> OpticalFlowFrame:
        # cv2.imread releases the GIL, so cam1 loads while this thread loads cam0.
        cam1_future = self._pool.submit(cv2.imread, str(self._data.get_cam1_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        cam0_img = cv2.imread(str(self._data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        return self._tracker.add_frame(ts, cam0_img, cam1_future.result())

    def compute_keyframe_features(self, ts: int) -> KeyframeFeatures:
        fd = detect_features_for_frame(self._data, ts)
        matches, points_3d = match_and_triangulate_stereo(
            self._geom, fd.cam0_keypoints, fd.cam0_descriptors, fd.cam1_keypoints, fd.cam1_descriptors)
        return KeyframeFeatures(fd, StereoMatchingFrame(timestamp_ns=ts, matches=matches, points_3d=points_3d))
