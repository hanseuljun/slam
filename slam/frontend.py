from dataclasses import dataclass

import cv2

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionFrame, FeatureDetectionResult, detect_features_for_frame
from slam.optical_flow import OpticalFlowFrame, OpticalFlowResult, OpticalFlowTracker
from slam.stereo_matching import StereoMatchingFrame, StereoMatchingResult, match_and_triangulate_stereo


@dataclass
class FrontendResult:
    feature_detection: FeatureDetectionResult
    stereo_matching: StereoMatchingResult
    optical_flow: OpticalFlowResult
    elapsed_s: float


class FrontendFrameComputer:
    """Feature detection + stereo matching + optical flow for one frame at a time, driven by
    slam.py's own per-frame loop (which also drives _KeyframeScanner/_GtsamBuilder). There is no
    separate batch frontend pass: the Frontend and Coordinate Mapping tabs read the containers
    SlamSolver fills (SlamSolver.frontend_result).

    (Feature detection, stereo matching and optical flow used to be three separate solvers, then
    one FrontendSolver, each only usable after the one before it finished. None is independently
    useful -- optical flow needs stereo matching's per-match depth to seed a track, and stereo
    matching needs feature detection's keypoints -- and the tracker carries state frame to frame,
    so this can't be parallelized across frames the way feature detection alone could.)

    Each camera image is loaded once and reused for both feature detection and optical flow
    (detect_features_for_frame accepts pre-loaded images for exactly this reason). A single
    OpticalFlowTracker is owned across the whole lifetime of this object, since track identity has
    to stay continuous frame to frame -- see OpticalFlowTracker's docstring.
    """

    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data
        self._tracker = OpticalFlowTracker()

    def add_frame(self, ts: int) -> tuple[FeatureDetectionFrame, StereoMatchingFrame, OpticalFlowFrame]:
        cam0_img = cv2.imread(str(self._data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        cam1_img = cv2.imread(str(self._data.get_cam1_image_path(ts)), cv2.IMREAD_GRAYSCALE)
        fd_frame = detect_features_for_frame(self._data, ts, cam0_img, cam1_img)

        matches, points_3d = match_and_triangulate_stereo(
            self._data, fd_frame.cam0_keypoints, fd_frame.cam0_descriptors,
            fd_frame.cam1_keypoints, fd_frame.cam1_descriptors)
        sm_frame = StereoMatchingFrame(timestamp_ns=ts, matches=matches, points_3d=points_3d)

        of_frame = self._tracker.add_frame(fd_frame, sm_frame, cam0_img)
        return fd_frame, sm_frame, of_frame
