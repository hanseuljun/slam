import threading
import time
from dataclasses import dataclass
from typing import Optional

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
    """Feature detection + stereo matching + optical flow for one frame at a time -- the per-frame
    body FrontendSolver.run() (below) loops over, pulled out so slam.py's own per-frame loop (which
    also drives _KeyframeScanner/_GtsamBuilder) can call it directly instead of needing a separate
    complete pass before keyframe selection can even start.

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


class FrontendSolver:
    """Batch driver over FrontendFrameComputer for a [start_s, start_s + duration_s] window --
    feature detection, stereo matching, and optical flow used to be three separate solvers
    (FeatureDetectionSolver -> StereoMatchingSolver -> OpticalFlowSolver, each independently
    loading images and only usable strictly after the one before it finished) with their own tabs.
    Merged into one here since none of the three is independently useful -- optical flow always
    needs stereo matching's per-frame match to seed a track's depth, and stereo matching always
    needs feature detection's keypoints -- so a caller only ever wants all three together, over the
    same per-frame image load, one frame at a time in strictly increasing order (optical flow's
    tracker carries state frame-to-frame, so this can't be thread-pool-parallelized across frames
    the way feature detection alone could be).
    """

    def __init__(
        self, data: EuRoCMAVData, start_s: float = 0.0, duration_s: float = 5.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress: float = 0.0

    def run(self) -> FrontendResult:
        first_ts = self._data.cam_timestamps_ns[0]
        min_ts = first_ts + int(self._start_s * 1e9)
        max_ts = min_ts + int(self._duration_s * 1e9)
        timestamps = [t for t in self._data.cam_timestamps_ns if min_ts <= t <= max_ts]
        n = len(timestamps)

        computer = FrontendFrameComputer(self._data)
        fd_frames: list[FeatureDetectionFrame] = []
        sm_frames: list[StereoMatchingFrame] = []
        of_frames: list[OpticalFlowFrame] = []

        t0 = time.monotonic()
        for i, ts in enumerate(timestamps):
            if self._cancel_event.is_set():
                break
            fd_frame, sm_frame, of_frame = computer.add_frame(ts)
            fd_frames.append(fd_frame)
            sm_frames.append(sm_frame)
            of_frames.append(of_frame)
            self.progress = (i + 1) / n

        elapsed_s = time.monotonic() - t0
        return FrontendResult(
            feature_detection=FeatureDetectionResult(frames=fd_frames, elapsed_s=elapsed_s),
            stereo_matching=StereoMatchingResult(frames=sm_frames, elapsed_s=elapsed_s),
            optical_flow=OpticalFlowResult(frames=of_frames, elapsed_s=elapsed_s),
            elapsed_s=elapsed_s,
        )
