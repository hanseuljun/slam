import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """ORB detect+compute on both cameras for a single frame -- a pure function of that frame's
    own images, no cross-frame state. Shared by FeatureDetectionSolver (batch, thread-pooled
    across every frame in a window, always loads its own images) and slam.py's own incremental
    per-frame loop (which loads cam0 once and passes it in here, since it also needs the same
    image for optical flow -- passing cam0_img/cam1_img in avoids loading either a second time).
    """
    # FAST_SCORE (reuses the FAST corner strength) instead of the default HARRIS_SCORE (a separate
    # Harris-corner response pass per keypoint) -- the same tradeoff real-time VO/VIO pipelines like
    # ORB-SLAM make; see tmp/orb_speedup_options.html.
    #
    # Separate cam0/cam1 ORB instances (rather than sharing one) so the two detectAndCompute calls
    # below can safely run concurrently -- cv2 Feature2D objects aren't documented as safe for
    # concurrent use from multiple threads.
    orb0 = cv2.ORB.create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
    orb1 = cv2.ORB.create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)
    if cam0_img is None:
        cam0_img = cv2.imread(str(data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE)
    if cam1_img is None:
        cam1_img = cv2.imread(str(data.get_cam1_image_path(ts)), cv2.IMREAD_GRAYSCALE)
    # cv2's stub types `mask` as non-Optional MatLike, but passing None (no mask) is the
    # standard, correct OpenCV idiom -- the stub is just missing the `| None`.
    #
    # cam0/cam1 are independent images -- run cam1's detectAndCompute on a second thread while this
    # one runs cam0's, instead of back-to-back. cv2's native calls release the GIL, so this overlaps
    # real work rather than just interleaving Python bytecode; see tmp/orb_speedup_options.html.
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


class FeatureDetectionSolver:
    def __init__(
        self, data: EuRoCMAVData, start_s: float = 0.0, duration_s: float = 5.0,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self._cancel_event = cancel_event if cancel_event is not None else threading.Event()
        self.progress: float = 0.0

    def _process_frame(self, ts: int) -> FeatureDetectionFrame:
        return detect_features_for_frame(self._data, ts)

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
                if self._cancel_event.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                i = future_to_index[future]
                frames[i] = future.result()
                completed += 1
                self.progress = completed / n
        elapsed_s = time.monotonic() - t0
        # Cancellation can break out of the loop above before every slot is filled; the caller
        # always discards a result once it sees its own cancel_event set, but drop the unfilled
        # slots here too so the return type stays honestly non-Optional either way.
        completed_frames = [f for f in frames if f is not None]
        return FeatureDetectionResult(frames=completed_frames, elapsed_s=elapsed_s)
