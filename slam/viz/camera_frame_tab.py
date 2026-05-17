from typing import Optional

import cv2
import numpy as np
from imgui_bundle import imgui, immvision

from slam import DataFolder


class CameraFramesState:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.frame_index: int = 0
        self._cached_index: int = -1
        self._cached_image: Optional[np.ndarray] = None

    def current_image(self) -> Optional[np.ndarray]:
        if self._cached_index != self.frame_index:
            ts = self.data.cam_timestamps_ns[self.frame_index]
            self._cached_image = cv2.imread(
                str(self.data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE
            )
            self._cached_index = self.frame_index
        return self._cached_image


def camera_frames_tab(state: CameraFramesState) -> None:
    n = len(state.data.cam_timestamps_ns)
    _, state.frame_index = imgui.slider_int("Frame", state.frame_index, 0, n - 1)
    image = state.current_image()
    if image is not None:
        immvision.image("##cam0", image, immvision.ImageParams())
