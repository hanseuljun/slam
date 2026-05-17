from typing import Optional

import cv2
import numpy as np
from nicegui import ui

from slam import DataFolder
from ui._utils import array_to_data_uri


class CameraFramesTabState:
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


def camera_frames_tab(state: CameraFramesTabState) -> None:
    n = len(state.data.cam_timestamps_ns)
    img = ui.image(source='').classes('w-full')

    def on_slide(e) -> None:
        state.frame_index = int(e.value)
        image = state.current_image()
        if image is not None:
            img.source = array_to_data_uri(image)

    ui.slider(min=0, max=n - 1, step=1, value=0, on_change=on_slide)

    image = state.current_image()
    if image is not None:
        img.source = array_to_data_uri(image)
