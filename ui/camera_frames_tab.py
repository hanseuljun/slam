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
    first_ts_ns = state.data.cam_timestamps_ns[0]
    img = ui.image(source='').classes('w-full')

    def update(index: int) -> None:
        state.frame_index = max(0, min(n - 1, index))
        ts_ns = state.data.cam_timestamps_ns[state.frame_index]
        slider.value = state.frame_index
        label_index.text = f'Frame {state.frame_index}'
        label_ts.text = f'{ts_ns} ns'
        label_rel.text = f'{(ts_ns - first_ts_ns) / 1e9:.3f} s'
        image = state.current_image()
        if image is not None:
            img.source = array_to_data_uri(image)

    with ui.row().classes('items-center'):
        ui.button(icon='keyboard_arrow_up', on_click=lambda: update(state.frame_index + 1))
        ui.button(icon='keyboard_arrow_down', on_click=lambda: update(state.frame_index - 1))
        label_index = ui.label()

    label_ts = ui.label()
    label_rel = ui.label()

    slider = ui.slider(min=0, max=n - 1, step=1, value=0,
                       on_change=lambda e: update(int(e.value))).classes('w-full')

    update(0)
