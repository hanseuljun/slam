from typing import Optional

import cv2
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam import DataFolder


def _to_texture(image: np.ndarray) -> hello_imgui.TextureGpu:
    if image.ndim == 2:
        rgba = np.stack([image, image, image, np.full_like(image, 255)], axis=-1)
    else:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return hello_imgui.create_texture_gpu_from_rgba_data(rgba)


class DataViewState:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.frame_index: int = 0
        self._cached_index: int = -1
        self._cached_image: Optional[np.ndarray] = None
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def current_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._cached_index != self.frame_index:
            self._texture = None
            ts = self.data.cam_timestamps_ns[self.frame_index]
            self._cached_image = cv2.imread(
                str(self.data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE
            )
            self._cached_index = self.frame_index
        if self._cached_image is not None and self._texture is None:
            self._texture = _to_texture(self._cached_image)
        return self._texture


def data_view(state: DataViewState) -> None:
    data = state.data
    n = len(data.cam_timestamps_ns)
    first_ts_ns = data.cam_timestamps_ns[0]

    tex = state.current_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##frame_slider", state.frame_index, 0, n - 1)
        if changed:
            state.frame_index = new_index

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##frame_input", state.frame_index, step=1)
    if changed:
        state.frame_index = max(0, min(n - 1, new_index))

    ts_ns = data.cam_timestamps_ns[state.frame_index]
    imgui.text(f"Timestamp: {ts_ns} ns")
    imgui.text(f"Time since first frame: {(ts_ns - first_ts_ns) / 1e9:.3f} s")
