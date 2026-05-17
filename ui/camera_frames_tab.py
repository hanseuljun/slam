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


class CameraFramesTabState:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.frame_index: int = 0
        self._cached_index: int = -1
        self._cached_image: Optional[np.ndarray] = None
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def current_image(self) -> Optional[np.ndarray]:
        if self._cached_index != self.frame_index:
            self._texture = None
            ts = self.data.cam_timestamps_ns[self.frame_index]
            self._cached_image = cv2.imread(
                str(self.data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE
            )
            self._cached_index = self.frame_index
        return self._cached_image

    def current_texture(self) -> Optional[hello_imgui.TextureGpu]:
        image = self.current_image()
        if image is None:
            return None
        if self._texture is None:
            self._texture = _to_texture(image)
        return self._texture


def camera_frames_tab(state: CameraFramesTabState) -> None:
    n = len(state.data.cam_timestamps_ns)
    _, state.frame_index = imgui.slider_int("Frame", state.frame_index, 0, n - 1)
    tex = state.current_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
