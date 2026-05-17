import cv2
from imgui_bundle import imgui, immvision

from slam import DataFolder


class CameraFramesState:
    def __init__(self, data: DataFolder):
        self.data = data
        self.frame_index = 0
        self._cached_index = -1
        self._cached_image = None

    def current_image(self):
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
