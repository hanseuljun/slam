import threading
from typing import Callable, Optional

import cv2
from imgui_bundle import hello_imgui, imgui

from slam.data import DataFolder
from slam.feature_detection import FeatureDetectionResult
from slam.stereo_matching import StereoMatchingResult, StereoMatchingSolver
from ui.utils import image_to_texture


class StereoMatchingViewModel:
    def __init__(self, data: DataFolder, on_result: Callable[[StereoMatchingResult], None]) -> None:
        self._data = data
        self._on_result = on_result
        self._solver: Optional[StereoMatchingSolver] = None
        self._feature_detection_result: Optional[FeatureDetectionResult] = None
        self._result: Optional[StereoMatchingResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self.frame_index: int = 0
        self._cached_index: int = -1
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def start(self, feature_detection_result: FeatureDetectionResult) -> None:
        self._feature_detection_result = feature_detection_result
        self._solver = StereoMatchingSolver(self._data, feature_detection_result)
        self._result = None
        self._loading = True
        self._error = None
        self._cached_index = -1
        self._texture = None
        threading.Thread(target=self._compute, daemon=True).start()

    def _compute(self) -> None:
        try:
            result = self._solver.run()
            self._result = result
            self._on_result(result)
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False

    def current_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None or self._feature_detection_result is None:
            return None
        if self._cached_index != self.frame_index:
            self._texture = None
            sm_frame = self._result.frames[self.frame_index]
            fd_frame = self._feature_detection_result.frames[self.frame_index]
            cam0_img = cv2.imread(
                str(self._data.get_cam0_image_path(sm_frame.timestamp_ns)),
                cv2.IMREAD_GRAYSCALE,
            )
            cam1_img = cv2.imread(
                str(self._data.get_cam1_image_path(sm_frame.timestamp_ns)),
                cv2.IMREAD_GRAYSCALE,
            )
            img_matches = cv2.drawMatches(
                cam0_img, fd_frame.cam0_keypoints,
                cam1_img, fd_frame.cam1_keypoints,
                sm_frame.matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            self._texture = image_to_texture(img_matches)
            self._cached_index = self.frame_index
        return self._texture


def stereo_matching_view(model: StereoMatchingViewModel) -> None:
    if model._solver is None:
        imgui.text("Waiting for feature detection...")
        return
    if model._loading:
        imgui.text("Computing stereo matches...")
        imgui.progress_bar(model._solver.progress, (-1, 0))
        return
    if model._error:
        imgui.text(f"Error: {model._error}")
        return
    if model._result is None:
        return

    data = model._data
    n = len(model._result.frames)
    first_ts_ns = data.cam_timestamps_ns[0]

    tex = model.current_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##sm_slider", model.frame_index, 0, n - 1)
        if changed:
            model.frame_index = new_index

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##sm_frame_input", model.frame_index, step=1)
    if changed:
        model.frame_index = max(0, min(n - 1, new_index))

    frame = model._result.frames[model.frame_index]
    imgui.text(f"Stereo matches: {len(frame.matches)}")
    imgui.text(f"3D points: {frame.points_3d.shape[1]}")
    imgui.text(f"Timestamp: {frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")
