import threading
from typing import Callable, Optional

import cv2
# hello_imgui is registered into sys.modules dynamically at runtime (imgui_bundle/__init__.py's
# _publish()), not via a normal static submodule -- Pylance can't verify that against source,
# even though its real .pyi stub (imgui_bundle/hello_imgui.pyi) is still used for type info.
from imgui_bundle import hello_imgui, imgui  # pyright: ignore[reportMissingModuleSource]

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult
from slam.optical_flow import OpticalFlowResult, OpticalFlowSolver
from slam.stereo_matching import StereoMatchingResult
from ui.utils import image_to_texture

_TRACK_COLOR = (0, 255, 0)
_TRAIL_COLOR = (0, 200, 255)


class OpticalFlowViewModel:
    def __init__(self, data: EuRoCMAVData, on_result: Callable[[OpticalFlowResult], None]) -> None:
        self._data = data
        self._on_result = on_result
        self._cancel_event = threading.Event()
        self._solver: Optional[OpticalFlowSolver] = None
        self._result: Optional[OpticalFlowResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self.frame_index: int = 0
        self.trail_length: int = 10
        self._cached_index: int = -1
        self._cached_trail_length: int = -1
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def start(
        self, feature_detection_result: FeatureDetectionResult, stereo_matching_result: StereoMatchingResult,
    ) -> None:
        self._solver = OpticalFlowSolver(
            self._data, feature_detection_result, stereo_matching_result, cancel_event=self._cancel_event)
        self._result = None
        self._loading = True
        self._error = None
        self.frame_index = 0
        self._cached_index = -1
        self._cached_trail_length = -1
        self._texture = None
        threading.Thread(target=self._compute, args=(self._solver,), daemon=True).start()

    def stop(self) -> None:
        self._cancel_event.set()
        # Breaks the Pipeline <-> view-model reference cycle formed by holding this bound-method
        # callback -- see the longer explanation on FeatureDetectionViewModel.stop().
        self._on_result = lambda _: None

    def _compute(self, solver: OpticalFlowSolver) -> None:
        try:
            result = solver.run()
            if self._cancel_event.is_set():
                return
            self._result = result
            self._on_result(result)
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False

    def current_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None:
            return None
        if self._cached_index != self.frame_index or self._cached_trail_length != self.trail_length:
            frame = self._result.frames[self.frame_index]
            img = cv2.imread(str(self._data.get_cam0_image_path(frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            trail_start = max(0, self.frame_index - self.trail_length)
            for tid, (u, v) in frame.track_uv.items():
                pt = (int(round(u)), int(round(v)))
                cv2.circle(img_color, pt, 3, _TRACK_COLOR, -1)
                # Trail: walk this track backward through recent frames for as long as it stayed
                # alive, so a lost-and-replenished track's trail correctly stops short rather than
                # jumping to whatever unrelated point now holds its old id.
                trail_pt = pt
                for j in range(self.frame_index - 1, trail_start - 1, -1):
                    prev_uv = self._result.frames[j].track_uv.get(tid)
                    if prev_uv is None:
                        break
                    prev_pt = (int(round(prev_uv[0])), int(round(prev_uv[1])))
                    cv2.line(img_color, trail_pt, prev_pt, _TRAIL_COLOR, 1)
                    trail_pt = prev_pt
            self._texture = image_to_texture(img_color)
            self._cached_index = self.frame_index
            self._cached_trail_length = self.trail_length
        return self._texture


def optical_flow_view(model: OpticalFlowViewModel) -> None:
    if model._solver is None:
        imgui.text("Waiting for stereo matching...")
        return
    if model._loading:
        imgui.text("Computing optical flow...")
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

    frame = model._result.frames[model.frame_index]

    tex = model.current_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##of_slider", model.frame_index, 0, n - 1)
        if changed:
            model.frame_index = new_index

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##of_frame_input", model.frame_index, step=1)
    if changed:
        model.frame_index = max(0, min(n - 1, new_index))

    imgui.set_next_item_width(200)
    changed, new_trail = imgui.slider_int("Trail length", model.trail_length, 0, 30)
    if changed:
        model.trail_length = new_trail

    imgui.text(f"Alive tracks: {len(frame.track_uv)}")
    imgui.text(f"Timestamp: {frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")
    imgui.text(f"Elapsed: {model._result.elapsed_s:.3f} s")
