import threading
from typing import Callable, Optional

import cv2
from imgui_bundle import hello_imgui, imgui

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult, FeatureDetectionSolver
from ui.utils import image_to_texture


class FeatureDetectionViewModel:
    def __init__(
        self,
        data: EuRoCMAVData,
        on_result: Callable[[FeatureDetectionResult], None],
        start_s: float = 0.0,
        duration_s: float = 5.0,
    ) -> None:
        self._data = data
        self._on_result = on_result
        self._cancel_event = threading.Event()
        self._solver = FeatureDetectionSolver(data, start_s, duration_s, cancel_event=self._cancel_event)
        self._result: Optional[FeatureDetectionResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._started: bool = False
        self.frame_index: int = 0
        self._cached_index: int = -1
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def _compute(self) -> None:
        try:
            result = self._solver.run()
            if self._cancel_event.is_set():
                return
            self._result = result
            self._on_result(result)
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._loading = True
        threading.Thread(target=self._compute, daemon=True).start()

    def stop(self) -> None:
        self._cancel_event.set()
        # Drop the callback into the owning Pipeline: it's a bound method, so holding onto it
        # keeps Pipeline <-> this view model alive as a reference cycle even after the Pipeline
        # itself is replaced on dataset switch. A cycle can only be freed by the *cyclic* GC, which
        # runs on whatever thread happens to trigger it -- e.g. a ThreadPoolExecutor worker from
        # the next dataset's stereo matching -- and its texture destructors call glDeleteTextures,
        # which segfaults off the main/GL thread. Clearing this breaks the cycle so everything
        # (including any cached texture) is freed by plain refcounting, synchronously, on
        # whichever thread calls stop() -- restart() calls it from the main thread.
        self._on_result = lambda _: None

    def current_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None:
            return None
        if self._cached_index != self.frame_index:
            self._texture = None
            frame = self._result.frames[self.frame_index]
            img = cv2.imread(
                str(self._data.get_cam0_image_path(frame.timestamp_ns)),
                cv2.IMREAD_GRAYSCALE,
            )
            img_with_kp = cv2.drawKeypoints(
                img, frame.cam0_keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            self._texture = image_to_texture(img_with_kp)
            self._cached_index = self.frame_index
        return self._texture


def feature_detection_view(model: FeatureDetectionViewModel) -> None:
    if model._loading:
        imgui.text("Detecting features...")
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
        changed, new_index = imgui.slider_int("##fd_slider", model.frame_index, 0, n - 1)
        if changed:
            model.frame_index = new_index

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##fd_frame_input", model.frame_index, step=1)
    if changed:
        model.frame_index = max(0, min(n - 1, new_index))

    frame = model._result.frames[model.frame_index]
    imgui.text(f"cam0 keypoints: {len(frame.cam0_keypoints)}")
    imgui.text(f"cam1 keypoints: {len(frame.cam1_keypoints)}")
    imgui.text(f"Timestamp: {frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")
    imgui.text(f"Elapsed: {model._result.elapsed_s:.3f} s")
