import threading
from typing import Callable, Optional

import cv2
from imgui_bundle import hello_imgui, imgui

from slam.data import EuRoCMAVData
from slam.frontend import FrontendResult, FrontendSolver
from ui.utils import image_to_texture

_TRACK_COLOR = (0, 255, 0)
_TRAIL_COLOR = (0, 200, 255)


class FrontendViewModel:
    """Feature detection + stereo matching + optical flow used to be three separate views, each
    with its own solver and its own independent frame-index slider even though all three are
    indexed by the exact same frame sequence -- see FrontendSolver's docstring for why the solvers
    were merged. One shared frame index here drives all three sections' textures/stats at once.
    """

    def __init__(
        self, data: EuRoCMAVData,
        on_result: Callable[[FrontendResult], None] = lambda _: None,
        start_s: float = 0.0, duration_s: float = 5.0,
    ) -> None:
        self._data = data
        self._on_result = on_result
        self._cancel_event = threading.Event()
        self._solver = FrontendSolver(data, start_s, duration_s, cancel_event=self._cancel_event)
        self._result: Optional[FrontendResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._started: bool = False
        self.frame_index: int = 0
        self.match_index_min: int = 0
        self.match_index_max: int = 0
        self.trail_length: int = 10
        self._cached_fd_index: int = -1
        self._fd_texture: Optional[hello_imgui.TextureGpu] = None
        self._cached_sm_index: int = -1
        self._cached_sm_match_range: tuple[int, int] = (-1, -1)
        self._sm_texture: Optional[hello_imgui.TextureGpu] = None
        self._cached_of_index: int = -1
        self._cached_of_trail_length: int = -1
        self._of_texture: Optional[hello_imgui.TextureGpu] = None

    def _compute(self) -> None:
        try:
            result = self._solver.run()
            if self._cancel_event.is_set():
                return
            self._result = result
            self.match_index_max = max(0, len(result.stereo_matching.frames[0].matches) - 1)
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
        # Breaks the Pipeline <-> view-model reference cycle formed by holding this bound-method
        # callback -- see FeatureDetectionViewModel.stop()'s longer historical explanation (now
        # folded into this view), still relevant: a cycle can only be freed by the cyclic GC, which
        # may run on a non-GL thread and segfault on a texture destructor's glDeleteTextures call.
        self._on_result = lambda _: None

    def set_frame_index(self, index: int) -> None:
        self.frame_index = index
        self.match_index_min = 0
        if self._result is not None:
            self.match_index_max = max(0, len(self._result.stereo_matching.frames[index].matches) - 1)

    def current_fd_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None:
            return None
        if self._cached_fd_index != self.frame_index:
            frame = self._result.feature_detection.frames[self.frame_index]
            img = cv2.imread(str(self._data.get_cam0_image_path(frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            img_with_kp = cv2.drawKeypoints(
                img, frame.cam0_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self._fd_texture = image_to_texture(img_with_kp)
            self._cached_fd_index = self.frame_index
        return self._fd_texture

    def current_sm_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None:
            return None
        match_range = (self.match_index_min, self.match_index_max)
        if self._cached_sm_index != self.frame_index or self._cached_sm_match_range != match_range:
            sm_frame = self._result.stereo_matching.frames[self.frame_index]
            fd_frame = self._result.feature_detection.frames[self.frame_index]
            cam0_img = cv2.imread(str(self._data.get_cam0_image_path(sm_frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            cam1_img = cv2.imread(str(self._data.get_cam1_image_path(sm_frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            matches = sm_frame.matches[self.match_index_min:self.match_index_max + 1]
            img_matches = cv2.drawMatches(
                cam0_img, fd_frame.cam0_keypoints,
                cam1_img, fd_frame.cam1_keypoints,
                matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
            self._sm_texture = image_to_texture(img_matches)
            self._cached_sm_index = self.frame_index
            self._cached_sm_match_range = match_range
        return self._sm_texture

    def current_of_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None:
            return None
        if self._cached_of_index != self.frame_index or self._cached_of_trail_length != self.trail_length:
            of_frames = self._result.optical_flow.frames
            frame = of_frames[self.frame_index]
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
                    prev_uv = of_frames[j].track_uv.get(tid)
                    if prev_uv is None:
                        break
                    prev_pt = (int(round(prev_uv[0])), int(round(prev_uv[1])))
                    cv2.line(img_color, trail_pt, prev_pt, _TRAIL_COLOR, 1)
                    trail_pt = prev_pt
            self._of_texture = image_to_texture(img_color)
            self._cached_of_index = self.frame_index
            self._cached_of_trail_length = self.trail_length
        return self._of_texture


def frontend_view(model: FrontendViewModel) -> None:
    if model._loading:
        imgui.text("Computing frontend (feature detection + stereo matching + optical flow)...")
        imgui.progress_bar(model._solver.progress, (-1, 0))
        return
    if model._error:
        imgui.text(f"Error: {model._error}")
        return
    if model._result is None:
        return

    result = model._result
    data = model._data
    n = len(result.feature_detection.frames)
    first_ts_ns = data.cam_timestamps_ns[0]

    fd_frame = result.feature_detection.frames[model.frame_index]
    sm_frame = result.stereo_matching.frames[model.frame_index]
    of_frame = result.optical_flow.frames[model.frame_index]

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##frontend_frame_input", model.frame_index, step=1)
    if changed:
        model.set_frame_index(max(0, min(n - 1, new_index)))

    imgui.text(f"Timestamp: {fd_frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(fd_frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")
    imgui.text(f"Elapsed: {result.elapsed_s:.3f} s")

    imgui.begin_child("##frontend_scroll", (0, 0), False)

    imgui.separator()
    imgui.text("Feature Detection")
    tex = model.current_fd_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##fd_slider", model.frame_index, 0, n - 1)
        if changed:
            model.set_frame_index(new_index)
    imgui.text(f"cam0 keypoints: {len(fd_frame.cam0_keypoints)}")
    imgui.text(f"cam1 keypoints: {len(fd_frame.cam1_keypoints)}")

    imgui.separator()
    imgui.text("Stereo Matching")
    num_matches = len(sm_frame.matches)
    if num_matches > 0:
        max_idx = num_matches - 1
        imgui.text("Match range")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_min, new_min = imgui.input_int("##sm_match_min", model.match_index_min, step=1)
        if changed_min:
            model.match_index_min = max(0, min(model.match_index_max, new_min))
        imgui.same_line()
        imgui.text("to")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_max, new_max = imgui.input_int("##sm_match_max", model.match_index_max, step=1)
        if changed_max:
            model.match_index_max = max(model.match_index_min, min(max_idx, new_max))
    tex = model.current_sm_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.text(f"Stereo matches: {num_matches}")
    imgui.text(f"3D points: {sm_frame.points_3d.shape[1]}")

    imgui.separator()
    imgui.text("Optical Flow")
    imgui.set_next_item_width(200)
    changed, new_trail = imgui.slider_int("Trail length", model.trail_length, 0, 30)
    if changed:
        model.trail_length = new_trail
    tex = model.current_of_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.text(f"Alive tracks: {len(of_frame.track_uv)}")

    imgui.end_child()
