from typing import Callable, Optional

import cv2
from imgui_bundle import hello_imgui, imgui

from slam.data import EuRoCMAVData
from slam.frontend import FrontendResult
from slam.slam import SlamSolver
from ui.utils import image_to_texture

_TRACK_COLOR = (0, 255, 0)
_TRAIL_COLOR = (0, 200, 255)


class FrontendViewModel:
    """Read-only view over the frontend data SlamSolver produces as it runs
    (SlamSolver.frontend_result) -- no frontend computation of its own. Feature detection, stereo
    matching and optical flow are indexed by the exact same frame sequence, so one shared frame
    index here drives all three sections' textures/stats at once.

    `get_solver` returns the current SlamSolver (None before SlamViewModel.start()). Frames show up
    as the solver processes them, so the tab is usable while SLAM is still running.
    """

    def __init__(self, data: EuRoCMAVData, get_solver: Callable[[], Optional[SlamSolver]]) -> None:
        self._data = data
        self._get_solver = get_solver
        self.frame_index: int = 0
        self.match_index_min: int = 0
        self.match_index_max: int = 0
        self._match_range_frame: int = -1  # frame_index match_index_min/max were last reset for
        self.trail_length: int = 10
        self._cached_fd_index: int = -1
        self._fd_texture: Optional[hello_imgui.TextureGpu] = None
        self._cached_sm_index: int = -1
        self._cached_sm_match_range: tuple[int, int] = (-1, -1)
        self._sm_texture: Optional[hello_imgui.TextureGpu] = None
        self._cached_of_index: int = -1
        self._cached_of_trail_length: int = -1
        self._of_texture: Optional[hello_imgui.TextureGpu] = None

    @property
    def solver(self) -> Optional[SlamSolver]:
        return self._get_solver()

    @property
    def _result(self) -> Optional[FrontendResult]:
        solver = self.solver
        return solver.frontend_result if solver is not None else None

    def num_ready_frames(self) -> int:
        """Frames whose feature detection, stereo matching and optical flow are all present --
        the three lists are appended one after another by the solver thread, so a reader must
        only index below the shortest."""
        result = self._result
        if result is None:
            return 0
        return min(len(result.feature_detection.frames), len(result.stereo_matching.frames),
                   len(result.optical_flow.frames))

    def set_frame_index(self, index: int) -> None:
        self.frame_index = index

    def sync_match_range(self, num_matches: int) -> None:
        """Reset the stereo match range to "all matches" whenever the shown frame changed."""
        if self._match_range_frame != self.frame_index:
            self.match_index_min = 0
            self.match_index_max = max(0, num_matches - 1)
            self._match_range_frame = self.frame_index

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
    solver = model.solver
    result = model._result
    if solver is None or result is None:
        imgui.text("Waiting for SLAM to start...")
        return
    if solver.error:
        imgui.text(f"Error: {solver.error}")
        return
    # Same reason as in slam_view: idling would stop this from being redrawn to notice new frames.
    hello_imgui.get_runner_params().fps_idling.enable_idling = not solver.loading
    n = model.num_ready_frames()
    if n == 0:
        imgui.text("Waiting for the first frame...")
        return
    if solver.loading:
        imgui.text(f"Frontend running as part of SLAM: {n} frames so far")
        imgui.progress_bar(solver.progress, (-1, 0))

    data = model._data
    first_ts_ns = data.cam_timestamps_ns[0]
    model.frame_index = min(model.frame_index, n - 1)

    fd_frame = result.feature_detection.frames[model.frame_index]
    sm_frame = result.stereo_matching.frames[model.frame_index]
    of_frame = result.optical_flow.frames[model.frame_index]
    model.sync_match_range(len(sm_frame.matches))

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##frontend_frame_input", model.frame_index, step=1)
    if changed:
        model.set_frame_index(max(0, min(n - 1, new_index)))

    imgui.text(f"Timestamp: {fd_frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(fd_frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")
    if not solver.loading:
        imgui.text(f"Elapsed (whole SLAM pass, frontend included): {result.elapsed_s:.3f} s")

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
