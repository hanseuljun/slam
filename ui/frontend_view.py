from typing import Callable, Optional

import cv2
import numpy as np
from imgui_bundle import hello_imgui, imgui

from slam.data import EuRoCMAVData
from slam.frontend import FrontendResult, KeyframeFeatures
from slam.slam import SlamSolver
from ui.utils import image_to_texture

_TRACK_COLOR = (0, 255, 0)
_NO_STEREO_COLOR = (0, 0, 255)
_TRAIL_COLOR = (0, 200, 255)
_STEREO_LINE_COLOR = (255, 200, 0)


class FrontendViewModel:
    """Read-only view over the frontend data SlamSolver produces as it runs
    (SlamSolver.frontend_result): KLT tracks with stereo depth for every frame, and ORB features
    for keyframes only (computed by loop closure). All share one frame index.

    `get_solver` returns the current SlamSolver (None before SlamViewModel.start()). Frames appear
    as the solver processes them.
    """

    def __init__(self, data: EuRoCMAVData, get_solver: Callable[[], Optional[SlamSolver]]) -> None:
        self._data = data
        self._get_solver = get_solver
        self.frame_index: int = 0
        self.match_index_min: int = 0
        self.match_index_max: int = 0
        self._match_range_frame: int = -1  # frame_index match_index_min/max were last reset for
        self.trail_length: int = 10
        self._cached_orb_index: int = -1
        self._orb_texture: Optional[hello_imgui.TextureGpu] = None
        self._cached_stereo_index: int = -1
        self._cached_stereo_match_range: tuple[int, int] = (-1, -1)
        self._stereo_texture: Optional[hello_imgui.TextureGpu] = None
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
        result = self._result
        return 0 if result is None else len(result.optical_flow.frames)

    def keyframe_features(self) -> Optional[KeyframeFeatures]:
        """ORB features of the shown frame, or None if it isn't a keyframe (or loop closure is off,
        which is what computes them).
        """
        result = self._result
        return None if result is None else result.keyframe_features.get(self.frame_index)

    def set_frame_index(self, index: int) -> None:
        self.frame_index = index

    def sync_match_range(self, num_matches: int) -> None:
        """Reset the stereo match range to "all matches" whenever the shown frame changed."""
        if self._match_range_frame != self.frame_index:
            self.match_index_min = 0
            self.match_index_max = max(0, num_matches - 1)
            self._match_range_frame = self.frame_index

    def current_orb_texture(self) -> Optional[hello_imgui.TextureGpu]:
        features = self.keyframe_features()
        if features is None:
            return None
        if self._cached_orb_index != self.frame_index:
            frame = features.fd
            img = cv2.imread(str(self._data.get_cam0_image_path(frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            img_with_kp = cv2.drawKeypoints(
                img, frame.cam0_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self._orb_texture = image_to_texture(img_with_kp)
            self._cached_orb_index = self.frame_index
        return self._orb_texture

    def current_stereo_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None:
            return None
        match_range = (self.match_index_min, self.match_index_max)
        if self._cached_stereo_index != self.frame_index or self._cached_stereo_match_range != match_range:
            frame = self._result.optical_flow.frames[self.frame_index]
            cam0_img = cv2.imread(str(self._data.get_cam0_image_path(frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            cam1_img = cv2.imread(str(self._data.get_cam1_image_path(frame.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            width = cam0_img.shape[1]
            canvas = cv2.cvtColor(np.hstack([cam0_img, cam1_img]), cv2.COLOR_GRAY2BGR)
            tids = list(frame.stereo_uv1)[self.match_index_min:self.match_index_max + 1]
            for tid in tids:
                u0, v0 = frame.track_uv[tid]
                u1, v1 = frame.stereo_uv1[tid]
                p0 = (int(round(u0)), int(round(v0)))
                p1 = (int(round(u1)) + width, int(round(v1)))
                cv2.line(canvas, p0, p1, _STEREO_LINE_COLOR, 1)
                cv2.circle(canvas, p0, 3, _TRACK_COLOR, -1)
                cv2.circle(canvas, p1, 3, _TRACK_COLOR, -1)
            self._stereo_texture = image_to_texture(canvas)
            self._cached_stereo_index = self.frame_index
            self._cached_stereo_match_range = match_range
        return self._stereo_texture

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
                cv2.circle(img_color, pt, 3, _TRACK_COLOR if tid in frame.stereo_uv1 else _NO_STEREO_COLOR, -1)
                # Trail: walk this track backward while it stayed alive; ids are never reused, so a
                # lost track's trail simply ends.
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

    of_frame = result.optical_flow.frames[model.frame_index]
    num_stereo = len(of_frame.stereo_uv1)
    model.sync_match_range(num_stereo)

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##frontend_frame_input", model.frame_index, step=1)
    if changed:
        model.set_frame_index(max(0, min(n - 1, new_index)))

    imgui.text(f"Timestamp: {of_frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(of_frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")
    if not solver.loading:
        imgui.text(f"Elapsed (whole SLAM pass, frontend included): {result.elapsed_s:.3f} s")

    imgui.begin_child("##frontend_scroll", (0, 0), False)

    imgui.separator()
    imgui.text("Optical Flow (KLT tracks)")
    imgui.set_next_item_width(200)
    changed, new_trail = imgui.slider_int("Trail length", model.trail_length, 0, 30)
    if changed:
        model.trail_length = new_trail
    tex = model.current_of_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##of_slider", model.frame_index, 0, n - 1)
        if changed:
            model.set_frame_index(new_index)
    imgui.text(f"Alive tracks: {len(of_frame.track_uv)} (green: with stereo, red: cam0 only)")

    imgui.separator()
    imgui.text("Stereo (tracks matched into cam1 by KLT)")
    if num_stereo > 0:
        max_idx = num_stereo - 1
        imgui.text("Match range")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_min, new_min = imgui.input_int("##stereo_match_min", model.match_index_min, step=1)
        if changed_min:
            model.match_index_min = max(0, min(model.match_index_max, new_min))
        imgui.same_line()
        imgui.text("to")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_max, new_max = imgui.input_int("##stereo_match_max", model.match_index_max, step=1)
        if changed_max:
            model.match_index_max = max(model.match_index_min, min(max_idx, new_max))
    tex = model.current_stereo_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.text(f"Stereo tracks: {num_stereo}")

    imgui.separator()
    imgui.text("ORB features (keyframes only, for loop closure)")
    features = model.keyframe_features()
    if features is None:
        imgui.text("Not a keyframe, or loop closure is off.")
    else:
        tex = model.current_orb_texture()
        if tex is not None:
            imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.text(f"cam0 keypoints: {len(features.fd.cam0_keypoints)}")
        imgui.text(f"cam1 keypoints: {len(features.fd.cam1_keypoints)}")
        imgui.text(f"ORB stereo matches: {len(features.sm.matches)}")

    imgui.end_child()
