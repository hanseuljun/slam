import json
import threading
from typing import Callable, Optional

import cv2
import numpy as np
# hello_imgui is registered into sys.modules dynamically at runtime (imgui_bundle/__init__.py's
# _publish()), not via a normal static submodule -- Pylance can't verify that against source,
# even though its real .pyi stub (imgui_bundle/hello_imgui.pyi) is still used for type info.
from imgui_bundle import imgui, hello_imgui, implot  # pyright: ignore[reportMissingModuleSource]

from slam.coordinate_mapping_check import (
    CoordinateMappingCheckResult,
    CoordinateMappingChecker,
)
from slam.data import EuRoCMAVData
from slam.optical_flow import OpticalFlowResult
from slam.slam import SlamSolver
from ui.utils import image_to_texture


def _match_color(i: int, total: int) -> tuple[int, int, int]:
    hue = int(i * 180 / max(total, 1)) % 180
    color_hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))


def _draw_result(result: CoordinateMappingCheckResult) -> None:
    times = np.ascontiguousarray(result.times, dtype=np.float64)
    mean_errors = np.array([np.mean(f.projection_errors) if f.projection_errors else float('nan') for f in result.frames])
    mean_icp_errors = np.array([np.mean(f.icp_projection_errors) if f.icp_projection_errors else float('nan') for f in result.frames])
    num_matches = np.array([len(f.track_ids) for f in result.frames], dtype=np.float64)

    imgui.text('Coordinate Mapping Check (Ground Truth Poses)')
    if implot.begin_subplots("##cm_plot", 2, 1, size=(-1, 520), flags=implot.SubplotFlags_.link_all_x):
        if implot.begin_plot("Mean Projection Error per Adjacent Frame Pair"):
            implot.setup_axes(
                'Time [s]', 'Projection Error [px]',
                implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.plot_line('GT', times, np.ascontiguousarray(mean_errors, dtype=np.float64))
            implot.plot_line('ICP', times, np.ascontiguousarray(mean_icp_errors, dtype=np.float64))
            implot.end_plot()
        if implot.begin_plot("Number of Matched Features per Adjacent Frame Pair"):
            implot.setup_axes(
                'Time [s]', 'Count', implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            implot.plot_line('matches', times, num_matches)
            implot.end_plot()
        implot.end_subplots()


class CoordinateMappingViewModel:
    """Starts itself (maybe_start, polled by the view) once SLAM has finished, since it needs the
    complete frontend output SlamSolver.frontend_result ends up holding -- there is no separate
    frontend pass to wait on.
    """

    def __init__(self, data: EuRoCMAVData, get_solver: Callable[[], Optional[SlamSolver]]) -> None:
        self._data = data
        self._get_solver = get_solver
        self._started: bool = False
        self._cancel_event = threading.Event()
        self._checker: Optional[CoordinateMappingChecker] = None
        self._optical_flow_result: Optional[OpticalFlowResult] = None
        self._result: Optional[CoordinateMappingCheckResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self.frame_index: int = 0
        self.match_index_min: int = 0
        self.match_index_max: int = 0
        self.show_projected: bool = True
        self._cache_key: tuple = ()
        self._match_texture: Optional[hello_imgui.TextureGpu] = None

    def maybe_start(self) -> None:
        if self._started:
            return
        solver = self._get_solver()
        if solver is None or solver.loading or solver.error:
            return
        self.start(solver.frontend_result.optical_flow)

    def start(self, optical_flow_result: OpticalFlowResult) -> None:
        self._started = True
        self._optical_flow_result = optical_flow_result
        self._checker = CoordinateMappingChecker(self._data, optical_flow_result, cancel_event=self._cancel_event)
        self._result = None
        self._loading = True
        self._error = None
        self.frame_index = 0
        self.match_index_min = 0
        self.match_index_max = 0
        self.show_projected = True
        self._cache_key = ()
        self._match_texture = None
        threading.Thread(target=self._compute, args=(self._checker,), daemon=True).start()

    def stop(self) -> None:
        self._cancel_event.set()

    def _compute(self, checker: CoordinateMappingChecker) -> None:
        try:
            result = checker.run()
            if self._cancel_event.is_set():
                return
            self._result = result
            self.match_index_max = max(0, len(self._result.frames[0].track_ids) - 1)
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False

    def current_match_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None or self._optical_flow_result is None:
            return None
        cache_key = (self.frame_index, self.match_index_min, self.match_index_max, self.show_projected)
        if self._cache_key != cache_key:
            frame = self._result.frames[self.frame_index]
            of_k = self._optical_flow_result.frames[self.frame_index]
            of_k1 = self._optical_flow_result.frames[self.frame_index + 1]

            cam0_img_k = cv2.imread(
                str(self._data.get_cam0_image_path(of_k.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            cam0_img_k1 = cv2.imread(
                str(self._data.get_cam0_image_path(of_k1.timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            offset_x = cam0_img_k.shape[1]
            canvas = cv2.cvtColor(np.hstack([cam0_img_k, cam0_img_k1]), cv2.COLOR_GRAY2BGR)

            tids = frame.track_ids[self.match_index_min:self.match_index_max + 1]
            projected = frame.projected_points[self.match_index_min:self.match_index_max + 1]
            total = len(tids)
            for i, tid in enumerate(tids):
                color = _match_color(i, total)
                u0, v0 = of_k.track_uv[tid]
                u1, v1 = of_k1.track_uv[tid]
                p0 = (int(round(u0)), int(round(v0)))
                p1 = (int(round(u1)) + offset_x, int(round(v1)))
                cv2.line(canvas, p0, p1, color, 1)
                cv2.circle(canvas, p0, 3, color, -1)
                cv2.circle(canvas, p1, 3, color, -1)

            if self.show_projected:
                for i, pt in enumerate(projected):
                    x, y = int(round(pt[0])) + offset_x, int(round(pt[1]))
                    cv2.drawMarker(canvas, (x, y), _match_color(i, total), cv2.MARKER_TILTED_CROSS,
                                   markerSize=12, thickness=1)

            self._match_texture = image_to_texture(canvas)
            self._cache_key = cache_key
        return self._match_texture


def _download_transforms(result: CoordinateMappingCheckResult) -> None:
    records = []
    for f in result.frames:
        records.append({
            "timestamp_ns": f.timestamp_ns,
            "gt_sample_index": f.gt_sample_index,
            "gt_transform": f.gt_transform.tolist(),
            "icp_transform": f.icp_transform.tolist() if f.icp_transform is not None else None,
        })
    import os
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/coordinate_mapping_check.json"
    with open(path, "w") as fp:
        json.dump(records, fp, indent=2)
    print(f"Saved {len(records)} records to {path}")


def coordinate_mapping_view(model: CoordinateMappingViewModel) -> None:
    model.maybe_start()
    if model._checker is None:
        solver = model._get_solver()
        if solver is not None and solver.error:
            imgui.text(f"SLAM failed: {solver.error}")
        else:
            imgui.text("Waiting for SLAM to finish (this check needs its complete frontend output)...")
        return
    if model._loading:
        imgui.text("Computing coordinate mapping check...")
        imgui.progress_bar(model._checker.progress, (-1, 0))
        return
    if model._error:
        imgui.text(f"Error: {model._error}")
        return
    if model._result is None:
        return

    n = len(model._result.frames)
    first_ts_ns = model._data.cam_timestamps_ns[0]

    frame = model._result.frames[model.frame_index]
    num_matches = len(frame.track_ids)

    tex = model.current_match_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##cm_slider", model.frame_index, 0, n - 1)
        if changed:
            model.frame_index = new_index
            model.match_index_min = 0
            model.match_index_max = max(0, len(model._result.frames[new_index].track_ids) - 1)

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##cm_frame_input", model.frame_index, step=1)
    if changed:
        model.frame_index = max(0, min(n - 1, new_index))
        model.match_index_min = 0
        model.match_index_max = max(0, len(model._result.frames[model.frame_index].track_ids) - 1)

    if num_matches > 0:
        max_idx = num_matches - 1
        imgui.text("Match range")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_min, new_min = imgui.input_int("##cm_match_min", model.match_index_min, step=1)
        if changed_min:
            model.match_index_min = max(0, min(model.match_index_max, new_min))
        imgui.same_line()
        imgui.text("to")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_max, new_max = imgui.input_int("##cm_match_max", model.match_index_max, step=1)
        if changed_max:
            model.match_index_max = max(model.match_index_min, min(max_idx, new_max))
        imgui.same_line()
        _, model.show_projected = imgui.checkbox("Show projected", model.show_projected)

    imgui.text(f"Matches: {num_matches}")
    mean_err = np.mean(frame.projection_errors) if frame.projection_errors else float('nan')
    imgui.text(f"Mean projection error (GT):  {mean_err:.2f} px")
    mean_icp_err = np.mean(frame.icp_projection_errors) if frame.icp_projection_errors else float('nan')
    imgui.text(f"Mean projection error (ICP): {mean_icp_err:.2f} px")
    imgui.text(f"Timestamp: {frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")

    imgui.separator()

    if imgui.button("Download coordinate_mapping_check.json"):
        _download_transforms(model._result)

    imgui.separator()

    _draw_result(model._result)
