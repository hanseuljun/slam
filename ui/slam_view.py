from typing import Optional

import matplotlib
matplotlib.use('Agg')
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.data import DataFolder
from slam.feature_detection import FeatureDetectionResult
from slam.plot import plot_angular_velocities, plot_attitudes, plot_positions
from slam.slam_solver import SlamResults, SlamSolver
from slam.stereo_matching import StereoMatchingResult
from ui.utils import figure_to_image, image_to_texture


def _render_plots(results: SlamResults) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fig_pos = plot_positions(series=[
        (results.pnp_times, results.pnp_positions, 'pnp'),
        (results.gt_times, results.gt_positions, 'gt'),
    ])
    fig_att = plot_attitudes(series=[
        (results.pnp_times, results.pnp_attitudes, 'pnp'),
        (results.imu_attitude_times, results.imu_attitudes, 'imu'),
        (results.gt_times, results.gt_attitudes, 'gt'),
        (results.pnp_times, results.optimized_attitudes, 'opt'),
    ])
    fig_omega = plot_angular_velocities(series=[
        (results.pnp_angular_velocity_times, results.pnp_angular_velocities, 'pnp'),
        (results.imu_times, results.imu_angular_velocities, 'imu'),
        (results.pnp_times, results.imu_angular_velocities_at_cam_times, 'imu@cam'),
        (results.gt_angular_velocity_times, results.gt_angular_velocities, 'gt'),
        (results.pnp_angular_velocity_times, results.optimized_angular_velocities, 'opt'),
    ])
    return figure_to_image(fig_pos), figure_to_image(fig_att), figure_to_image(fig_omega)


class SlamViewModel:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._solver: Optional[SlamSolver] = None
        self._last_solver: Optional[SlamSolver] = None
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_angular_velocities: Optional[hello_imgui.TextureGpu] = None

    def start(
        self,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
    ) -> None:
        if self._solver is not None:
            self._solver._stop_event.set()
        self._solver = SlamSolver(self._data, feature_detection_result, stereo_matching_result)
        self._solver.start()

    def stop(self) -> None:
        if self._solver is not None:
            self._solver._stop_event.set()
        self._solver = None


def slam_view(model: SlamViewModel) -> None:
    if model._last_solver is not model._solver:
        model._tex_positions = None
        model._tex_attitudes = None
        model._tex_angular_velocities = None
        model._last_solver = model._solver

    solver = model._solver
    if solver is None:
        imgui.text("Waiting for feature detection...")
        return
    if solver.loading:
        imgui.text(solver.progress_label)
        imgui.progress_bar(solver.progress, (-1, 0))
        return
    if solver.error:
        imgui.text(f"Error: {solver.error}")
        return
    if solver.plots is None:
        return

    if model._tex_positions is None:
        pos_img, att_img, omega_img = _render_plots(solver.plots)
        model._tex_positions = image_to_texture(pos_img)
        model._tex_attitudes = image_to_texture(att_img)
        model._tex_angular_velocities = image_to_texture(omega_img)

    imgui.begin_child("##slam_scroll", (0, 0), False)
    tex = model._tex_positions
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    tex = model._tex_attitudes
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    tex = model._tex_angular_velocities
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.end_child()
