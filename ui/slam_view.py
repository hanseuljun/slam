from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.data import DataFolder
from slam.feature_detection import FeatureDetectionResult
from slam.slam_solver import SlamResults, SlamSolver
from slam.stereo_matching import StereoMatchingResult
from ui.utils import figure_to_image, image_to_texture


def _plot_positions(series: list[tuple[np.ndarray, np.ndarray, str]]) -> plt.Figure:
    fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Position')
    for ax, i, label in zip([ax_x, ax_y, ax_z], range(3), ['X', 'Y', 'Z']):
        for times, positions, name in series:
            ax.plot(times, positions[:, i], label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [m]')
        ax.legend()
    plt.tight_layout()
    return fig


def _plot_attitudes(series: list[tuple[np.ndarray, np.ndarray, str]]) -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle('Rotation Axes')
    axis_names = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
    component_names = ['X', 'Y', 'Z']
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            for times, attitudes, name in series:
                ax.plot(times, attitudes[:, col, row], label=name)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{axis_names[row]} {component_names[col]}')
            ax.legend()
    plt.tight_layout()
    return fig


def _plot_angular_velocities(series: list[tuple[np.ndarray, np.ndarray, str]]) -> plt.Figure:
    fig, (ax_wx, ax_wy, ax_wz) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Angular Velocity in Body Frame')
    for ax, i, label in zip([ax_wx, ax_wy, ax_wz], range(3), ['wx', 'wy', 'wz']):
        for times, angular_velocities, name in series:
            ax.plot(times, angular_velocities[:, i], label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [rad/s]')
        ax.legend()
    plt.tight_layout()
    return fig


def _render_positions(results: SlamResults, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.pnp_times, results.pnp_positions, 'pnp'),
        (results.gt_times, results.gt_positions, 'gt'),
    ]
    return figure_to_image(_plot_positions([s for s in all_series if enabled[s[2]]]))


def _render_attitudes(results: SlamResults, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.pnp_times, results.pnp_attitudes, 'pnp'),
        (results.imu_attitude_times, results.imu_attitudes, 'imu'),
        (results.gt_times, results.gt_attitudes, 'gt'),
        (results.pnp_times, results.optimized_attitudes, 'opt'),
    ]
    return figure_to_image(_plot_attitudes([s for s in all_series if enabled[s[2]]]))


def _render_angular_velocities(results: SlamResults, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.pnp_angular_velocity_times, results.pnp_angular_velocities, 'pnp'),
        (results.imu_times, results.imu_angular_velocities, 'imu'),
        (results.pnp_times, results.imu_angular_velocities_at_cam_times, 'imu@cam'),
        (results.gt_angular_velocity_times, results.gt_angular_velocities, 'gt'),
        (results.pnp_angular_velocity_times, results.optimized_angular_velocities, 'opt'),
    ]
    return figure_to_image(_plot_angular_velocities([s for s in all_series if enabled[s[2]]]))


class SlamViewModel:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._solver: Optional[SlamSolver] = None
        self._last_solver: Optional[SlamSolver] = None
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_angular_velocities: Optional[hello_imgui.TextureGpu] = None
        self.pos_enabled: dict[str, bool] = {'pnp': True, 'gt': True}
        self.att_enabled: dict[str, bool] = {'pnp': True, 'imu': True, 'gt': True, 'opt': True}
        self.omega_enabled: dict[str, bool] = {'pnp': True, 'imu': True, 'imu@cam': True, 'gt': True, 'opt': True}

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


def _checkboxes(enabled: dict[str, bool], id_suffix: str) -> bool:
    changed = False
    labels = list(enabled)
    for i, label in enumerate(labels):
        c, enabled[label] = imgui.checkbox(f"{label}##{id_suffix}", enabled[label])
        changed = changed or c
        if i < len(labels) - 1:
            imgui.same_line()
    return changed


def slam_view(model: SlamViewModel) -> None:
    if model._last_solver is not model._solver:
        model._tex_positions = None
        model._tex_attitudes = None
        model._tex_angular_velocities = None
        model._last_solver = model._solver

    solver = model._solver
    if solver is None:
        imgui.text("Waiting for stereo matching...")
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

    results = solver.plots

    imgui.begin_child("##slam_scroll", (0, 0), False)

    if _checkboxes(model.pos_enabled, "pos"):
        model._tex_positions = None
    if model._tex_positions is None:
        model._tex_positions = image_to_texture(_render_positions(results, model.pos_enabled))
    tex = model._tex_positions
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))

    if _checkboxes(model.att_enabled, "att"):
        model._tex_attitudes = None
    if model._tex_attitudes is None:
        model._tex_attitudes = image_to_texture(_render_attitudes(results, model.att_enabled))
    tex = model._tex_attitudes
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))

    if _checkboxes(model.omega_enabled, "omega"):
        model._tex_angular_velocities = None
    if model._tex_angular_velocities is None:
        model._tex_angular_velocities = image_to_texture(_render_angular_velocities(results, model.omega_enabled))
    tex = model._tex_angular_velocities
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))

    imgui.end_child()
