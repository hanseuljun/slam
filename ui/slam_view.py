import threading
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.data import EuRoCMAVData
from slam.feature_detection import FeatureDetectionResult
from slam.imu_initialization import ImuInitializationResult
from slam.slam import SlamResult, SlamSolver
from slam.stereo_matching import StereoMatchingResult
from ui.utils import figure_to_image, image_to_texture


def _plot_positions(series: list[tuple[np.ndarray, np.ndarray, str]]) -> plt.Figure:
    fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Position in World Frame')
    for ax, i, label in zip([ax_x, ax_y, ax_z], range(3), ['X', 'Y', 'Z']):
        for times, positions, name in series:
            ax.plot(times, positions[:, i], label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [m]')
        ax.legend()
    plt.tight_layout()
    return fig


def _plot_attitudes(series: list[tuple[np.ndarray, np.ndarray, str]], title: str = 'Attitude (Rotation Vector) in World Frame') -> plt.Figure:
    fig, (ax_x, ax_y, ax_z) = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)
    for ax, i, label in zip([ax_x, ax_y, ax_z], range(3), ['X', 'Y', 'Z']):
        for times, attitudes, name in series:
            ax.plot(times, attitudes[:, i], label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [rad]')
        ax.legend()
    plt.tight_layout()
    return fig


def _plot_rotation_matrices(series: list[tuple[np.ndarray, np.ndarray, str]], title: str = 'Rotation Axes in World Frame') -> plt.Figure:
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle(title)
    axis_names = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
    component_names = ['X', 'Y', 'Z']
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            for times, rotation_matrices, name in series:
                ax.plot(times, rotation_matrices[:, col, row], label=name)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{axis_names[row]} {component_names[col]}')
            ax.legend()
    plt.tight_layout()
    return fig


def _plot_linear_accelerations(series: list[tuple[np.ndarray, np.ndarray, str]], title: str = 'Linear Acceleration in World Frame', gravity: np.ndarray | None = None) -> plt.Figure:
    fig, (ax_ax, ax_ay, ax_az) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(title)
    for ax, i, label in zip([ax_ax, ax_ay, ax_az], range(3), ['ax', 'ay', 'az']):
        for times, linear_accelerations, name in series:
            ax.plot(times, linear_accelerations[:, i], label=name)
        if gravity is not None:
            ax.axhline(-gravity[i], color='gray', linestyle='--', linewidth=1, label='-gravity')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [m/s²]')
        ax.legend()
    plt.tight_layout()
    return fig


def _plot_velocities(series: list[tuple[np.ndarray, np.ndarray, str]]) -> plt.Figure:
    fig, (ax_vx, ax_vy, ax_vz) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Velocity in World Frame')
    for ax, i, label in zip([ax_vx, ax_vy, ax_vz], range(3), ['vx', 'vy', 'vz']):
        for times, velocities, name in series:
            ax.plot(times, velocities[:, i], label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [m/s]')
        ax.legend()
    plt.tight_layout()
    return fig


def _plot_biases(series: list[tuple[np.ndarray, np.ndarray, str]], title: str = 'IMU Bias (Body Frame)') -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(title)
    row_labels = ['accel bias', 'gyro bias']
    row_units = ['m/s²', 'rad/s']
    component_names = ['x', 'y', 'z']
    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            for times, biases, name in series:
                ax.plot(times, biases[:, row * 3 + col], label=name)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{row_labels[row]} {component_names[col]} [{row_units[row]}]')
            ax.legend()
    plt.tight_layout()
    return fig


def _plot_gtsam_diagnostics(times: np.ndarray, position_errors: np.ndarray,
                            reprojection_rmse: np.ndarray, landmark_counts: np.ndarray) -> plt.Figure:
    fig, (ax_err, ax_rmse, ax_cnt) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('GTSAM Diagnostics')

    rmse = float(np.sqrt(np.mean(position_errors ** 2))) if len(position_errors) else float('nan')
    ax_err.plot(times, position_errors, color='tab:red', marker='.', label='position error')
    ax_err.axhline(rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {rmse:.3f} m')
    ax_err.set_ylabel('pos error vs GT [m]')
    ax_err.legend()

    valid = ~np.isnan(reprojection_rmse)
    mean_reproj = float(np.mean(reprojection_rmse[valid])) if np.any(valid) else float('nan')
    ax_rmse.plot(times, reprojection_rmse, color='tab:blue', marker='.', label='reprojection RMSE')
    ax_rmse.axhline(mean_reproj, color='gray', linestyle='--', linewidth=1, label=f'mean = {mean_reproj:.2f} px')
    ax_rmse.set_ylabel('reprojection RMSE [px]')
    ax_rmse.legend()

    ax_cnt.plot(times, landmark_counts, color='tab:green', marker='.', label='landmarks / keyframe')
    ax_cnt.set_ylabel('# landmarks')
    ax_cnt.set_xlabel('Time [s]')
    ax_cnt.legend()

    plt.tight_layout()
    return fig


def _plot_angular_velocities(series: list[tuple[np.ndarray, np.ndarray, str]], title: str = 'Angular Velocity in World Frame') -> plt.Figure:
    fig, (ax_wx, ax_wy, ax_wz) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(title)
    for ax, i, label in zip([ax_wx, ax_wy, ax_wz], range(3), ['wx', 'wy', 'wz']):
        for times, angular_velocities, name in series:
            ax.plot(times, angular_velocities[:, i], label=name)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'{label} [rad/s]')
        ax.legend()
    plt.tight_layout()
    return fig


def _render_positions(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.gt.times, results.gt.positions, 'gt'),
        (results.pnp.times, results.pnp.positions, 'pnp'),
        (results.gtsam.times, results.gtsam.positions, 'gtsam'),
    ]
    return figure_to_image(_plot_positions([s for s in all_series if enabled[s[2]]]))


def _render_attitudes(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.gt.times, results.gt.attitudes, 'gt'),
        (results.pnp.times, results.pnp.attitudes, 'pnp'),
        (results.gtsam.times, results.gtsam.attitudes, 'gtsam'),
    ]
    return figure_to_image(_plot_attitudes([s for s in all_series if enabled[s[2]]]))


def _render_rotation_matrices(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.gt.times, results.gt.rotation_matrices, 'gt'),
        (results.pnp.times, results.pnp.rotation_matrices, 'pnp'),
        (results.gtsam.times, results.gtsam.rotation_matrices, 'gtsam'),
    ]
    return figure_to_image(_plot_rotation_matrices([s for s in all_series if enabled[s[2]]]))


def _render_velocities(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.gtsam.times, results.gtsam.velocities, 'gtsam'),
    ]
    return figure_to_image(_plot_velocities([s for s in all_series if enabled[s[2]]]))


def _render_biases(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.gtsam.times, results.gtsam.biases, 'gtsam'),
    ]
    return figure_to_image(_plot_biases([s for s in all_series if enabled[s[2]]]))


def _render_gtsam_diagnostics(results: SlamResult) -> np.ndarray:
    g = results.gtsam
    return figure_to_image(_plot_gtsam_diagnostics(
        g.times, g.position_errors, g.reprojection_rmse, g.landmark_counts))


def _render_linear_accelerations(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.imu.times, results.extra.linear_accelerations_in_world, 'imu'),
        (results.gtsam.angular_velocity_times, results.gtsam.linear_accelerations, 'gtsam'),
    ]
    return figure_to_image(_plot_linear_accelerations([s for s in all_series if enabled[s[2]]], gravity=results.extra.gravity))


def _render_angular_velocities(results: SlamResult, enabled: dict[str, bool]) -> np.ndarray:
    all_series = [
        (results.gt.angular_velocity_times, results.gt.angular_velocities, 'gt'),
        (results.pnp.angular_velocity_times, results.pnp.angular_velocities, 'pnp'),
        (results.gtsam.angular_velocity_times, results.gtsam.angular_velocities, 'gtsam'),
    ]
    return figure_to_image(_plot_angular_velocities([s for s in all_series if enabled[s[2]]]))


def _render_imu_attitudes(results: SlamResult) -> np.ndarray:
    series = [(results.imu.times, results.imu.attitudes, 'imu')]
    return figure_to_image(_plot_attitudes(series, 'Attitude (Rotation Vector) in Body Frame'))


def _render_imu_rotation_matrices(results: SlamResult) -> np.ndarray:
    series = [(results.imu.times, results.imu.rotation_matrices, 'imu')]
    return figure_to_image(_plot_rotation_matrices(series, 'Rotation Axes in Body Frame'))


def _render_imu_angular_velocities(results: SlamResult) -> np.ndarray:
    series = [(results.imu.times, results.imu.angular_velocities, 'imu')]
    return figure_to_image(_plot_angular_velocities(series, 'Angular Velocity in Body Frame'))


def _render_imu_linear_accelerations(results: SlamResult) -> np.ndarray:
    series = [(results.imu.times, results.imu.linear_accelerations, 'imu')]
    return figure_to_image(_plot_linear_accelerations(series, 'Linear Acceleration in Body Frame'))


class SlamViewModel:
    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data
        self._solver: Optional[SlamSolver] = None
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_rotation_matrices: Optional[hello_imgui.TextureGpu] = None
        self._tex_linear_accelerations: Optional[hello_imgui.TextureGpu] = None
        self._tex_angular_velocities: Optional[hello_imgui.TextureGpu] = None
        self._tex_velocities: Optional[hello_imgui.TextureGpu] = None
        self._tex_biases: Optional[hello_imgui.TextureGpu] = None
        self._tex_diagnostics: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_rotation_matrices: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_angular_velocities: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_linear_accelerations: Optional[hello_imgui.TextureGpu] = None
        self._stale_textures: list[hello_imgui.TextureGpu] = []
        self.pos_enabled: dict[str, bool] = {'gt': True, 'pnp': True, 'gtsam': True}
        self.att_enabled: dict[str, bool] = {'gt': True, 'pnp': True, 'gtsam': True}
        self.vel_enabled: dict[str, bool] = {'gtsam': True}
        self.bias_enabled: dict[str, bool] = {'gtsam': True}
        self.lin_acc_enabled: dict[str, bool] = {'imu': True, 'gtsam': True}
        self.omega_enabled: dict[str, bool] = {'gt': True, 'pnp': True, 'gtsam': True}

    def start(
        self,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
    ) -> None:
        self._solver = SlamSolver(self._data, feature_detection_result, stereo_matching_result)
        # Stash old textures so GC doesn't run glDeleteTextures on this (non-render) thread.
        # slam_view() clears _stale_textures on the main render thread.
        for tex in [self._tex_positions, self._tex_attitudes, self._tex_rotation_matrices,
                    self._tex_linear_accelerations, self._tex_angular_velocities, self._tex_velocities,
                    self._tex_biases, self._tex_diagnostics,
                    self._tex_imu_attitudes, self._tex_imu_rotation_matrices,
                    self._tex_imu_angular_velocities, self._tex_imu_linear_accelerations]:
            if tex is not None:
                self._stale_textures.append(tex)
        self._tex_positions = None
        self._tex_attitudes = None
        self._tex_rotation_matrices = None
        self._tex_linear_accelerations = None
        self._tex_angular_velocities = None
        self._tex_velocities = None
        self._tex_biases = None
        self._tex_diagnostics = None
        self._tex_imu_attitudes = None
        self._tex_imu_rotation_matrices = None
        self._tex_imu_angular_velocities = None
        self._tex_imu_linear_accelerations = None
        threading.Thread(target=self._solver.run, daemon=True).start()

    def stop(self) -> None:
        self._solver = None


# Every figure texture on the model, in one place so slam_view can tell when the whole batch
# has finished rendering (see the idling toggle at the end of slam_view).
_FIGURE_TEX_ATTRS = [
    "_tex_positions", "_tex_attitudes", "_tex_rotation_matrices", "_tex_velocities",
    "_tex_biases", "_tex_diagnostics", "_tex_angular_velocities", "_tex_linear_accelerations",
    "_tex_imu_attitudes", "_tex_imu_rotation_matrices", "_tex_imu_angular_velocities",
    "_tex_imu_linear_accelerations",
]


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
    model._stale_textures.clear()
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
    if solver.result is None:
        return

    result = solver.result

    imgui.text(f"PnP: {result.pnp.elapsed_time:.1f}s   GTSAM: {result.gtsam.elapsed_time:.1f}s")

    imgui.begin_child("##slam_scroll", (0, 0), False)

    # Building all ~12 matplotlib figures takes ~3s; doing it in one frame freezes the window
    # right after the solver finishes. Materialize at most one new figure per frame so the view
    # appears immediately and plots pop in one by one while the UI stays responsive.
    render_budget = [1]

    def show(tex_attr: str, render_fn: Callable[[], np.ndarray]) -> None:
        tex = getattr(model, tex_attr)
        if tex is None:
            if render_budget[0] <= 0:
                imgui.text("Rendering plot...")
                return
            render_budget[0] -= 1
            tex = image_to_texture(render_fn())
            setattr(model, tex_attr, tex)
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))

    if _checkboxes(model.pos_enabled, "pos"):
        model._tex_positions = None
    show("_tex_positions", lambda: _render_positions(result, model.pos_enabled))

    if _checkboxes(model.att_enabled, "att"):
        model._tex_attitudes = None
        model._tex_rotation_matrices = None
    show("_tex_attitudes", lambda: _render_attitudes(result, model.att_enabled))
    show("_tex_rotation_matrices", lambda: _render_rotation_matrices(result, model.att_enabled))

    if _checkboxes(model.vel_enabled, "vel"):
        model._tex_velocities = None
    show("_tex_velocities", lambda: _render_velocities(result, model.vel_enabled))

    if _checkboxes(model.bias_enabled, "bias"):
        model._tex_biases = None
    show("_tex_biases", lambda: _render_biases(result, model.bias_enabled))

    show("_tex_diagnostics", lambda: _render_gtsam_diagnostics(result))

    if _checkboxes(model.omega_enabled, "omega"):
        model._tex_angular_velocities = None
    show("_tex_angular_velocities", lambda: _render_angular_velocities(result, model.omega_enabled))

    if _checkboxes(model.lin_acc_enabled, "lin_acc"):
        model._tex_linear_accelerations = None
    show("_tex_linear_accelerations", lambda: _render_linear_accelerations(result, model.lin_acc_enabled))
    g = result.extra.gravity
    imgui.text(f"Gravity: [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]")

    imgui.separator()
    imgui.text("IMU (Body Frame)")

    show("_tex_imu_attitudes", lambda: _render_imu_attitudes(result))
    show("_tex_imu_rotation_matrices", lambda: _render_imu_rotation_matrices(result))
    show("_tex_imu_angular_velocities", lambda: _render_imu_angular_velocities(result))
    show("_tex_imu_linear_accelerations", lambda: _render_imu_linear_accelerations(result))

    imgui.end_child()

    # Idling (the default) throttles redraws when there's no input, which would stall the
    # per-frame figure rendering above. Keep frames flowing while any figure is still pending,
    # then restore idling once everything is built.
    pending = any(getattr(model, attr) is None for attr in _FIGURE_TEX_ATTRS)
    hello_imgui.get_runner_params().fps_idling.enable_idling = not pending
