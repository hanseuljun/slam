import threading
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.data import EuRoCMAVData
from slam.imu_initialization import ImuInitializationResult
from slam.slam import RPE_DELTA_S, SlamResult, SlamSolver
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


def _plot_ate_rpe(times: np.ndarray, ate_position_errors: np.ndarray, ate_rotation_errors: np.ndarray,
                   rpe_translation_errors: np.ndarray, rpe_rotation_errors: np.ndarray, rpe_delta_s: float) -> plt.Figure:
    fig, (ax_ate_pos, ax_ate_rot, ax_rpe_trans, ax_rpe_rot) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle('ATE / RPE (batch-aligned, yaw + translation)')

    ate_rmse = float(np.sqrt(np.mean(ate_position_errors ** 2))) if len(ate_position_errors) else float('nan')
    ax_ate_pos.plot(times, ate_position_errors, color='tab:red', marker='.', label='ATE position error')
    ax_ate_pos.axhline(ate_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {ate_rmse:.3f} m')
    ax_ate_pos.set_ylabel('ATE pos [m]')
    ax_ate_pos.legend()

    ate_rot_rmse = float(np.sqrt(np.mean(ate_rotation_errors ** 2))) if len(ate_rotation_errors) else float('nan')
    ax_ate_rot.plot(times, ate_rotation_errors, color='tab:orange', marker='.', label='ATE rotation error')
    ax_ate_rot.axhline(ate_rot_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {ate_rot_rmse:.3f} deg')
    ax_ate_rot.set_ylabel('ATE rot [deg]')
    ax_ate_rot.legend()

    valid = ~np.isnan(rpe_translation_errors)
    rpe_trans_rmse = float(np.sqrt(np.mean(rpe_translation_errors[valid] ** 2))) if np.any(valid) else float('nan')
    ax_rpe_trans.plot(times, rpe_translation_errors, color='tab:blue', marker='.', label=f'RPE translation ({rpe_delta_s:g}s window)')
    ax_rpe_trans.axhline(rpe_trans_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {rpe_trans_rmse:.3f} m')
    ax_rpe_trans.set_ylabel('RPE trans [m]')
    ax_rpe_trans.legend()

    rpe_rot_rmse = float(np.sqrt(np.mean(rpe_rotation_errors[valid] ** 2))) if np.any(valid) else float('nan')
    ax_rpe_rot.plot(times, rpe_rotation_errors, color='tab:green', marker='.', label=f'RPE rotation ({rpe_delta_s:g}s window)')
    ax_rpe_rot.axhline(rpe_rot_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {rpe_rot_rmse:.3f} deg')
    ax_rpe_rot.set_ylabel('RPE rot [deg]')
    ax_rpe_rot.set_xlabel('Time [s]')
    ax_rpe_rot.legend()

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


def _render_ate_rpe(results: SlamResult) -> np.ndarray:
    g = results.gtsam
    return figure_to_image(_plot_ate_rpe(
        g.times, g.ate_position_errors, g.ate_rotation_errors,
        g.rpe_translation_errors, g.rpe_rotation_errors, RPE_DELTA_S))


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
    def __init__(
        self, data: EuRoCMAVData, start_s: float, duration_s: float, run_loop_closure: bool = False,
    ) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self._run_loop_closure = run_loop_closure
        self._solver: Optional[SlamSolver] = None
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_rotation_matrices: Optional[hello_imgui.TextureGpu] = None
        self._tex_linear_accelerations: Optional[hello_imgui.TextureGpu] = None
        self._tex_angular_velocities: Optional[hello_imgui.TextureGpu] = None
        self._tex_velocities: Optional[hello_imgui.TextureGpu] = None
        self._tex_biases: Optional[hello_imgui.TextureGpu] = None
        self._tex_diagnostics: Optional[hello_imgui.TextureGpu] = None
        self._tex_ate_rpe: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_rotation_matrices: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_angular_velocities: Optional[hello_imgui.TextureGpu] = None
        self._tex_imu_linear_accelerations: Optional[hello_imgui.TextureGpu] = None
        self._stale_textures: list[hello_imgui.TextureGpu] = []
        # The result currently on screen (all 13 _tex_* textures reflect exactly this snapshot),
        # vs. the next one being assembled in the background -- see _advance_pending_batch in
        # slam_view for why these two are kept apart instead of invalidating _tex_* the moment a
        # new keyframe result arrives: rendering the 13 figures takes several frames at the
        # existing 1-figure/frame budget, and a new SlamResult can arrive every keyframe, faster
        # than that -- clearing _tex_* immediately made most plots flicker to "Rendering plot..."
        # and back on every keyframe. Building the new set into _pending_images (plain arrays, no
        # GPU texture yet) while _tex_* keeps showing the old-but-complete batch, then swapping
        # all 13 textures in at once, means the display only ever shows a fully-rendered batch.
        self._displayed_result: Optional[SlamResult] = None
        self._pending_result: Optional[SlamResult] = None
        self._pending_images: dict[str, np.ndarray] = {}
        self._building: bool = False
        self.pos_enabled: dict[str, bool] = {'gt': True, 'pnp': True, 'gtsam': True}
        self.att_enabled: dict[str, bool] = {'gt': True, 'pnp': True, 'gtsam': True}
        self.vel_enabled: dict[str, bool] = {'gtsam': True}
        self.bias_enabled: dict[str, bool] = {'gtsam': True}
        self.lin_acc_enabled: dict[str, bool] = {'imu': True, 'gtsam': True}
        self.omega_enabled: dict[str, bool] = {'gt': True, 'pnp': True, 'gtsam': True}

    def start(self) -> None:
        self._solver = SlamSolver(
            self._data, self._start_s, self._duration_s, enable_loop_closure=self._run_loop_closure)
        # Stash old textures so GC doesn't run glDeleteTextures on this (non-render) thread.
        # slam_view() clears _stale_textures on the main render thread.
        for tex in [self._tex_positions, self._tex_attitudes, self._tex_rotation_matrices,
                    self._tex_linear_accelerations, self._tex_angular_velocities, self._tex_velocities,
                    self._tex_biases, self._tex_diagnostics, self._tex_ate_rpe,
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
        self._tex_ate_rpe = None
        self._tex_imu_attitudes = None
        self._tex_imu_rotation_matrices = None
        self._tex_imu_angular_velocities = None
        self._tex_imu_linear_accelerations = None
        self._displayed_result = None
        self._pending_result = None
        self._pending_images = {}
        self._building = False
        threading.Thread(target=self._solver.run, daemon=True).start()

    def stop(self) -> None:
        if self._solver is not None:
            self._solver.cancel()
        self._solver = None


# Every figure texture on the model, paired with how to render it from a (result, model)
# snapshot -- one place so slam_view's background batch-builder (_advance_pending_batch) can
# render all 13 the same way it draws them, and so it can tell when the whole batch is done (see
# the idling toggle at the end of slam_view).
_FIGURE_SPECS: list[tuple[str, Callable[[SlamResult, "SlamViewModel"], np.ndarray]]] = [
    ("_tex_positions", lambda r, m: _render_positions(r, m.pos_enabled)),
    ("_tex_attitudes", lambda r, m: _render_attitudes(r, m.att_enabled)),
    ("_tex_rotation_matrices", lambda r, m: _render_rotation_matrices(r, m.att_enabled)),
    ("_tex_velocities", lambda r, m: _render_velocities(r, m.vel_enabled)),
    ("_tex_biases", lambda r, m: _render_biases(r, m.bias_enabled)),
    ("_tex_diagnostics", lambda r, m: _render_gtsam_diagnostics(r)),
    ("_tex_ate_rpe", lambda r, m: _render_ate_rpe(r)),
    ("_tex_angular_velocities", lambda r, m: _render_angular_velocities(r, m.omega_enabled)),
    ("_tex_linear_accelerations", lambda r, m: _render_linear_accelerations(r, m.lin_acc_enabled)),
    ("_tex_imu_attitudes", lambda r, m: _render_imu_attitudes(r)),
    ("_tex_imu_rotation_matrices", lambda r, m: _render_imu_rotation_matrices(r)),
    ("_tex_imu_angular_velocities", lambda r, m: _render_imu_angular_velocities(r)),
    ("_tex_imu_linear_accelerations", lambda r, m: _render_imu_linear_accelerations(r)),
]
_FIGURE_TEX_ATTRS = [attr for attr, _ in _FIGURE_SPECS]


def _advance_pending_batch(model: "SlamViewModel", render_budget: list[int]) -> None:
    """Renders up to render_budget[0] not-yet-built figures of the pending batch (plain arrays,
    no GPU texture yet -- model._tex_* is untouched so the old batch keeps displaying). Once
    every figure in the batch is done, swaps all 13 textures in at once and makes this the new
    displayed result -- see _pending_images's docstring on SlamViewModel for why this needs to be
    all-or-nothing rather than swapping each texture in as it finishes.
    """
    for attr, render_fn in _FIGURE_SPECS:
        if render_budget[0] <= 0:
            return
        if attr in model._pending_images:
            continue
        render_budget[0] -= 1
        assert model._pending_result is not None
        model._pending_images[attr] = render_fn(model._pending_result, model)

    if len(model._pending_images) < len(_FIGURE_SPECS):
        return
    for attr, _ in _FIGURE_SPECS:
        setattr(model, attr, image_to_texture(model._pending_images[attr]))
    model._displayed_result = model._pending_result
    model._pending_result = None
    model._pending_images = {}
    model._building = False


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
    # A result can arrive (and keep changing) while still loading -- see on_partial_result in
    # SlamSolver.run -- so loading/error are shown as a status line above whatever's already
    # there, not as an early return that hides it.
    if solver.loading:
        imgui.text(solver.progress_label)
        imgui.progress_bar(solver.progress, (-1, 0))
    elif solver.error:
        imgui.text(f"Error: {solver.error}")
    # Start assembling the next batch once the previous one is fully swapped in -- see
    # _advance_pending_batch and _pending_images's docstring on SlamViewModel. While
    # model._building is True, newer solver results are simply skipped (not queued) -- once the
    # in-flight batch lands, the very next check here picks up whatever's newest by then, so a
    # burst of keyframes just coalesces into one batch instead of piling up a backlog.
    if (not model._building and solver.result is not None
            and solver.result is not model._displayed_result):
        model._pending_result = solver.result
        model._pending_images = {}
        model._building = True

    # Building all ~12 matplotlib figures takes ~3s total; doing it in one frame would freeze the
    # window. Render at most one new figure per frame -- shared between advancing the pending
    # batch below and any single-figure rebuild a checkbox toggle triggers further down.
    render_budget = [1]
    if model._building:
        _advance_pending_batch(model, render_budget)

    result = model._displayed_result
    if result is None:
        return

    imgui.text(f"PnP: {result.pnp.elapsed_time:.1f}s   GTSAM: {result.gtsam.elapsed_time:.1f}s")

    imgui.begin_child("##slam_scroll", (0, 0), False)

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
    show("_tex_ate_rpe", lambda: _render_ate_rpe(result))

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
    # per-frame figure rendering above. Keep frames flowing while a pending batch is still being
    # assembled (model._building) or any displayed figure is mid-rebuild from a checkbox toggle,
    # then restore idling once everything is settled.
    pending = model._building or any(getattr(model, attr) is None for attr in _FIGURE_TEX_ATTRS)
    hello_imgui.get_runner_params().fps_idling.enable_idling = not pending
