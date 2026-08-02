import threading
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui
from matplotlib.lines import Line2D

from slam.data import EuRoCMAVData
from slam.imu_initialization import ImuInitializationResult
from slam.slam import RPE_DELTA_S, SlamResult, SlamSolver
from ui.utils import image_to_texture, rasterize_figure


class _ReusableSeriesPlot:
    """A grid of subplots, each overlaying up to len(series_names) named line series -- built
    once, then updated via set_data()/relim()/autoscale_view() on every render() call instead of
    torn down and rebuilt from scratch every time. Rebuilding from scratch (the old behavior) is
    ~2x more expensive than this even after rasterize_figure's PNG-round-trip skip -- the extra
    cost is subplots()/tight_layout()/legend() construction, not the line drawing itself. A series
    absent from a given render() call (its checkbox disabled) gets its line hidden rather than
    removed, so a fixed line per (row, col, name) can be kept and just updated in place.
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        figsize: tuple[float, float],
        suptitle: str,
        series_names: list[str],
        ylabel_fn: Callable[[int, int], str],
        value_fn: Callable[[int, int, np.ndarray], np.ndarray],
        axhline_fn: Optional[Callable[[int, int, object], Optional[tuple[float, str]]]] = None,
        xlabel: str = 'Time [s]',
    ) -> None:
        self._series_names = series_names
        self._value_fn = value_fn
        self._axhline_fn = axhline_fn
        self._fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        self._fig.suptitle(suptitle)
        self._axes = axes
        self._lines: dict[tuple[int, int, str], Line2D] = {}
        self._axhlines: dict[tuple[int, int], Optional[Line2D]] = {}
        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row][col]
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel_fn(row, col))
                for name in series_names:
                    (line,) = ax.plot([], [], label=name)
                    self._lines[(row, col, name)] = line
                self._axhlines[(row, col)] = None
        # Deferred, not called here: tight_layout()'s margins depend on tick-label extents (digit
        # count, minus signs, decimals), which only reflect reality once real data has been set --
        # computed now, against empty axes' default 0-1 ticks, it'd bake in margins that don't
        # match the real plot, producing a persistent few-pixel misalignment. Still only paid
        # once, on the first render() call, not on every one -- that's the actual expensive part
        # this class exists to avoid paying repeatedly.
        self._laid_out = False

    def render(
        self, series: list[tuple[np.ndarray, np.ndarray, str]], axhline_context: object = None,
    ) -> np.ndarray:
        present = {name: (times, data) for times, data, name in series}
        nrows, ncols = self._axes.shape
        for row in range(nrows):
            for col in range(ncols):
                ax = self._axes[row][col]
                handles = []
                for name in self._series_names:
                    line = self._lines[(row, col, name)]
                    if name in present:
                        times, data = present[name]
                        line.set_data(times, self._value_fn(row, col, data))
                        line.set_visible(True)
                        handles.append(line)
                    else:
                        line.set_data([], [])
                        line.set_visible(False)
                old_axhline = self._axhlines[(row, col)]
                if old_axhline is not None:
                    old_axhline.remove()
                    self._axhlines[(row, col)] = None
                if self._axhline_fn is not None:
                    spec = self._axhline_fn(row, col, axhline_context)
                    if spec is not None:
                        value, label = spec
                        axhline = ax.axhline(value, color='gray', linestyle='--', linewidth=1, label=label)
                        self._axhlines[(row, col)] = axhline
                        handles.append(axhline)
                ax.relim()
                ax.autoscale_view()
                ax.legend(handles=handles)
        if not self._laid_out:
            self._fig.tight_layout()
            self._laid_out = True
        return rasterize_figure(self._fig)

    def close(self) -> None:
        plt.close(self._fig)


class _ReusableGtsamDiagnosticsPlot:
    """Persistent version of the GTSAM diagnostics figure. Each subplot here has exactly one line
    pulling from a different array (not an overlay of named series), so it doesn't fit
    _ReusableSeriesPlot's shape -- see that class's docstring for why reuse matters at all.
    """

    def __init__(self) -> None:
        self._fig, (self._ax_err, self._ax_rmse, self._ax_cnt) = plt.subplots(
            3, 1, figsize=(12, 9), sharex=True)
        self._fig.suptitle('GTSAM Diagnostics')

        (self._line_err,) = self._ax_err.plot([], [], color='tab:red', marker='.', label='position error')
        self._axhline_err: Optional[Line2D] = None
        self._ax_err.set_ylabel('pos error vs GT [m]')

        (self._line_rmse,) = self._ax_rmse.plot([], [], color='tab:blue', marker='.', label='reprojection RMSE')
        self._axhline_rmse: Optional[Line2D] = None
        self._ax_rmse.set_ylabel('reprojection RMSE [px]')

        (self._line_cnt,) = self._ax_cnt.plot([], [], color='tab:green', marker='.', label='landmarks / keyframe')
        self._ax_cnt.set_ylabel('# landmarks')
        self._ax_cnt.set_xlabel('Time [s]')

        # Deferred to first render() -- see _ReusableSeriesPlot's matching comment.
        self._laid_out = False

    def render(
        self, times: np.ndarray, position_errors: np.ndarray,
        reprojection_rmse: np.ndarray, landmark_counts: np.ndarray,
    ) -> np.ndarray:
        rmse = float(np.sqrt(np.mean(position_errors ** 2))) if len(position_errors) else float('nan')
        self._line_err.set_data(times, position_errors)
        if self._axhline_err is not None:
            self._axhline_err.remove()
        self._axhline_err = self._ax_err.axhline(
            rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {rmse:.3f} m')
        self._ax_err.relim()
        self._ax_err.autoscale_view()
        self._ax_err.legend()

        valid = ~np.isnan(reprojection_rmse)
        mean_reproj = float(np.mean(reprojection_rmse[valid])) if np.any(valid) else float('nan')
        self._line_rmse.set_data(times, reprojection_rmse)
        if self._axhline_rmse is not None:
            self._axhline_rmse.remove()
        self._axhline_rmse = self._ax_rmse.axhline(
            mean_reproj, color='gray', linestyle='--', linewidth=1, label=f'mean = {mean_reproj:.2f} px')
        self._ax_rmse.relim()
        self._ax_rmse.autoscale_view()
        self._ax_rmse.legend()

        self._line_cnt.set_data(times, landmark_counts)
        self._ax_cnt.relim()
        self._ax_cnt.autoscale_view()
        self._ax_cnt.legend()

        if not self._laid_out:
            self._fig.tight_layout()
            self._laid_out = True
        return rasterize_figure(self._fig)

    def close(self) -> None:
        plt.close(self._fig)


class _ReusableAteRpePlot:
    """Persistent version of the ATE/RPE figure -- same one-line-per-subplot shape as
    _ReusableGtsamDiagnosticsPlot, not an overlay of named series."""

    def __init__(self) -> None:
        self._fig, (self._ax_ate_pos, self._ax_ate_rot, self._ax_rpe_trans, self._ax_rpe_rot) = \
            plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        self._fig.suptitle('ATE / RPE (batch-aligned, yaw + translation)')

        (self._line_ate_pos,) = self._ax_ate_pos.plot([], [], color='tab:red', marker='.', label='ATE position error')
        self._axhline_ate_pos: Optional[Line2D] = None
        self._ax_ate_pos.set_ylabel('ATE pos [m]')

        (self._line_ate_rot,) = self._ax_ate_rot.plot([], [], color='tab:orange', marker='.', label='ATE rotation error')
        self._axhline_ate_rot: Optional[Line2D] = None
        self._ax_ate_rot.set_ylabel('ATE rot [deg]')

        (self._line_rpe_trans,) = self._ax_rpe_trans.plot([], [], color='tab:blue', marker='.')
        self._axhline_rpe_trans: Optional[Line2D] = None
        self._ax_rpe_trans.set_ylabel('RPE trans [m]')

        (self._line_rpe_rot,) = self._ax_rpe_rot.plot([], [], color='tab:green', marker='.')
        self._axhline_rpe_rot: Optional[Line2D] = None
        self._ax_rpe_rot.set_ylabel('RPE rot [deg]')
        self._ax_rpe_rot.set_xlabel('Time [s]')

        # Deferred to first render() -- see _ReusableSeriesPlot's matching comment.
        self._laid_out = False

    def render(
        self, times: np.ndarray, ate_position_errors: np.ndarray, ate_rotation_errors: np.ndarray,
        rpe_translation_errors: np.ndarray, rpe_rotation_errors: np.ndarray, rpe_delta_s: float,
    ) -> np.ndarray:
        ate_rmse = float(np.sqrt(np.mean(ate_position_errors ** 2))) if len(ate_position_errors) else float('nan')
        self._line_ate_pos.set_data(times, ate_position_errors)
        if self._axhline_ate_pos is not None:
            self._axhline_ate_pos.remove()
        self._axhline_ate_pos = self._ax_ate_pos.axhline(
            ate_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {ate_rmse:.3f} m')
        self._ax_ate_pos.relim()
        self._ax_ate_pos.autoscale_view()
        self._ax_ate_pos.legend()

        ate_rot_rmse = float(np.sqrt(np.mean(ate_rotation_errors ** 2))) if len(ate_rotation_errors) else float('nan')
        self._line_ate_rot.set_data(times, ate_rotation_errors)
        if self._axhline_ate_rot is not None:
            self._axhline_ate_rot.remove()
        self._axhline_ate_rot = self._ax_ate_rot.axhline(
            ate_rot_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {ate_rot_rmse:.3f} deg')
        self._ax_ate_rot.relim()
        self._ax_ate_rot.autoscale_view()
        self._ax_ate_rot.legend()

        valid = ~np.isnan(rpe_translation_errors)
        rpe_trans_rmse = float(np.sqrt(np.mean(rpe_translation_errors[valid] ** 2))) if np.any(valid) else float('nan')
        self._line_rpe_trans.set_data(times, rpe_translation_errors)
        self._line_rpe_trans.set_label(f'RPE translation ({rpe_delta_s:g}s window)')
        if self._axhline_rpe_trans is not None:
            self._axhline_rpe_trans.remove()
        self._axhline_rpe_trans = self._ax_rpe_trans.axhline(
            rpe_trans_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {rpe_trans_rmse:.3f} m')
        self._ax_rpe_trans.relim()
        self._ax_rpe_trans.autoscale_view()
        self._ax_rpe_trans.legend()

        rpe_rot_rmse = float(np.sqrt(np.mean(rpe_rotation_errors[valid] ** 2))) if np.any(valid) else float('nan')
        self._line_rpe_rot.set_data(times, rpe_rotation_errors)
        self._line_rpe_rot.set_label(f'RPE rotation ({rpe_delta_s:g}s window)')
        if self._axhline_rpe_rot is not None:
            self._axhline_rpe_rot.remove()
        self._axhline_rpe_rot = self._ax_rpe_rot.axhline(
            rpe_rot_rmse, color='gray', linestyle='--', linewidth=1, label=f'RMSE = {rpe_rot_rmse:.3f} deg')
        self._ax_rpe_rot.relim()
        self._ax_rpe_rot.autoscale_view()
        self._ax_rpe_rot.legend()

        if not self._laid_out:
            self._fig.tight_layout()
            self._laid_out = True
        return rasterize_figure(self._fig)

    def close(self) -> None:
        plt.close(self._fig)


def _render_positions(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.gt.times, results.gt.positions, 'gt'),
        (results.pnp.times, results.pnp.positions, 'pnp'),
        (results.gtsam.times, results.gtsam.positions, 'gtsam'),
    ]
    return model._plot_positions.render([s for s in all_series if model.pos_enabled[s[2]]])


def _render_attitudes(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.gt.times, results.gt.attitudes, 'gt'),
        (results.pnp.times, results.pnp.attitudes, 'pnp'),
        (results.gtsam.times, results.gtsam.attitudes, 'gtsam'),
    ]
    return model._plot_attitudes.render([s for s in all_series if model.att_enabled[s[2]]])


def _render_rotation_matrices(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.gt.times, results.gt.rotation_matrices, 'gt'),
        (results.pnp.times, results.pnp.rotation_matrices, 'pnp'),
        (results.gtsam.times, results.gtsam.rotation_matrices, 'gtsam'),
    ]
    return model._plot_rotation_matrices.render([s for s in all_series if model.att_enabled[s[2]]])


def _render_velocities(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.gtsam.times, results.gtsam.velocities, 'gtsam'),
    ]
    return model._plot_velocities.render([s for s in all_series if model.vel_enabled[s[2]]])


def _render_biases(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.gtsam.times, results.gtsam.biases, 'gtsam'),
    ]
    return model._plot_biases.render([s for s in all_series if model.bias_enabled[s[2]]])


def _render_gtsam_diagnostics(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    g = results.gtsam
    return model._plot_diagnostics.render(g.times, g.position_errors, g.reprojection_rmse, g.landmark_counts)


def _render_ate_rpe(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    g = results.gtsam
    return model._plot_ate_rpe.render(
        g.times, g.ate_position_errors, g.ate_rotation_errors,
        g.rpe_translation_errors, g.rpe_rotation_errors, RPE_DELTA_S)


def _render_linear_accelerations(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.imu.times, results.extra.linear_accelerations_in_world, 'imu'),
        (results.gtsam.angular_velocity_times, results.gtsam.linear_accelerations, 'gtsam'),
    ]
    return model._plot_linear_accelerations.render(
        [s for s in all_series if model.lin_acc_enabled[s[2]]], axhline_context=results.extra.gravity)


def _render_angular_velocities(results: SlamResult, model: "SlamViewModel") -> np.ndarray:
    all_series = [
        (results.gt.angular_velocity_times, results.gt.angular_velocities, 'gt'),
        (results.pnp.angular_velocity_times, results.pnp.angular_velocities, 'pnp'),
        (results.gtsam.angular_velocity_times, results.gtsam.angular_velocities, 'gtsam'),
    ]
    return model._plot_angular_velocities.render([s for s in all_series if model.omega_enabled[s[2]]])


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
        self._stale_textures: list[hello_imgui.TextureGpu] = []

        # One persistent Figure/Axes per plot (13, matching the _tex_* attrs above), built once
        # here and updated in place by _render_* on every call -- see _ReusableSeriesPlot's
        # docstring. Never rebuilt for the lifetime of this model; closed in stop().
        _xyz = ['X', 'Y', 'Z']
        _axis_names = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
        _bias_row_labels = ['accel bias', 'gyro bias']
        _bias_row_units = ['m/s²', 'rad/s']
        _bias_component_names = ['x', 'y', 'z']
        self._plot_positions = _ReusableSeriesPlot(
            1, 3, (12, 4), 'Position in World Frame', ['gt', 'pnp', 'gtsam'],
            ylabel_fn=lambda row, col: f'{_xyz[col]} [m]', value_fn=lambda row, col, data: data[:, col])
        self._plot_attitudes = _ReusableSeriesPlot(
            1, 3, (12, 4), 'Attitude (Rotation Vector) in World Frame', ['gt', 'pnp', 'gtsam'],
            ylabel_fn=lambda row, col: f'{_xyz[col]} [rad]', value_fn=lambda row, col, data: data[:, col])
        self._plot_rotation_matrices = _ReusableSeriesPlot(
            3, 3, (12, 9), 'Rotation Axes in World Frame', ['gt', 'pnp', 'gtsam'],
            ylabel_fn=lambda row, col: f'{_axis_names[row]} {_xyz[col]}',
            value_fn=lambda row, col, data: data[:, col, row])
        self._plot_velocities = _ReusableSeriesPlot(
            3, 1, (12, 9), 'Velocity in World Frame', ['gtsam'],
            ylabel_fn=lambda row, col: f'{["vx", "vy", "vz"][row]} [m/s]',
            value_fn=lambda row, col, data: data[:, row])
        self._plot_biases = _ReusableSeriesPlot(
            2, 3, (12, 6), 'IMU Bias (Body Frame)', ['gtsam'],
            ylabel_fn=lambda row, col: f'{_bias_row_labels[row]} {_bias_component_names[col]} [{_bias_row_units[row]}]',
            value_fn=lambda row, col, data: data[:, row * 3 + col])
        self._plot_diagnostics = _ReusableGtsamDiagnosticsPlot()
        self._plot_ate_rpe = _ReusableAteRpePlot()
        self._plot_angular_velocities = _ReusableSeriesPlot(
            3, 1, (12, 9), 'Angular Velocity in World Frame', ['gt', 'pnp', 'gtsam'],
            ylabel_fn=lambda row, col: f'{["wx", "wy", "wz"][row]} [rad/s]',
            value_fn=lambda row, col, data: data[:, row])
        self._plot_linear_accelerations = _ReusableSeriesPlot(
            3, 1, (12, 9), 'Linear Acceleration in World Frame', ['imu', 'gtsam'],
            ylabel_fn=lambda row, col: f'{["ax", "ay", "az"][row]} [m/s²]',
            value_fn=lambda row, col, data: data[:, row],
            axhline_fn=lambda row, col, gravity: (-gravity[row], '-gravity') if gravity is not None else None)
        # The result currently on screen (all 13 _tex_* textures reflect exactly this snapshot),
        # vs. the next one being assembled on a background thread -- see _render_pending_batch and
        # slam_view for why these two are kept apart instead of invalidating _tex_* the moment a
        # new keyframe result arrives. _pending_images is None while no batch is in flight, and
        # only ever assigned once, as a single already-complete dict, by the background thread's
        # last line -- so the main thread never has to worry about observing a partially-filled
        # batch (see _render_pending_batch's docstring).
        self._displayed_result: Optional[SlamResult] = None
        self._pending_result: Optional[SlamResult] = None
        self._pending_images: Optional[dict[str, np.ndarray]] = None
        self._building: bool = False
        # Set by a checkbox toggle (a series enabled/disabled) to request a redraw of the
        # currently displayed data with the new checkbox state -- see slam_view's scheduling logic
        # for why this feeds into the same background-batch mechanism as a new keyframe result,
        # rather than triggering its own separate rebuild.
        self._checkbox_dirty: bool = False
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
                    self._tex_biases, self._tex_diagnostics, self._tex_ate_rpe]:
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
        self._displayed_result = None
        self._pending_result = None
        self._pending_images = None
        self._building = False
        self._checkbox_dirty = False
        threading.Thread(target=self._solver.run, daemon=True).start()

    def stop(self) -> None:
        if self._solver is not None:
            self._solver.cancel()
        self._solver = None

    def close_plots(self) -> None:
        """Closes all 13 persistent Figures -- without this, restarting on a new dataset (see
        RootViewModel.restart in main.py, which calls Pipeline.stop -> this) would leak the old
        model's Figures forever: pyplot keeps every open Figure registered globally until
        plt.close() is called on it, and a fresh SlamViewModel with its own 13 Figures gets built
        right after this one is torn down.
        """
        for attr in _PLOT_ATTRS:
            getattr(self, attr).close()


# Every persistent plot object on the model, in one place so close_plots (above) can tear them
# all down without listing each by name.
_PLOT_ATTRS = [
    "_plot_positions", "_plot_attitudes", "_plot_rotation_matrices", "_plot_velocities",
    "_plot_biases", "_plot_diagnostics", "_plot_ate_rpe", "_plot_angular_velocities",
    "_plot_linear_accelerations",
]

# Every figure texture on the model, paired with how to render it from a (result, model)
# snapshot -- one place so slam_view's background batch-builder (_render_pending_batch) can
# render all of them the same way it draws them, and so it can tell when the whole batch is done
# (see the idling toggle at the end of slam_view).
_FIGURE_SPECS: list[tuple[str, Callable[[SlamResult, "SlamViewModel"], np.ndarray]]] = [
    ("_tex_positions", _render_positions),
    ("_tex_attitudes", _render_attitudes),
    ("_tex_rotation_matrices", _render_rotation_matrices),
    ("_tex_velocities", _render_velocities),
    ("_tex_biases", _render_biases),
    ("_tex_diagnostics", _render_gtsam_diagnostics),
    ("_tex_ate_rpe", _render_ate_rpe),
    ("_tex_angular_velocities", _render_angular_velocities),
    ("_tex_linear_accelerations", _render_linear_accelerations),
]
_FIGURE_TEX_ATTRS = [attr for attr, _ in _FIGURE_SPECS]


def _render_pending_batch(model: "SlamViewModel", pending_result: SlamResult) -> None:
    """The target of a dedicated background thread (see slam_view) -- renders every figure in
    _FIGURE_SPECS from pending_result into a local dict, entirely off the main/UI thread, so a
    batch's ~500ms of matplotlib work (even with _ReusableSeriesPlot's reuse) never stalls a UI
    frame. Only touches model._pending_images once, as its very last action, assigning the whole
    already-finished dict in one attribute write -- so the main thread polling model._pending_images
    can never observe a partially-built batch (no lock needed: CPython's GIL makes that single
    attribute write atomic with respect to another thread's read).

    Safe to run concurrently with the main thread's own imgui/GPU work because it never touches
    GPU state (figure_to_image's GPU texture upload happens back on the main thread once this
    finishes) -- but it does mutate the model's 13 persistent plot objects (Line2D data, axes
    limits), so slam_view must never let two of these run at once, or run one of these alongside a
    direct call into the same persistent plot objects. model._building enforces the former; there
    is no other code path left that calls into them (see slam_view's checkbox handling, which now
    only sets model._checkbox_dirty instead of rendering directly).
    """
    images = {attr: render_fn(pending_result, model) for attr, render_fn in _FIGURE_SPECS}
    model._pending_images = images


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
    # Start assembling the next batch, on a background thread, once the previous one is fully
    # swapped in -- see _render_pending_batch's docstring. Triggered by either a new keyframe
    # result or a checkbox toggle (model._checkbox_dirty); either way it renders from whatever's
    # freshest right now (solver.result if a run is in progress, else the already-displayed
    # result, e.g. a checkbox toggled after the run finished). While model._building is True,
    # further triggers are simply left pending (not queued) -- once the in-flight batch lands, the
    # very next check here starts a new one from whatever's newest/dirty by then, so a burst of
    # keyframes coalesces into one batch instead of piling up a backlog of stale ones.
    if not model._building and (
            (solver.result is not None and solver.result is not model._displayed_result)
            or model._checkbox_dirty):
        pending_result = solver.result if solver.result is not None else model._displayed_result
        assert pending_result is not None  # checkboxes only render once model._displayed_result exists
        model._pending_result = pending_result
        model._pending_images = None
        model._checkbox_dirty = False
        model._building = True
        threading.Thread(target=_render_pending_batch, args=(model, pending_result), daemon=True).start()

    # The background thread's batch landed: swap all 13 textures in at once (cheap GPU uploads,
    # a few ms total -- the expensive matplotlib work already happened off-thread) and make it the
    # new displayed result.
    if model._building and model._pending_images is not None:
        for attr, _ in _FIGURE_SPECS:
            setattr(model, attr, image_to_texture(model._pending_images[attr]))
        model._displayed_result = model._pending_result
        model._pending_result = None
        model._pending_images = None
        model._building = False

    result = model._displayed_result
    if result is None:
        return

    imgui.text(f"PnP: {result.pnp.elapsed_time:.1f}s   GTSAM: {result.gtsam.elapsed_time:.1f}s")

    imgui.begin_child("##slam_scroll", (0, 0), False)

    def show(tex_attr: str) -> None:
        # No on-demand rendering here anymore -- every texture is produced exclusively by the
        # background batch thread above (see _render_pending_batch and model._checkbox_dirty).
        # Only reachable as None before the very first batch for this model has ever landed.
        tex = getattr(model, tex_attr)
        if tex is None:
            imgui.text("Rendering plot...")
            return
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))

    if _checkboxes(model.pos_enabled, "pos"):
        model._checkbox_dirty = True
    show("_tex_positions")

    if _checkboxes(model.att_enabled, "att"):
        model._checkbox_dirty = True
    show("_tex_attitudes")
    show("_tex_rotation_matrices")

    if _checkboxes(model.vel_enabled, "vel"):
        model._checkbox_dirty = True
    show("_tex_velocities")

    if _checkboxes(model.bias_enabled, "bias"):
        model._checkbox_dirty = True
    show("_tex_biases")

    show("_tex_diagnostics")
    show("_tex_ate_rpe")

    if _checkboxes(model.omega_enabled, "omega"):
        model._checkbox_dirty = True
    show("_tex_angular_velocities")

    if _checkboxes(model.lin_acc_enabled, "lin_acc"):
        model._checkbox_dirty = True
    show("_tex_linear_accelerations")
    g = result.extra.gravity
    imgui.text(f"Gravity: [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]")

    imgui.end_child()

    # Idling (the default) throttles redraws when there's no input, which would stop this
    # function from ever being called to notice the background batch thread has finished. Keep
    # frames flowing while a batch is in flight (model._building) or the very first one hasn't
    # landed yet, then restore idling once everything is settled.
    pending = model._building or any(getattr(model, attr) is None for attr in _FIGURE_TEX_ATTRS)
    hello_imgui.get_runner_params().fps_idling.enable_idling = not pending
