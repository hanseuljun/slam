import threading
from typing import Callable, Optional

import numpy as np
from imgui_bundle import hello_imgui, imgui, implot, implot3d

from slam.data import EuRoCMAVData
from slam.slam import RPE_DELTA_S, SlamResult, SlamSolver

_XYZ = ['X', 'Y', 'Z']
_AXIS_NAMES = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
_BIAS_ROW_LABELS = ['accel bias', 'gyro bias']
_BIAS_ROW_UNITS = ['m/s²', 'rad/s']
_BIAS_COMPONENT_NAMES = ['x', 'y', 'z']


def _plot_series_grid(
    title: str,
    nrows: int,
    ncols: int,
    height: float,
    series: list[tuple[np.ndarray, np.ndarray, str]],
    ylabel_fn: Callable[[int, int], str],
    value_fn: Callable[[int, int, np.ndarray], np.ndarray],
    axhline_fn: Optional[Callable[[int, int], Optional[tuple[float, str]]]] = None,
    xlabel: str = 'Time [s]',
) -> None:
    """A grid of subplots, each overlaying every entry in `series` -- drawn fresh every call
    (ImPlot is immediate mode: there's no Figure/Axes to build once and reuse, unlike the
    matplotlib version this replaced). That's exactly why this is cheap enough to call every UI
    frame directly from whatever SlamSolver.result currently is, with no caching, no GPU-texture
    upload, and no background thread: ImPlot submits draw commands into the same GPU pipeline the
    rest of the UI already uses, instead of rasterizing to a CPU bitmap first (measured well under
    1ms even for a 9-subplot grid, vs. ~30-40ms/figure for matplotlib even after persistent
    Figure/Axes reuse -- see tmp/investigate/implot_positions_prototype.py).
    """
    imgui.text(title)
    flags = implot.SubplotFlags_.link_all_x if nrows * ncols > 1 else 0
    if implot.begin_subplots(f"##{title}", nrows, ncols, size=(-1, height), flags=flags):
        for row in range(nrows):
            for col in range(ncols):
                if implot.begin_plot(f"##{title}_{row}_{col}"):
                    implot.setup_axes(
                        xlabel, ylabel_fn(row, col),
                        implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
                    for times, data, name in series:
                        if len(times) == 0:
                            continue
                        x = np.ascontiguousarray(times, dtype=np.float64)
                        y = np.ascontiguousarray(value_fn(row, col, data), dtype=np.float64)
                        implot.plot_line(name, x, y)
                    if axhline_fn is not None:
                        spec = axhline_fn(row, col)
                        if spec is not None and not np.isnan(spec[0]):
                            value, label = spec
                            implot.plot_inf_lines(
                                label, np.array([value], dtype=np.float64),
                                spec=implot.Spec(flags=implot.InfLinesFlags_.horizontal))
                    implot.end_plot()
        implot.end_subplots()


def _plot_single_lines(
    title: str,
    height: float,
    rows: list[tuple[np.ndarray, np.ndarray, str, Optional[tuple[float, str]]]],
    xlabel: str = 'Time [s]',
) -> None:
    """An Nx1 grid, one line (plus an optional reference axhline) per row, each row pulling from
    its own distinct array rather than an overlay of named series -- see _plot_series_grid's
    docstring for why no persistent state is needed here either.
    rows: (times, values, ylabel, axhline_spec) per subplot; axhline_spec is (value, label) or None.
    """
    imgui.text(title)
    if implot.begin_subplots(f"##{title}", len(rows), 1, size=(-1, height), flags=implot.SubplotFlags_.link_all_x):
        for times, values, ylabel, axhline_spec in rows:
            if implot.begin_plot(f"##{title}_{ylabel}"):
                implot.setup_axes(
                    xlabel, ylabel, implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
                if len(times) > 0:
                    implot.plot_line(
                        ylabel, np.ascontiguousarray(times, dtype=np.float64),
                        np.ascontiguousarray(values, dtype=np.float64))
                if axhline_spec is not None and not np.isnan(axhline_spec[0]):
                    value, label = axhline_spec
                    implot.plot_inf_lines(
                        label, np.array([value], dtype=np.float64),
                        spec=implot.Spec(flags=implot.InfLinesFlags_.horizontal))
                implot.end_plot()
        implot.end_subplots()


def _draw_3d_trajectory(model: "SlamViewModel", result: SlamResult) -> None:
    """The same gt/gtsam positions _draw_positions breaks into 3 separate X/Y/Z-vs-time charts,
    drawn instead as one actual 3D path -- ImPlot3D is bundled with imgui_bundle (the same
    package already providing ImPlot), so this needs no new dependency and renders in-process, the
    same as every other panel here (see tmp/3d_viz_library_recommendations.html). pnp deliberately
    left out here (unlike the 2D panels) -- its raw dead-reckoning drift makes the path hard to
    read at 3D-path scale, and gt/gtsam already show what actually matters.

    A slider below the plot (model.trajectory_scrub_t, persisted across frames so it survives a
    scroll/redraw) picks a point in time; a bigger marker on each line shows exactly where that
    trajectory was at that moment, the 3D-path equivalent of scrubbing a video.
    """
    imgui.text('3D Trajectory')
    all_series = [
        (result.gt.times, result.gt.positions, 'gt'),
        (result.gtsam.times, result.gtsam.positions, 'gtsam'),
    ]
    max_t = max((times[-1] for times, positions, _ in all_series if len(times) > 0), default=0.0)
    model.trajectory_scrub_t = min(model.trajectory_scrub_t, max_t)

    # Flags_.equal: keeps X/Y/Z at the same scale so the path's shape is geometrically correct
    # (an auto-fit-per-axis plot would otherwise stretch/squash it to fill the frame).
    if implot3d.begin_plot("##3d_trajectory", size=(-1, 500), flags=implot3d.Flags_.equal):
        implot3d.setup_axes(
            'X [m]', 'Y [m]', 'Z [m]',
            implot3d.AxisFlags_.auto_fit, implot3d.AxisFlags_.auto_fit, implot3d.AxisFlags_.auto_fit)
        for times, positions, name in all_series:
            if len(positions) == 0:
                continue
            implot3d.plot_line(
                name,
                np.ascontiguousarray(positions[:, 0], dtype=np.float64),
                np.ascontiguousarray(positions[:, 1], dtype=np.float64),
                np.ascontiguousarray(positions[:, 2], dtype=np.float64))
        for times, positions, name in all_series:
            if len(times) == 0:
                continue
            idx = min(int(np.searchsorted(times, model.trajectory_scrub_t)), len(times) - 1)
            p = positions[idx]
            implot3d.plot_scatter(
                f"{name} (at t)", np.array([p[0]]), np.array([p[1]]), np.array([p[2]]),
                spec=implot3d.Spec(marker=implot3d.Marker_.circle, marker_size=8))
        implot3d.end_plot()

    imgui.set_next_item_width(-1)
    _, model.trajectory_scrub_t = imgui.slider_float(
        "##trajectory_scrub", model.trajectory_scrub_t, 0.0, max(max_t, 1e-3), "t = %.2f s")


def _draw_positions(result: SlamResult) -> None:
    all_series = [
        (result.gt.times, result.gt.positions, 'gt'),
        (result.pnp.times, result.pnp.positions, 'pnp'),
        (result.gtsam.times, result.gtsam.positions, 'gtsam'),
    ]
    _plot_series_grid(
        'Position in World Frame', 1, 3, 260, all_series,
        ylabel_fn=lambda row, col: f'{_XYZ[col]} [m]', value_fn=lambda row, col, data: data[:, col])


def _draw_attitudes(result: SlamResult) -> None:
    all_series = [
        (result.gt.times, result.gt.attitudes, 'gt'),
        (result.pnp.times, result.pnp.attitudes, 'pnp'),
        (result.gtsam.times, result.gtsam.attitudes, 'gtsam'),
    ]
    _plot_series_grid(
        'Attitude (Rotation Vector) in World Frame', 1, 3, 260, all_series,
        ylabel_fn=lambda row, col: f'{_XYZ[col]} [rad]', value_fn=lambda row, col, data: data[:, col])


def _draw_rotation_matrices(result: SlamResult) -> None:
    all_series = [
        (result.gt.times, result.gt.rotation_matrices, 'gt'),
        (result.pnp.times, result.pnp.rotation_matrices, 'pnp'),
        (result.gtsam.times, result.gtsam.rotation_matrices, 'gtsam'),
    ]
    _plot_series_grid(
        'Rotation Axes in World Frame', 3, 3, 640, all_series,
        ylabel_fn=lambda row, col: f'{_AXIS_NAMES[row]} {_XYZ[col]}',
        value_fn=lambda row, col, data: data[:, col, row])


def _draw_velocities(result: SlamResult) -> None:
    all_series = [(result.gtsam.times, result.gtsam.velocities, 'gtsam')]
    _plot_series_grid(
        'Velocity in World Frame', 3, 1, 340, all_series,
        ylabel_fn=lambda row, col: f'{["vx", "vy", "vz"][row]} [m/s]',
        value_fn=lambda row, col, data: data[:, row])


def _draw_biases(result: SlamResult) -> None:
    all_series = [(result.gtsam.times, result.gtsam.biases, 'gtsam')]
    _plot_series_grid(
        'IMU Bias (Body Frame)', 2, 3, 240, all_series,
        ylabel_fn=lambda row, col: f'{_BIAS_ROW_LABELS[row]} {_BIAS_COMPONENT_NAMES[col]} [{_BIAS_ROW_UNITS[row]}]',
        value_fn=lambda row, col, data: data[:, row * 3 + col])


def _draw_gtsam_diagnostics(result: SlamResult) -> None:
    g = result.gtsam
    rmse = float(np.sqrt(np.mean(g.position_errors ** 2))) if len(g.position_errors) else float('nan')
    valid = ~np.isnan(g.reprojection_rmse)
    mean_reproj = float(np.mean(g.reprojection_rmse[valid])) if np.any(valid) else float('nan')
    _plot_single_lines('GTSAM Diagnostics', 340, [
        (g.times, g.position_errors, 'pos error vs GT [m]', (rmse, f'RMSE = {rmse:.3f} m')),
        (g.times, g.reprojection_rmse, 'reprojection RMSE [px]', (mean_reproj, f'mean = {mean_reproj:.2f} px')),
        (g.times, g.landmark_counts, '# landmarks', None),
    ])


def _draw_ate_rpe(result: SlamResult) -> None:
    g = result.gtsam
    ate_rmse = float(np.sqrt(np.mean(g.ate_position_errors ** 2))) if len(g.ate_position_errors) else float('nan')
    ate_rot_rmse = float(np.sqrt(np.mean(g.ate_rotation_errors ** 2))) if len(g.ate_rotation_errors) else float('nan')
    valid = ~np.isnan(g.rpe_translation_errors)
    rpe_trans_rmse = float(np.sqrt(np.mean(g.rpe_translation_errors[valid] ** 2))) if np.any(valid) else float('nan')
    rpe_rot_rmse = float(np.sqrt(np.mean(g.rpe_rotation_errors[valid] ** 2))) if np.any(valid) else float('nan')
    _plot_single_lines('ATE / RPE (batch-aligned, yaw + translation)', 440, [
        (g.times, g.ate_position_errors, 'ATE pos [m]', (ate_rmse, f'RMSE = {ate_rmse:.3f} m')),
        (g.times, g.ate_rotation_errors, 'ATE rot [deg]', (ate_rot_rmse, f'RMSE = {ate_rot_rmse:.3f} deg')),
        (g.times, g.rpe_translation_errors, f'RPE trans [m] ({RPE_DELTA_S:g}s window)',
         (rpe_trans_rmse, f'RMSE = {rpe_trans_rmse:.3f} m')),
        (g.times, g.rpe_rotation_errors, f'RPE rot [deg] ({RPE_DELTA_S:g}s window)',
         (rpe_rot_rmse, f'RMSE = {rpe_rot_rmse:.3f} deg')),
    ])


def _draw_angular_velocities(result: SlamResult) -> None:
    all_series = [
        (result.gt.angular_velocity_times, result.gt.angular_velocities, 'gt'),
        (result.pnp.angular_velocity_times, result.pnp.angular_velocities, 'pnp'),
        (result.gtsam.angular_velocity_times, result.gtsam.angular_velocities, 'gtsam'),
    ]
    _plot_series_grid(
        'Angular Velocity in World Frame', 3, 1, 340, all_series,
        ylabel_fn=lambda row, col: f'{["wx", "wy", "wz"][row]} [rad/s]',
        value_fn=lambda row, col, data: data[:, row])


def _draw_linear_accelerations(result: SlamResult) -> None:
    all_series = [
        (result.imu.times, result.extra.linear_accelerations_in_world, 'imu'),
        (result.gtsam.angular_velocity_times, result.gtsam.linear_accelerations, 'gtsam'),
    ]
    gravity = result.extra.gravity
    _plot_series_grid(
        'Linear Acceleration in World Frame', 3, 1, 340, all_series,
        ylabel_fn=lambda row, col: f'{["ax", "ay", "az"][row]} [m/s²]',
        value_fn=lambda row, col, data: data[:, row],
        axhline_fn=lambda row, col: (-gravity[row], '-gravity'))


class SlamViewModel:
    def __init__(
        self, data: EuRoCMAVData, start_s: float, duration_s: float, run_loop_closure: bool = False,
    ) -> None:
        self._data = data
        self._start_s = start_s
        self._duration_s = duration_s
        self._run_loop_closure = run_loop_closure
        self._solver: Optional[SlamSolver] = None
        # Current position of the 3D trajectory's timeline scrubber -- see _draw_3d_trajectory.
        self.trajectory_scrub_t: float = 0.0

    def start(self) -> None:
        self._solver = SlamSolver(
            self._data, self._start_s, self._duration_s, enable_loop_closure=self._run_loop_closure)
        threading.Thread(target=self._solver.run, daemon=True).start()

    def stop(self) -> None:
        if self._solver is not None:
            self._solver.cancel()
        self._solver = None


def slam_view(model: SlamViewModel) -> None:
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
    # Idling (the default) throttles redraws when there's no input, which would stop this
    # function from ever being called to notice the background solver thread's progress. Only
    # needed while a run is still in flight -- once it's done there's nothing left to poll for.
    hello_imgui.get_runner_params().fps_idling.enable_idling = not solver.loading

    result = solver.result
    if result is None:
        return

    imgui.text(f"PnP: {result.pnp.elapsed_time:.1f}s   GTSAM: {result.gtsam.elapsed_time:.1f}s")

    imgui.begin_child("##slam_scroll", (0, 0), False)

    _draw_3d_trajectory(model, result)
    _draw_positions(result)
    _draw_attitudes(result)
    _draw_rotation_matrices(result)
    _draw_velocities(result)
    _draw_biases(result)
    _draw_gtsam_diagnostics(result)
    _draw_ate_rpe(result)
    _draw_angular_velocities(result)
    _draw_linear_accelerations(result)
    g = result.extra.gravity
    imgui.text(f"Gravity: [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]")

    imgui.end_child()
