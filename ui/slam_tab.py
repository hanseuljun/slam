import io

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nicegui import ui

from slam.data import DataFolder
from slam.plot import plot_positions, plot_attitudes_and_angular_velocities
from slam.slam_solver import SlamResults, SlamSolver
from ui._utils import array_to_data_uri


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return img


def _render_plots(results: SlamResults) -> tuple[np.ndarray, np.ndarray]:
    fig_pos = plot_positions(series=[
        (results.pnp_times, results.pnp_positions, 'pnp'),
        (results.gt_times, results.gt_positions, 'gt'),
    ])
    fig_att = plot_attitudes_and_angular_velocities(
        attitude_series=[
            (results.pnp_times, results.pnp_attitudes, 'pnp'),
            (results.imu_attitude_times, results.imu_attitudes, 'imu'),
            (results.gt_times, results.gt_attitudes, 'gt'),
            (results.pnp_times, results.optimized_attitudes, 'opt'),
        ],
        angular_velocity_series=[
            (results.pnp_angular_velocity_times, results.pnp_angular_velocities, 'pnp'),
            (results.imu_times, results.imu_angular_velocities, 'imu'),
            (results.pnp_times, results.imu_angular_velocities_at_cam_times, 'imu@cam'),
            (results.gt_angular_velocity_times, results.gt_angular_velocities, 'gt'),
            (results.pnp_angular_velocity_times, results.optimized_angular_velocities, 'opt'),
        ],
    )
    return _fig_to_image(fig_pos), _fig_to_image(fig_att)


class SlamTabState:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self.duration_s: float = 20.0
        self._solver = SlamSolver(data, self.duration_s)
        self._solver.start()

    def restart(self) -> None:
        self._solver._stop_event.set()
        self._solver = SlamSolver(self._data, self.duration_s)
        self._solver.start()


def slam_tab(state: SlamTabState) -> None:
    with ui.column().classes('w-full'):
        progress_label = ui.label('')
        progress_bar = ui.linear_progress(value=0).classes('w-full')
        error_label = ui.label('').classes('text-red-500').set_visibility(False)
        img_positions = ui.image('').classes('w-full').set_visibility(False)
        img_attitudes = ui.image('').classes('w-full').set_visibility(False)

        def show_progress() -> None:
            progress_label.set_visibility(True)
            progress_bar.set_visibility(True)
            error_label.set_visibility(False)
            img_positions.set_visibility(False)
            img_attitudes.set_visibility(False)

        def poll() -> None:
            solver = state._solver
            if solver.loading:
                progress_label.text = solver.progress_label
                progress_bar.value = solver.progress
            elif solver.error:
                progress_label.set_visibility(False)
                progress_bar.set_visibility(False)
                error_label.text = f'Error: {solver.error}'
                error_label.set_visibility(True)
                timer.deactivate()
            elif solver.plots is not None:
                progress_label.set_visibility(False)
                progress_bar.set_visibility(False)
                pos_img, att_img = _render_plots(solver.plots)
                img_positions.source = array_to_data_uri(pos_img)
                img_attitudes.source = array_to_data_uri(att_img)
                img_positions.set_visibility(True)
                img_attitudes.set_visibility(True)
                timer.deactivate()

        def on_run_again() -> None:
            show_progress()
            state.restart()
            timer.activate()

        timer = ui.timer(0.5, poll)
        with ui.row().classes('items-center'):
            ui.number('End time (s)', value=state.duration_s, min=1, step=1,
                      on_change=lambda e: setattr(state, 'duration_s', float(e.value)))
            ui.button('Run Again', on_click=on_run_again)
