from nicegui import ui

from slam.slam_solver import SlamSolver
from ui._utils import array_to_data_uri


class SlamTabState:
    def __init__(self, solver: SlamSolver) -> None:
        self._solver = solver


def slam_tab(state: SlamTabState) -> None:
    solver = state._solver

    with ui.column().classes('w-full'):
        progress_label = ui.label('')
        progress_bar = ui.linear_progress(value=0).classes('w-full')
        error_label = ui.label('').classes('text-red-500').set_visibility(False)
        img_positions = ui.image('').classes('w-full').set_visibility(False)
        img_attitudes = ui.image('').classes('w-full').set_visibility(False)

        def poll() -> None:
            if solver.loading:
                progress_label.text = solver.progress_label
                progress_bar.value = solver.progress
            elif solver.error:
                progress_label.set_visibility(False)
                progress_bar.set_visibility(False)
                error_label.text = f'Error: {solver.error}'
                error_label.set_visibility(True)
                timer.cancel()
            elif solver.plots is not None:
                progress_label.set_visibility(False)
                progress_bar.set_visibility(False)
                img_positions.source = array_to_data_uri(solver.plots.positions)
                img_attitudes.source = array_to_data_uri(solver.plots.attitudes_and_angular_velocities)
                img_positions.set_visibility(True)
                img_attitudes.set_visibility(True)
                timer.cancel()

        timer = ui.timer(0.5, poll)
