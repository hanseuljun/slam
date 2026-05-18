from nicegui import ui

from slam.data import DataFolder
from slam.slam_solver import SlamSolver
from ui._utils import array_to_data_uri


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
                img_positions.source = array_to_data_uri(solver.plots.positions)
                img_attitudes.source = array_to_data_uri(solver.plots.attitudes_and_angular_velocities)
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
