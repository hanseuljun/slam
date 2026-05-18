from pathlib import Path

from nicegui import ui

from slam import DataFolder
from slam.slam_solver import SlamSolver
from ui.camera_frames_tab import CameraFramesTabState, camera_frames_tab
from ui.slam_tab import SlamTabState, slam_tab
from ui.triangulation_tab import TriangulationTabState, triangulation_tab


class App:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.camera_frames_tab_state = CameraFramesTabState(data)
        self.slam_solver = SlamSolver(data)
        self.slam_solver.start()
        self.slam_tab_state = SlamTabState(self.slam_solver, on_restart=self.restart_slam)
        self.triangulation_tab_state = TriangulationTabState(data)
        self.triangulation_tab_state.start()

    def restart_slam(self) -> None:
        self.slam_solver._stop_event.set()
        self.slam_solver = SlamSolver(self.data, self.slam_tab_state.start_s, self.slam_tab_state.duration_s)
        self.slam_solver.start()
        self.slam_tab_state._solver = self.slam_solver

    def setup_ui(self) -> None:
        with ui.tabs().classes('w-full') as tabs:
            tab_camera_frames = ui.tab('Camera Frames')
            tab_slam = ui.tab('SLAM')
            tab_triangulation = ui.tab('Triangulation')

        with ui.row().classes('items-center'):
            ui.number('Start time (s)', value=self.slam_tab_state.start_s, min=0, step=1,
                      on_change=lambda e: setattr(self.slam_tab_state, 'start_s', float(e.value)))
            ui.number('Duration (s)', value=self.slam_tab_state.duration_s, min=1, step=1,
                      on_change=lambda e: setattr(self.slam_tab_state, 'duration_s', float(e.value)))
            ui.button('Run Again', on_click=lambda: self.slam_tab_state.on_run_again())

        with ui.tab_panels(tabs, value=tab_camera_frames).classes('w-full'):
            with ui.tab_panel(tab_camera_frames):
                camera_frames_tab(self.camera_frames_tab_state)
            with ui.tab_panel(tab_slam):
                slam_tab(self.slam_tab_state)
            with ui.tab_panel(tab_triangulation):
                triangulation_tab(self.triangulation_tab_state)


def main() -> None:
    data = DataFolder.load(Path('data/machine_hall/MH_01_easy/mav0'))
    instance = App(data)
    instance.setup_ui()
    ui.run(title='slam')

if __name__ in {"__main__", "__mp_main__"}:
    main()
