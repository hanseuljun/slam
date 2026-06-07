from pathlib import Path

from imgui_bundle import imgui, hello_imgui, immapp

from slam import DataFolder
from slam.slam_solver import SlamSolver
from ui.camera_frames_tab import CameraFramesTabState, camera_frames_tab
from ui.slam_tab import SlamTabState, slam_tab
from ui.time_range_view import TimeRangeState, time_range_view
from ui.triangulation_tab import TriangulationTabState, triangulation_tab


class App:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.camera_frames_tab_state = CameraFramesTabState(data)
        self.slam_solver = SlamSolver(data)
        self.slam_solver.start()
        self.slam_tab_state = SlamTabState(self.slam_solver)
        self.time_range_state = TimeRangeState()
        self.triangulation_tab_state = TriangulationTabState(data)
        self.triangulation_tab_state.start()

    def restart_slam(self) -> None:
        self.slam_solver._stop_event.set()
        self.slam_solver = SlamSolver(
            self.data,
            self.time_range_state.start_s,
            self.time_range_state.duration_s,
        )
        self.slam_solver.start()
        self.slam_tab_state._solver = self.slam_solver

    def render(self) -> None:
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.work_pos)
        imgui.set_next_window_size(viewport.work_size)
        imgui.begin(
            "##main",
            flags=imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar,
        )

        time_range_view(self.time_range_state, self.restart_slam)

        if imgui.begin_tab_bar("##tabs"):
            if imgui.begin_tab_item("Camera Frames")[0]:
                camera_frames_tab(self.camera_frames_tab_state)
                imgui.end_tab_item()

            if imgui.begin_tab_item("SLAM")[0]:
                slam_tab(self.slam_tab_state)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Triangulation")[0]:
                triangulation_tab(self.triangulation_tab_state)
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    app = App(data)

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "slam"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "slam.ini"
    runner_params.callbacks.show_gui = app.render

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
