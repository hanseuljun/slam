from pathlib import Path

from imgui_bundle import imgui, hello_imgui, immapp

from slam import DataFolder
from ui.camera_frames_tab import CameraFramesState, camera_frames_tab
from ui.data_tab import data_tab
from ui.slam_tab import SlamTabState, slam_tab
from ui.triangulation_tab import TriangulationTabState, triangulation_tab


class App:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.camera_frames = CameraFramesState(data)
        self.slam = SlamTabState(data)
        self.slam.start()
        self.triangulation = TriangulationTabState(data)
        self.triangulation.start()

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

        if imgui.begin_tab_bar("##tabs"):
            if imgui.begin_tab_item("Camera Frames")[0]:
                camera_frames_tab(self.camera_frames)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Data")[0]:
                data_tab(self.data)
                imgui.end_tab_item()

            if imgui.begin_tab_item("SLAM")[0]:
                slam_tab(self.slam)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Triangulation")[0]:
                triangulation_tab(self.triangulation)
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    app = App(data)

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "SLAM Visualizer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "main.ini"
    runner_params.callbacks.show_gui = app.render

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
