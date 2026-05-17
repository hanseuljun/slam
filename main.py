from pathlib import Path

from imgui_bundle import hello_imgui, immapp, imgui

from slam import DataFolder
from slam.viz import (
    CameraFramesState, camera_frames_tab,
    data_tab,
    SlamTabState, slam_tab,
    TriangulationTabState, triangulation_tab,
)

from imgui_bundle import immvision
immvision.use_bgr_color_order()


def make_gui(
    data: DataFolder,
    camera_frames: CameraFramesState,
    slam: SlamTabState,
    triangulation: TriangulationTabState,
):
    def gui():
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
            if imgui.begin_tab_item("Overview")[0]:
                imgui.end_tab_item()

            if imgui.begin_tab_item("Camera Frames")[0]:
                camera_frames_tab(camera_frames)
                imgui.end_tab_item()

            if imgui.begin_tab_item("SLAM")[0]:
                slam_tab(slam)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Triangulation")[0]:
                triangulation_tab(triangulation)
                imgui.end_tab_item()

            if imgui.begin_tab_item("Data")[0]:
                data_tab(data)
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

    return gui


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    camera_frames = CameraFramesState(data)
    slam = SlamTabState(data)
    slam.start()
    triangulation = TriangulationTabState(data)
    triangulation.start()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "SLAM Visualizer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "main.ini"
    runner_params.callbacks.show_gui = make_gui(data, camera_frames, slam, triangulation)

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
