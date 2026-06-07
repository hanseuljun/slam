from pathlib import Path

from imgui_bundle import imgui, hello_imgui, immapp

from slam import DataFolder
from slam.slam_solver import SlamSolver
from ui.data_view import DataViewModel, data_view
from ui.feature_detection_view import FeatureDetectionViewModel, feature_detection_view
from ui.slam_view import SlamViewModel, slam_view
from ui.time_range_view import TimeRangeModel, time_range_view
from ui.triangulation_view import TriangulationViewModel, triangulation_view


class RootViewModel:
    def __init__(self, data: DataFolder) -> None:
        self.data = data
        self.data_view_model = DataViewModel(data)
        self.slam_solver = SlamSolver(data)
        self.slam_solver.start()
        self.slam_view_model = SlamViewModel(self.slam_solver)
        self.time_range_model = TimeRangeModel()
        self.feature_detection_view_model = FeatureDetectionViewModel(data)
        self.feature_detection_view_model.start()
        self.triangulation_view_model = TriangulationViewModel(data)
        self.triangulation_view_model.start()

    def restart_slam(self) -> None:
        self.slam_solver._stop_event.set()
        self.slam_solver = SlamSolver(
            self.data,
            self.time_range_model.start_s,
            self.time_range_model.duration_s,
        )
        self.slam_solver.start()
        self.slam_view_model._solver = self.slam_solver


def root_view(model: RootViewModel) -> None:
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

    time_range_view(model.time_range_model, model.restart_slam)

    if imgui.begin_tab_bar("##tabs"):
        if imgui.begin_tab_item("Data")[0]:
            data_view(model.data_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Feature Detection")[0]:
            feature_detection_view(model.feature_detection_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Triangulation")[0]:
            triangulation_view(model.triangulation_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("SLAM")[0]:
            slam_view(model.slam_view_model)
            imgui.end_tab_item()

        imgui.end_tab_bar()

    imgui.end()


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    model = RootViewModel(data)

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "slam"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "slam.ini"
    runner_params.callbacks.show_gui = lambda: root_view(model)

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
