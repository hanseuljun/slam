from pathlib import Path
from typing import Optional

from imgui_bundle import imgui, hello_imgui, immapp

from slam import EuRoCMAVData, FeatureDetectionResult, StereoMatchingResult
from ui.coordinate_mapping_view import CoordinateMappingViewModel, coordinate_mapping_view
from ui.data_view import DataViewModel, data_view
from ui.feature_detection_view import FeatureDetectionViewModel, feature_detection_view
from ui.slam_view import SlamViewModel, slam_view
from ui.stereo_matching_view import StereoMatchingViewModel, stereo_matching_view
from ui.config_view import ConfigViewModel, config_view

_DATA_PATHS = [
    "data/machine_hall/MH_01_easy/mav0",
    "data/machine_hall/MH_02_easy/mav0",
    "data/machine_hall/MH_03_medium/mav0",
    "data/machine_hall/MH_04_difficult/mav0",
    "data/machine_hall/MH_05_difficult/mav0",
]


class RootViewModel:
    def __init__(self) -> None:
        self.time_range_view_model = ConfigViewModel(data_paths=_DATA_PATHS)
        self.data = EuRoCMAVData.load(Path(self.time_range_view_model.data_path_str))
        self.data_view_model = DataViewModel(self.data)
        self.slam_view_model = SlamViewModel(self.data)
        self.feature_detection_result: Optional[FeatureDetectionResult] = None
        self.feature_detection_view_model = FeatureDetectionViewModel(
            self.data,
            on_result=self._on_feature_detection_result,
            start_s=self.time_range_view_model.start_s,
            duration_s=self.time_range_view_model.duration_s,
        )
        self.feature_detection_view_model.start()
        self.stereo_matching_result: Optional[StereoMatchingResult] = None
        self.stereo_matching_view_model = StereoMatchingViewModel(
            self.data,
            on_result=self._on_stereo_matching_result,
        )
        self.coordinate_mapping_view_model = CoordinateMappingViewModel(self.data)

    def _on_feature_detection_result(self, result: FeatureDetectionResult) -> None:
        self.feature_detection_result = result
        self.stereo_matching_view_model.start(result)

    def _on_stereo_matching_result(self, result: StereoMatchingResult) -> None:
        self.stereo_matching_result = result
        if self.time_range_view_model.run_coordinate_mapping_check:
            self.coordinate_mapping_view_model.start(self.feature_detection_result, result)
        self.slam_view_model.start(self.feature_detection_result, result)

    def restart(self) -> None:
        self.slam_view_model.stop()
        self.data = EuRoCMAVData.load(Path(self.time_range_view_model.data_path_str))
        self.data_view_model = DataViewModel(self.data)
        self.slam_view_model = SlamViewModel(self.data)
        self.stereo_matching_result = None
        self.feature_detection_result = None
        self.coordinate_mapping_view_model = CoordinateMappingViewModel(self.data)
        self.stereo_matching_view_model = StereoMatchingViewModel(
            self.data,
            on_result=self._on_stereo_matching_result,
        )
        self.feature_detection_view_model = FeatureDetectionViewModel(
            self.data,
            on_result=self._on_feature_detection_result,
            start_s=self.time_range_view_model.start_s,
            duration_s=self.time_range_view_model.duration_s,
        )
        self.feature_detection_view_model.start()


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

    config_view(model.time_range_view_model, model.restart)

    if imgui.begin_tab_bar("##tabs"):
        if imgui.begin_tab_item("Data")[0]:
            data_view(model.data_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Feature Detection")[0]:
            feature_detection_view(model.feature_detection_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Stereo Matching")[0]:
            stereo_matching_view(model.stereo_matching_view_model)
            imgui.end_tab_item()

        if model.time_range_view_model.run_coordinate_mapping_check:
            if imgui.begin_tab_item("Coordinate Mapping")[0]:
                coordinate_mapping_view(model.coordinate_mapping_view_model)
                imgui.end_tab_item()

        if imgui.begin_tab_item("SLAM")[0]:
            slam_view(model.slam_view_model)
            imgui.end_tab_item()

        imgui.end_tab_bar()

    imgui.end()


def main():
    model = RootViewModel()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "slam"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "slam.ini"
    runner_params.callbacks.show_gui = lambda: root_view(model)

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
