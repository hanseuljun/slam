from pathlib import Path
from typing import Optional

# hello_imgui is registered into sys.modules dynamically at runtime (imgui_bundle/__init__.py's
# _publish()), not via a normal static submodule -- Pylance can't verify that against source,
# even though its real .pyi stub (imgui_bundle/hello_imgui.pyi) is still used for type info.
from imgui_bundle import imgui, hello_imgui, immapp  # pyright: ignore[reportMissingModuleSource]

from slam import EuRoCMAVData, FeatureDetectionResult, OpticalFlowResult, StereoMatchingResult
from ui.coordinate_mapping_view import CoordinateMappingViewModel, coordinate_mapping_view
from ui.data_view import DataViewModel, data_view
from ui.feature_detection_view import FeatureDetectionViewModel, feature_detection_view
from ui.optical_flow_view import OpticalFlowViewModel, optical_flow_view
from ui.slam_view import SlamViewModel, slam_view
from ui.stereo_matching_view import StereoMatchingViewModel, stereo_matching_view
from ui.config_view import ConfigViewModel, config_view
from ui.imu_initialization_view import ImuInitializationViewModel, imu_initialization_view

_DATA_PATHS = [
    "data/machine_hall/MH_01_easy/mav0",
    "data/machine_hall/MH_02_easy/mav0",
    "data/machine_hall/MH_03_medium/mav0",
    "data/machine_hall/MH_04_difficult/mav0",
    "data/machine_hall/MH_05_difficult/mav0",
    "data/vicon_room1/V1_01_easy/mav0",
    "data/vicon_room1/V1_02_medium/mav0",
    "data/vicon_room1/V1_03_difficult/mav0",
    "data/vicon_room2/V2_01_easy/mav0",
    "data/vicon_room2/V2_02_medium/mav0",
    "data/vicon_room2/V2_03_difficult/mav0",
]


class Pipeline:
    """Owns the feature detection -> (optical flow, stereo matching) -> (coordinate mapping, SLAM)
    chain for one loaded dataset. One object per run: constructing a Pipeline wires every stage's on_result
    callback to the next stage on *this* instance, so a callback firing late from a superseded
    Pipeline can only ever touch that Pipeline's own (by-then-cancelled) view models, never a
    newer one's -- restarting is "stop this Pipeline, build a new one," not "rewire a shared set
    of attributes in place," which is what let a stale callback reach into the wrong dataset's
    state before (see stop(), and the cam0/cam1 dataset-mismatch investigation this came from).
    """

    def __init__(
        self,
        data: EuRoCMAVData,
        start_s: float,
        duration_s: float,
        run_coordinate_mapping_check: bool,
        run_imu_initialization: bool,
    ) -> None:
        self.data = data
        self.feature_detection_result: Optional[FeatureDetectionResult] = None
        self.stereo_matching_result: Optional[StereoMatchingResult] = None
        self.optical_flow_result: Optional[OpticalFlowResult] = None
        self._run_coordinate_mapping_check = run_coordinate_mapping_check
        self._run_imu_initialization = run_imu_initialization

        self.feature_detection_view_model = FeatureDetectionViewModel(
            data, on_result=self._on_feature_detection_result, start_s=start_s, duration_s=duration_s)
        self.optical_flow_view_model = OpticalFlowViewModel(data, on_result=self._on_optical_flow_result)
        self.stereo_matching_view_model = StereoMatchingViewModel(data, on_result=self._on_stereo_matching_result)
        self.coordinate_mapping_view_model = CoordinateMappingViewModel(data)
        self.imu_initialization_view_model = ImuInitializationViewModel(data)
        self.slam_view_model = SlamViewModel(data)

    def start(self) -> None:
        self.feature_detection_view_model.start()
        if self._run_imu_initialization:
            self.imu_initialization_view_model.start()

    def stop(self) -> None:
        # Every stage, in one place: adding a stage here is the only thing needed to make
        # restart() stop it too, instead of a second call site that's easy to forget.
        self.feature_detection_view_model.stop()
        self.optical_flow_view_model.stop()
        self.stereo_matching_view_model.stop()
        self.coordinate_mapping_view_model.stop()
        self.imu_initialization_view_model.stop()
        self.slam_view_model.stop()

    def _on_feature_detection_result(self, result: FeatureDetectionResult) -> None:
        self.feature_detection_result = result
        self.stereo_matching_view_model.start(result)

    def _on_optical_flow_result(self, result: OpticalFlowResult) -> None:
        self.optical_flow_result = result
        # Always set by now: optical flow only starts (from _on_stereo_matching_result) after
        # both of these are -- Pylance can't prove that invariant from the Optional types.
        assert self.feature_detection_result is not None
        assert self.stereo_matching_result is not None
        # SLAM waits on optical flow too now: _build_landmark_tracks links landmark observations
        # across keyframes using optical flow's track ids instead of re-matching ORB descriptors.
        self.slam_view_model.start(self.feature_detection_result, self.stereo_matching_result, result)

    def _on_stereo_matching_result(self, result: StereoMatchingResult) -> None:
        self.stereo_matching_result = result
        # Always set by now: this callback only fires after _on_feature_detection_result already
        # set it, and stereo matching starts strictly after that -- Pylance can't prove that
        # invariant from the attribute's Optional type.
        assert self.feature_detection_result is not None
        # Optical flow runs after stereo matching (rather than in parallel with it) so it can
        # reuse stereo matching's per-frame cam0<->cam1 match instead of redoing it for seeding.
        self.optical_flow_view_model.start(self.feature_detection_result, result)
        if self._run_coordinate_mapping_check:
            self.coordinate_mapping_view_model.start(self.feature_detection_result, result)


class RootViewModel:
    def __init__(self) -> None:
        self.time_range_view_model = ConfigViewModel(data_paths=_DATA_PATHS)
        data = EuRoCMAVData.load(Path(self.time_range_view_model.data_path_str))
        self.data_view_model = DataViewModel(data)
        self.pipeline = self._new_pipeline(data)
        self.pipeline.start()

    def _new_pipeline(self, data: EuRoCMAVData) -> Pipeline:
        cfg = self.time_range_view_model
        return Pipeline(
            data,
            start_s=cfg.start_s,
            duration_s=cfg.duration_s,
            run_coordinate_mapping_check=cfg.run_coordinate_mapping_check,
            run_imu_initialization=cfg.run_imu_initialization,
        )

    def restart(self) -> None:
        # Stop the current pipeline before anything reads self.pipeline again, so a stale thread
        # from the previous dataset can't deliver a result built from it into the new one.
        self.pipeline.stop()
        data = EuRoCMAVData.load(Path(self.time_range_view_model.data_path_str))
        self.data_view_model = DataViewModel(data)
        self.pipeline = self._new_pipeline(data)
        self.pipeline.start()


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

    pipeline = model.pipeline
    if imgui.begin_tab_bar("##tabs"):
        if imgui.begin_tab_item("Data")[0]:
            data_view(model.data_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Feature Detection")[0]:
            feature_detection_view(pipeline.feature_detection_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Stereo Matching")[0]:
            stereo_matching_view(pipeline.stereo_matching_view_model)
            imgui.end_tab_item()

        if imgui.begin_tab_item("Optical Flow")[0]:
            optical_flow_view(pipeline.optical_flow_view_model)
            imgui.end_tab_item()

        if model.time_range_view_model.run_coordinate_mapping_check:
            if imgui.begin_tab_item("Coordinate Mapping")[0]:
                coordinate_mapping_view(pipeline.coordinate_mapping_view_model)
                imgui.end_tab_item()

        if model.time_range_view_model.run_imu_initialization:
            if imgui.begin_tab_item("IMU Initialization")[0]:
                imu_initialization_view(pipeline.imu_initialization_view_model)
                imgui.end_tab_item()

        if imgui.begin_tab_item("SLAM")[0]:
            slam_view(pipeline.slam_view_model)
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
