from pathlib import Path

# hello_imgui is registered into sys.modules dynamically at runtime (imgui_bundle/__init__.py's
# _publish()), not via a normal static submodule -- Pylance can't verify that against source,
# even though its real .pyi stub (imgui_bundle/hello_imgui.pyi) is still used for type info.
from imgui_bundle import imgui, hello_imgui, immapp  # pyright: ignore[reportMissingModuleSource]
from imgui_bundle import icons_fontawesome_6 as fa

from slam import EuRoCMAVData
from ui.coordinate_mapping_view import CoordinateMappingViewModel, coordinate_mapping_view
from ui.data_view import DataViewModel, data_view
from ui.frontend_view import FrontendViewModel, frontend_view
from ui.slam_view import SlamViewModel, slam_view
from ui.config_view import ConfigViewModel, config_view

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
    """Owns everything for one loaded dataset: the SLAM run (SlamViewModel -> SlamSolver, which
    computes the frontend -- feature detection + stereo matching + optical flow, see
    slam/frontend.py -- internally as part of its single per-frame pass) and the two diagnostic
    view models that only *read* its output: the Frontend tab (live, as frames are processed) and
    the Coordinate Mapping check (starts itself once SLAM finishes and its frontend data is
    complete). Neither diagnostic computes a frontend of its own, so showing them costs nothing
    extra beyond the coordinate mapping check's own pass.

    One object per run: restarting is "stop this Pipeline, build a new one," not "rewire a shared
    set of attributes in place" -- a late callback or poll from a superseded Pipeline can then only
    ever touch that Pipeline's own (by-then-cancelled) view models, never a newer one's (this is
    what let a stale callback reach into the wrong dataset's state before; see the cam0/cam1
    dataset-mismatch investigation).
    """

    def __init__(
        self,
        data: EuRoCMAVData,
        start_s: float,
        duration_s: float,
        run_loop_closure: bool,
    ) -> None:
        self.data = data
        self.slam_view_model = SlamViewModel(data, start_s, duration_s, run_loop_closure=run_loop_closure)
        self.frontend_view_model = FrontendViewModel(data, get_solver=lambda: self.slam_view_model.solver)
        self.coordinate_mapping_view_model = CoordinateMappingViewModel(
            data, get_solver=lambda: self.slam_view_model.solver)

    def start(self) -> None:
        self.slam_view_model.start()

    def stop(self) -> None:
        # Every stage, in one place: adding a stage here is the only thing needed to make
        # restart() stop it too, instead of a second call site that's easy to forget.
        self.coordinate_mapping_view_model.stop()
        self.slam_view_model.stop()


class RootViewModel:
    def __init__(self) -> None:
        self.time_range_view_model = ConfigViewModel(data_paths=_DATA_PATHS)
        data = EuRoCMAVData.load(Path(self.time_range_view_model.data_path_str))
        self.data_view_model = DataViewModel(data)
        self.pipeline = self._new_pipeline(data)
        self.pipeline.start()
        self.show_config: bool = True

    def _new_pipeline(self, data: EuRoCMAVData) -> Pipeline:
        cfg = self.time_range_view_model
        return Pipeline(
            data,
            start_s=cfg.start_s,
            duration_s=cfg.duration_s,
            run_loop_closure=cfg.run_loop_closure,
        )

    def restart(self) -> None:
        # Stop the current pipeline before anything reads self.pipeline again, so a stale thread
        # from the previous dataset can't deliver a result built from it into the new one.
        self.pipeline.stop()
        data = EuRoCMAVData.load(Path(self.time_range_view_model.data_path_str))
        self.data_view_model = DataViewModel(data)
        self.pipeline = self._new_pipeline(data)
        self.pipeline.start()


_CONFIG_SIDEBAR_WIDTH = 260


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

    # Hamburger toggles the config sidebar rather than always showing it, so the tab content
    # below (plots especially) gets the full window width back when it's not needed. Lives
    # inside the sidebar itself while it's open (so it reads as part of that panel), and falls
    # back to the main area -- the only place left -- once the sidebar is closed, so there's
    # always a way to bring it back.
    if model.show_config:
        # No built-in child border (that draws all four sides) -- only the edge against the
        # main content should read as a divider, drawn manually after the child closes.
        imgui.begin_child("##config_sidebar", (_CONFIG_SIDEBAR_WIDTH, 0), False)
        if imgui.button(fa.ICON_FA_BARS):
            model.show_config = not model.show_config
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        config_view(model.time_range_view_model, model.restart)
        imgui.end_child()
        rect_min = imgui.get_item_rect_min()
        rect_max = imgui.get_item_rect_max()
        imgui.get_window_draw_list().add_line(
            (rect_max.x, rect_min.y), (rect_max.x, rect_max.y),
            imgui.get_color_u32(imgui.Col_.border))
    else:
        if imgui.button(fa.ICON_FA_BARS):
            model.show_config = not model.show_config

    imgui.same_line()
    imgui.begin_child("##main_content", (0, 0), False)

    pipeline = model.pipeline
    if imgui.begin_tab_bar("##tabs"):
        if imgui.begin_tab_item("Data")[0]:
            data_view(model.data_view_model)
            imgui.end_tab_item()

        if model.time_range_view_model.run_frontend:
            if imgui.begin_tab_item("Frontend")[0]:
                frontend_view(pipeline.frontend_view_model)
                imgui.end_tab_item()

        if model.time_range_view_model.run_coordinate_mapping_check:
            if imgui.begin_tab_item("Coordinate Mapping")[0]:
                coordinate_mapping_view(pipeline.coordinate_mapping_view_model)
                imgui.end_tab_item()

        if imgui.begin_tab_item("SLAM")[0]:
            slam_view(pipeline.slam_view_model)
            imgui.end_tab_item()

        imgui.end_tab_bar()

    imgui.end_child()

    imgui.end()


def main():
    model = RootViewModel()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "slam"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "slam.ini"
    runner_params.callbacks.show_gui = lambda: root_view(model)

    # with_implot/with_implot3d: the SLAM view's plots are ImPlot/ImPlot3D-based (see
    # ui/slam_view.py) -- both need their own context created before any begin_plot call, which
    # this handles.
    immapp.run(
        runner_params,
        add_ons_params=immapp.AddOnsParams(with_implot=True, with_implot3d=True))


if __name__ == "__main__":
    main()
