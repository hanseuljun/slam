from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from imgui_bundle import imgui


@dataclass
class ConfigViewModel:
    data_paths: list[str] = field(default_factory=list)
    selected_index: int = 0
    start_s: float = 0.0
    duration_s: float = 200.0
    # Whether to show the Frontend tab. Free to leave on: it reads the frontend data SLAM already
    # computes (SlamSolver.frontend_result) instead of running its own pass.
    run_frontend: bool = True
    # Off by default: unlike the Frontend tab this is real extra compute (it runs
    # CoordinateMappingChecker over SLAM's frontend output once SLAM finishes).
    run_coordinate_mapping_check: bool = False
    # Drives SLAM's own solve (SlamSolver's enable_loop_closure), unlike the diagnostics
    # above. On by default; see _LoopClosureDetector for the known tradeoff.
    run_loop_closure: bool = True

    @property
    def data_path_str(self) -> str:
        return self.data_paths[self.selected_index]


def config_view(model: ConfigViewModel, on_run: Callable[[], None]) -> None:
    # Stacked vertically (rather than same_line()-chained) since this now renders in the
    # collapsible left sidebar (see root_view in main.py) instead of a full-width top bar.
    labels = [Path(p).parent.name for p in model.data_paths]
    imgui.text("Dataset")
    imgui.set_next_item_width(-1)
    _, model.selected_index = imgui.combo("##data_path", model.selected_index, labels)

    imgui.text("Start (s)")
    imgui.set_next_item_width(-1)
    _, model.start_s = imgui.input_float("##start_s", model.start_s, step=1.0)

    imgui.text("Duration (s)")
    imgui.set_next_item_width(-1)
    _, model.duration_s = imgui.input_float("##duration_s", model.duration_s, step=1.0)

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    _, model.run_frontend = imgui.checkbox("Frontend tab", model.run_frontend)

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    _, model.run_coordinate_mapping_check = imgui.checkbox(
        "Coordinate Mapping Check", model.run_coordinate_mapping_check)
    _, model.run_loop_closure = imgui.checkbox(
        "Loop Closure", model.run_loop_closure
    )

    imgui.spacing()
    if imgui.button("Run Again", (-1, 0)):
        on_run()
