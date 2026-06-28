from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from imgui_bundle import imgui


@dataclass
class ConfigViewModel:
    data_paths: list[str] = field(default_factory=list)
    selected_index: int = 0
    start_s: float = 0.0
    duration_s: float = 10.0
    run_coordinate_mapping_check: bool = False

    @property
    def data_path_str(self) -> str:
        return self.data_paths[self.selected_index]


def config_view(model: ConfigViewModel, on_run: Callable[[], None]) -> None:
    labels = [Path(p).parent.name for p in model.data_paths]
    imgui.set_next_item_width(180)
    changed, model.selected_index = imgui.combo("##data_path", model.selected_index, labels)
    if changed:
        on_run()
    imgui.same_line()
    imgui.text("Start (s)")
    imgui.same_line()
    imgui.set_next_item_width(150)
    _, model.start_s = imgui.input_float("##start_s", model.start_s, step=1.0)
    imgui.same_line()
    imgui.text("Duration (s)")
    imgui.same_line()
    imgui.set_next_item_width(150)
    _, model.duration_s = imgui.input_float("##duration_s", model.duration_s, step=1.0)
    imgui.same_line()
    _, model.run_coordinate_mapping_check = imgui.checkbox(
        "Coordinate Mapping Check", model.run_coordinate_mapping_check
    )
    imgui.same_line()
    if imgui.button("Run Again"):
        on_run()
