from dataclasses import dataclass
from typing import Callable

from imgui_bundle import imgui


@dataclass
class TimeRangeModel:
    start_s: float = 0.0
    duration_s: float = 5.0


def time_range_view(model: TimeRangeModel, on_run: Callable[[], None]) -> None:
    imgui.text("Start time (s)")
    imgui.same_line()
    imgui.set_next_item_width(200)
    _, model.start_s = imgui.input_float("##start_s", model.start_s, step=1.0)
    imgui.same_line()
    imgui.text("Duration (s)")
    imgui.same_line()
    imgui.set_next_item_width(200)
    _, model.duration_s = imgui.input_float("##duration_s", model.duration_s, step=1.0)
    imgui.same_line()
    if imgui.button("Run Again"):
        on_run()
