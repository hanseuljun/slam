from dataclasses import dataclass
from typing import Callable

from imgui_bundle import imgui


@dataclass
class TimeRangeState:
    start_s: float = 0.0
    duration_s: float = 5.0


def time_range_view(state: TimeRangeState, on_run: Callable[[], None]) -> None:
    imgui.text("Start time (s)")
    imgui.same_line()
    imgui.set_next_item_width(200)
    _, state.start_s = imgui.input_float("##start_s", state.start_s, step=1.0)
    imgui.same_line()
    imgui.text("Duration (s)")
    imgui.same_line()
    imgui.set_next_item_width(200)
    _, state.duration_s = imgui.input_float("##duration_s", state.duration_s, step=1.0)
    imgui.same_line()
    if imgui.button("Run Again"):
        on_run()
