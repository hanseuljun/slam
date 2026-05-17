from typing import Optional

import cv2
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam import DataFolder
from slam.slam_solver import SlamSolver


def _to_texture(image: np.ndarray) -> hello_imgui.TextureGpu:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return hello_imgui.create_texture_gpu_from_rgba_data(rgba)


class SlamTabState:
    def __init__(self, data: DataFolder) -> None:
        self._solver = SlamSolver(data)
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None

    def start(self) -> None:
        self._solver.start()


def slam_tab(state: SlamTabState) -> None:
    solver = state._solver
    if solver.loading:
        imgui.text(solver.progress_label)
        imgui.progress_bar(solver.progress, (-1, 0))
        return
    if solver.error:
        imgui.text(f"Error: {solver.error}")
        return
    if solver.plots is None:
        return

    if state._tex_positions is None:
        state._tex_positions = _to_texture(solver.plots.positions)
        state._tex_attitudes = _to_texture(solver.plots.attitudes_and_angular_velocities)

    imgui.begin_child("##slam_scroll", (0, 0), False)
    imgui.image(imgui.ImTextureRef(state._tex_positions.texture_id()), (state._tex_positions.width, state._tex_positions.height))
    imgui.image(imgui.ImTextureRef(state._tex_attitudes.texture_id()), (state._tex_attitudes.width, state._tex_attitudes.height))
    imgui.end_child()
