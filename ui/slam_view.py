import io
from typing import Optional

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.plot import plot_angular_velocities, plot_attitudes, plot_positions
from slam.slam_solver import SlamResults, SlamSolver


def _to_texture(image: np.ndarray) -> hello_imgui.TextureGpu:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return hello_imgui.create_texture_gpu_from_rgba_data(rgba)


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return img


def _render_plots(results: SlamResults) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fig_pos = plot_positions(series=[
        (results.pnp_times, results.pnp_positions, 'pnp'),
        (results.gt_times, results.gt_positions, 'gt'),
    ])
    fig_att = plot_attitudes(series=[
        (results.pnp_times, results.pnp_attitudes, 'pnp'),
        (results.imu_attitude_times, results.imu_attitudes, 'imu'),
        (results.gt_times, results.gt_attitudes, 'gt'),
        (results.pnp_times, results.optimized_attitudes, 'opt'),
    ])
    fig_omega = plot_angular_velocities(series=[
        (results.pnp_angular_velocity_times, results.pnp_angular_velocities, 'pnp'),
        (results.imu_times, results.imu_angular_velocities, 'imu'),
        (results.pnp_times, results.imu_angular_velocities_at_cam_times, 'imu@cam'),
        (results.gt_angular_velocity_times, results.gt_angular_velocities, 'gt'),
        (results.pnp_angular_velocity_times, results.optimized_angular_velocities, 'opt'),
    ])
    return _fig_to_image(fig_pos), _fig_to_image(fig_att), _fig_to_image(fig_omega)


class SlamViewModel:
    def __init__(self, solver: SlamSolver) -> None:
        self._solver = solver
        self._last_solver: Optional[SlamSolver] = None
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._tex_angular_velocities: Optional[hello_imgui.TextureGpu] = None


def slam_view(model: SlamViewModel) -> None:
    if model._last_solver is not model._solver:
        model._tex_positions = None
        model._tex_attitudes = None
        model._tex_angular_velocities = None
        model._last_solver = model._solver

    solver = model._solver
    if solver.loading:
        imgui.text(solver.progress_label)
        imgui.progress_bar(solver.progress, (-1, 0))
        return
    if solver.error:
        imgui.text(f"Error: {solver.error}")
        return
    if solver.plots is None:
        return

    if model._tex_positions is None:
        pos_img, att_img, omega_img = _render_plots(solver.plots)
        model._tex_positions = _to_texture(pos_img)
        model._tex_attitudes = _to_texture(att_img)
        model._tex_angular_velocities = _to_texture(omega_img)

    imgui.begin_child("##slam_scroll", (0, 0), False)
    tex = model._tex_positions
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    tex = model._tex_attitudes
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    tex = model._tex_angular_velocities
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.end_child()
