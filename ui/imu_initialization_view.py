import threading
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.data import EuRoCMAVData
from slam.imu_initialization import ImuInitializationResult, ImuInitializationSolver
from ui.utils import figure_to_image, image_to_texture


def _plot_norms(result: ImuInitializationResult) -> plt.Figure:
    fig, (ax_acc, ax_gyro) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('IMU Norms — Static Period Detection')

    ax_acc.plot(result.times, result.lin_acc_norms)
    ax_acc.axvspan(result.static_start_s, result.static_end_s, alpha=0.3, color='green', label='static')
    ax_acc.set_ylabel('||acc|| [m/s²]')
    ax_acc.legend()

    ax_gyro.plot(result.times, result.ang_vel_norms)
    ax_gyro.axvspan(result.static_start_s, result.static_end_s, alpha=0.3, color='green', label='static')
    ax_gyro.set_xlabel('Time [s]')
    ax_gyro.set_ylabel('||gyro|| [rad/s]')
    ax_gyro.legend()

    plt.tight_layout()
    return fig


class ImuInitializationViewModel:
    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data
        self._result: Optional[ImuInitializationResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def start(self) -> None:
        self._result = None
        self._loading = True
        self._error = None
        self._texture = None
        threading.Thread(
            target=self._compute,
            args=(ImuInitializationSolver(self._data),),
            daemon=True,
        ).start()

    def _compute(self, solver: ImuInitializationSolver) -> None:
        try:
            self._result = solver.run()
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False


def imu_initialization_view(model: ImuInitializationViewModel) -> None:
    if model._loading:
        imgui.text("Computing IMU initialization...")
        return
    if model._error:
        imgui.text(f"Error: {model._error}")
        return
    if model._result is None:
        return

    result = model._result

    if model._texture is None:
        model._texture = image_to_texture(figure_to_image(_plot_norms(result)))
    tex = model._texture
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))

    imgui.text(f"Static period: {result.static_start_s:.3f} s — {result.static_end_s:.3f} s")

    g = result.gravity_in_body
    imgui.text(f"Gravity in body frame:  [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]  magnitude: {np.linalg.norm(g):.4f} m/s²")

    gw = result.gravity_in_world
    imgui.text(f"Gravity in world frame: [{gw[0]:.4f}, {gw[1]:.4f}, {gw[2]:.4f}]  magnitude: {np.linalg.norm(gw):.4f} m/s²")
