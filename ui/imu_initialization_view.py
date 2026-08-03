import threading
from typing import Optional

import numpy as np
from imgui_bundle import imgui, implot

from slam.data import EuRoCMAVData
from slam.imu_initialization import ImuInitializationResult, ImuInitializationSolver


def _draw_norms(result: ImuInitializationResult) -> None:
    imgui.text('IMU Norms — Static Period Detection')
    static_bounds = np.array([result.static_start_s, result.static_end_s], dtype=np.float64)
    if implot.begin_subplots("##imu_norms", 2, 1, size=(-1, 400), flags=implot.SubplotFlags_.link_all_x):
        for values, ylabel in [
            (result.lin_acc_norms, '||acc|| [m/s²]'),
            (result.ang_vel_norms, '||gyro|| [rad/s]'),
        ]:
            if implot.begin_plot(f"##imu_norms_{ylabel}"):
                implot.setup_axes('Time [s]', ylabel)
                implot.plot_line(
                    ylabel, np.ascontiguousarray(result.times, dtype=np.float64),
                    np.ascontiguousarray(values, dtype=np.float64))
                # Vertical lines marking the detected static period -- ImPlot's equivalent of
                # matplotlib's axvspan, minus the fill (a filled span needs a known y-range to
                # shade between, which ImPlot's autofit doesn't expose up front).
                implot.plot_inf_lines('static period', static_bounds)
                implot.end_plot()
        implot.end_subplots()


class ImuInitializationViewModel:
    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data
        self._cancel_event = threading.Event()
        self._result: Optional[ImuInitializationResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None

    def start(self) -> None:
        self._result = None
        self._loading = True
        self._error = None
        threading.Thread(
            target=self._compute,
            args=(ImuInitializationSolver(self._data, cancel_event=self._cancel_event),),
            daemon=True,
        ).start()

    def stop(self) -> None:
        self._cancel_event.set()

    def _compute(self, solver: ImuInitializationSolver) -> None:
        try:
            result = solver.run()
            if self._cancel_event.is_set():
                return
            self._result = result
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

    _draw_norms(result)

    imgui.text(f"Static period: {result.static_start_s:.3f} s — {result.static_end_s:.3f} s")

    g = result.gravity_in_body
    imgui.text(f"Gravity in body frame:  [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]  magnitude: {np.linalg.norm(g):.4f} m/s²")

    gw = result.gravity_in_world
    imgui.text(f"Gravity in world frame: [{gw[0]:.4f}, {gw[1]:.4f}, {gw[2]:.4f}]  magnitude: {np.linalg.norm(gw):.4f} m/s²")
