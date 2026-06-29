from imgui_bundle import imgui

from slam.data import EuRoCMAVData


class ImuInitializationViewModel:
    def __init__(self, data: EuRoCMAVData) -> None:
        self._data = data


def imu_initialization_view(model: ImuInitializationViewModel) -> None:
    imgui.text("IMU Initialization")
