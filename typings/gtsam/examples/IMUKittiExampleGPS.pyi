import argparse
import gtsam
from _typeshed import Incomplete
from gtsam import ISAM2 as ISAM2, Pose3 as Pose3, noiseModel as noiseModel
from gtsam.symbol_shorthand import B as B, V as V, X as X

GRAVITY: float

class KittiCalibration:
    bodyTimu: Incomplete
    accelerometer_sigma: Incomplete
    gyroscope_sigma: Incomplete
    integration_sigma: Incomplete
    accelerometer_bias_sigma: Incomplete
    gyroscope_bias_sigma: Incomplete
    average_delta_t: Incomplete
    def __init__(self, body_ptx: float, body_pty: float, body_ptz: float, body_prx: float, body_pry: float, body_prz: float, accelerometer_sigma: float, gyroscope_sigma: float, integration_sigma: float, accelerometer_bias_sigma: float, gyroscope_bias_sigma: float, average_delta_t: float) -> None: ...

class ImuMeasurement:
    time: Incomplete
    dt: Incomplete
    accelerometer: Incomplete
    gyroscope: Incomplete
    def __init__(self, time: float, dt: float, accelerometer: gtsam.Point3, gyroscope: gtsam.Point3) -> None: ...

class GpsMeasurement:
    time: Incomplete
    position: Incomplete
    def __init__(self, time: float, position: gtsam.Point3) -> None: ...

def loadImuData(imu_data_file: str) -> list[ImuMeasurement]: ...
def loadGpsData(gps_data_file: str) -> list[GpsMeasurement]: ...
def loadKittiData(imu_data_file: str = 'KittiEquivBiasedImu.txt', gps_data_file: str = 'KittiGps_converted.txt', imu_metadata_file: str = 'KittiEquivBiasedImu_metadata.txt') -> tuple[KittiCalibration, list[ImuMeasurement], list[GpsMeasurement]]: ...
def getImuParams(kitti_calibration: KittiCalibration): ...
def save_results(isam: gtsam.ISAM2, output_filename: str, first_gps_pose: int, gps_measurements: list[GpsMeasurement]): ...
def parse_args() -> argparse.Namespace: ...
def optimize(gps_measurements: list[GpsMeasurement], imu_measurements: list[ImuMeasurement], sigma_init_x: gtsam.noiseModel.Diagonal, sigma_init_v: gtsam.noiseModel.Diagonal, sigma_init_b: gtsam.noiseModel.Diagonal, noise_model_gps: gtsam.noiseModel.Diagonal, kitti_calibration: KittiCalibration, first_gps_pose: int, gps_skip: int) -> gtsam.ISAM2: ...
def main() -> None: ...
