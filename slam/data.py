import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import yaml


def read_timestamps(data_csv_path: Path) -> list[int]:
    """Read timestamps from a data.csv file."""
    timestamps = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timestamp_ns = int(row[0])
            timestamps.append(timestamp_ns)
    return timestamps


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_sensor_yaml(cls, path: Path) -> Self:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        intrinsics = data["intrinsics"]
        return cls(fx=intrinsics[0], fy=intrinsics[1], cx=intrinsics[2], cy=intrinsics[3])

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ])


def read_extrinsics(sensor_yaml_path: Path) -> np.ndarray:
    """Read camera extrinsics (T_BS) from sensor.yaml as a 4x4 matrix."""
    with open(sensor_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    t_bs = data["T_BS"]
    return np.array(t_bs["data"]).reshape(t_bs["rows"], t_bs["cols"])


@dataclass
class ImuSample:
    timestamp: int
    angular_velocity: tuple[float, float, float]  # w_x, w_y, w_z [rad/s]
    linear_acceleration: tuple[float, float, float]  # a_x, a_y, a_z [m/s^2]


def read_imu_samples(data_csv_path: Path) -> list[ImuSample]:
    """Read IMU data from a data.csv file."""
    imu_samples = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timestamp = int(row[0])
            angular_velocity = (float(row[1]), float(row[2]), float(row[3]))
            linear_acceleration = (float(row[4]), float(row[5]), float(row[6]))
            imu_samples.append(ImuSample(timestamp, angular_velocity, linear_acceleration))
    return imu_samples


@dataclass
class DataFolder:
    path: Path
    cam_timestamps: list[int]
    imu_samples: list[ImuSample]
    cam0_extrinsics: np.ndarray  # 4x4 transformation matrix (T_BS)
    cam1_extrinsics: np.ndarray  # 4x4 transformation matrix (T_BS)
    cam0_intrinsics: CameraIntrinsics
    cam1_intrinsics: CameraIntrinsics

    @classmethod
    def load(cls, path: Path) -> Self:
        cam0_timestamps = read_timestamps(path / "cam0" / "data.csv")
        cam1_timestamps = read_timestamps(path / "cam1" / "data.csv")
        if cam0_timestamps != cam1_timestamps:
            raise ValueError("cam0 and cam1 timestamps do not match")
        imu_samples = read_imu_samples(path / "imu0" / "data.csv")
        cam0_extrinsics = read_extrinsics(path / "cam0" / "sensor.yaml")
        cam1_extrinsics = read_extrinsics(path / "cam1" / "sensor.yaml")
        cam0_intrinsics = CameraIntrinsics.from_sensor_yaml(path / "cam0" / "sensor.yaml")
        cam1_intrinsics = CameraIntrinsics.from_sensor_yaml(path / "cam1" / "sensor.yaml")
        return cls(
            path,
            cam0_timestamps,
            imu_samples,
            cam0_extrinsics,
            cam1_extrinsics,
            cam0_intrinsics,
            cam1_intrinsics,
        )

    def get_cam0_image_path(self, timestamp: int) -> Path:
        return self.path / "cam0" / "data" / f"{timestamp}.png"

    def get_cam1_image_path(self, timestamp: int) -> Path:
        return self.path / "cam1" / "data" / f"{timestamp}.png"
