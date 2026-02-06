import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import yaml


def read_timestamps(data_csv_path: Path) -> list[int]:
    """Read timestamps from a data.csv file."""
    timestamps_ns = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timestamp_ns = int(row[0])
            timestamps_ns.append(timestamp_ns)
    return timestamps_ns


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float  # radial distortion coefficient
    k2: float  # radial distortion coefficient
    p1: float  # tangential distortion coefficient
    p2: float  # tangential distortion coefficient

    @classmethod
    def from_sensor_yaml(cls, path: Path) -> Self:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        intrinsics = data["intrinsics"]
        distortion = data["distortion_coefficients"]
        return cls(
            fx=intrinsics[0],
            fy=intrinsics[1],
            cx=intrinsics[2],
            cy=intrinsics[3],
            k1=distortion[0],
            k2=distortion[1],
            p1=distortion[2],
            p2=distortion[3],
        )

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
    timestamp_ns: int
    angular_velocity: tuple[float, float, float]  # w_x, w_y, w_z [rad/s]
    linear_acceleration: tuple[float, float, float]  # a_x, a_y, a_z [m/s^2]


@dataclass
class GroundTruthSample:
    timestamp_ns: int
    position: tuple[float, float, float]  # p_RS_R_x, p_RS_R_y, p_RS_R_z [m]
    quaternion: tuple[float, float, float, float]  # q_RS_w, q_RS_x, q_RS_y, q_RS_z
    velocity: tuple[float, float, float]  # v_RS_R_x, v_RS_R_y, v_RS_R_z [m/s]
    gyroscope_bias: tuple[float, float, float]  # b_w_RS_S_x, b_w_RS_S_y, b_w_RS_S_z [rad/s]
    accelerometer_bias: tuple[float, float, float]  # b_a_RS_S_x, b_a_RS_S_y, b_a_RS_S_z [m/s^2]


@dataclass
class LeicaSample:
    timestamp_ns: int
    position: tuple[float, float, float]  # p_RS_R_x, p_RS_R_y, p_RS_R_z [m]


def read_leica_samples(data_csv_path: Path) -> list[LeicaSample]:
    """Read Leica position data from a data.csv file."""
    samples = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            sample = LeicaSample(
                timestamp_ns=int(row[0]),
                position=(float(row[1]), float(row[2]), float(row[3])),
            )
            samples.append(sample)
    return samples


def read_ground_truth_samples(data_csv_path: Path) -> list[GroundTruthSample]:
    """Read ground truth data from a data.csv file."""
    samples = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            sample = GroundTruthSample(
                timestamp_ns=int(row[0]),
                position=(float(row[1]), float(row[2]), float(row[3])),
                quaternion=(float(row[4]), float(row[5]), float(row[6]), float(row[7])),
                velocity=(float(row[8]), float(row[9]), float(row[10])),
                gyroscope_bias=(float(row[11]), float(row[12]), float(row[13])),
                accelerometer_bias=(float(row[14]), float(row[15]), float(row[16])),
            )
            samples.append(sample)
    return samples


def read_imu_samples(data_csv_path: Path) -> list[ImuSample]:
    """Read IMU data from a data.csv file."""
    imu_samples = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timestamp_ns = int(row[0])
            angular_velocity = (float(row[1]), float(row[2]), float(row[3]))
            linear_acceleration = (float(row[4]), float(row[5]), float(row[6]))
            imu_samples.append(ImuSample(timestamp_ns, angular_velocity, linear_acceleration))
    return imu_samples


@dataclass
class DataFolder:
    path: Path
    cam_timestamps_ns: list[int]
    imu_samples: list[ImuSample]
    ground_truth_samples: list[GroundTruthSample]
    leica_samples: list[LeicaSample]
    cam0_extrinsics: np.ndarray  # 4x4 transformation matrix (T_BS)
    cam1_extrinsics: np.ndarray  # 4x4 transformation matrix (T_BS)
    cam0_intrinsics: CameraIntrinsics
    cam1_intrinsics: CameraIntrinsics

    @classmethod
    def load(cls, path: Path) -> Self:
        cam0_timestamps_ns = read_timestamps(path / "cam0" / "data.csv")
        cam1_timestamps_ns = read_timestamps(path / "cam1" / "data.csv")
        if cam0_timestamps_ns != cam1_timestamps_ns:
            raise ValueError("cam0 and cam1 timestamps do not match")
        imu_samples = read_imu_samples(path / "imu0" / "data.csv")
        ground_truth_samples = read_ground_truth_samples(path / "state_groundtruth_estimate0" / "data.csv")
        leica_samples = read_leica_samples(path / "leica0" / "data.csv")
        cam0_extrinsics = read_extrinsics(path / "cam0" / "sensor.yaml")
        cam1_extrinsics = read_extrinsics(path / "cam1" / "sensor.yaml")
        cam0_intrinsics = CameraIntrinsics.from_sensor_yaml(path / "cam0" / "sensor.yaml")
        cam1_intrinsics = CameraIntrinsics.from_sensor_yaml(path / "cam1" / "sensor.yaml")
        return cls(
            path=path,
            cam_timestamps_ns=cam0_timestamps_ns,
            imu_samples=imu_samples,
            ground_truth_samples=ground_truth_samples,
            leica_samples=leica_samples,
            cam0_extrinsics=cam0_extrinsics,
            cam1_extrinsics=cam1_extrinsics,
            cam0_intrinsics=cam0_intrinsics,
            cam1_intrinsics=cam1_intrinsics,
        )

    def get_cam0_image_path(self, timestamp_ns: int) -> Path:
        return self.path / "cam0" / "data" / f"{timestamp_ns}.png"

    def get_cam1_image_path(self, timestamp_ns: int) -> Path:
        return self.path / "cam1" / "data" / f"{timestamp_ns}.png"
