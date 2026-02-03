import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self


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

    @classmethod
    def load(cls, path: Path) -> Self:
        cam0_timestamps = read_timestamps(path / "cam0" / "data.csv")
        cam1_timestamps = read_timestamps(path / "cam1" / "data.csv")
        if cam0_timestamps != cam1_timestamps:
            raise ValueError("cam0 and cam1 timestamps do not match")
        imu_samples = read_imu_samples(path / "imu0" / "data.csv")
        return cls(path, cam0_timestamps, imu_samples)

    def get_cam0_image_path(self, timestamp: int) -> Path:
        return self.path / "cam0" / "data" / f"{timestamp}.png"

    def get_cam1_image_path(self, timestamp: int) -> Path:
        return self.path / "cam1" / "data" / f"{timestamp}.png"
