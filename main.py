import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
class ImuData:
    timestamp: int
    angular_velocity: tuple[float, float, float]  # w_x, w_y, w_z [rad/s]
    linear_acceleration: tuple[float, float, float]  # a_x, a_y, a_z [m/s^2]


def read_imu_data(data_csv_path: Path) -> list[ImuData]:
    """Read IMU data from a data.csv file."""
    imu_data = []
    with open(data_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            timestamp = int(row[0])
            angular_velocity = (float(row[1]), float(row[2]), float(row[3]))
            linear_acceleration = (float(row[4]), float(row[5]), float(row[6]))
            imu_data.append(ImuData(timestamp, angular_velocity, linear_acceleration))
    return imu_data


@dataclass
class DataFolder:
    path: Path
    cam_timestamps: list[int]
    imu_data: list[ImuData]

    @classmethod
    def load(cls, path: Path) -> Self:
        cam0_timestamps = read_timestamps(path / "cam0" / "data.csv")
        cam1_timestamps = read_timestamps(path / "cam1" / "data.csv")
        if cam0_timestamps != cam1_timestamps:
            raise ValueError("cam0 and cam1 timestamps do not match")
        imu_data = read_imu_data(path / "imu0" / "data.csv")
        return cls(path, cam0_timestamps, imu_data)

    def get_cam0_image_path(self, timestamp: int) -> Path:
        return self.path / "cam0" / "data" / f"{timestamp}.png"

    def get_cam1_image_path(self, timestamp: int) -> Path:
        return self.path / "cam1" / "data" / f"{timestamp}.png"


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} camera frames")
    print(f"Found {len(data.imu_data)} IMU samples")

    # Show first images from cam0 and cam1 side by side
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    cam0_img = mpimg.imread(data.get_cam0_image_path(data.cam_timestamps[0]))
    ax0.imshow(cam0_img, cmap="gray")
    ax0.set_title(f"cam0 - {data.cam_timestamps[0]}")

    cam1_img = mpimg.imread(data.get_cam1_image_path(data.cam_timestamps[0]))
    ax1.imshow(cam1_img, cmap="gray")
    ax1.set_title(f"cam1 - {data.cam_timestamps[0]}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
