import csv
from dataclasses import dataclass
from pathlib import Path

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
class DataFolder:
    path: Path
    cam_timestamps: list[int]

    @classmethod
    def load(cls, path: Path) -> "DataFolder":
        cam0_timestamps = read_timestamps(path / "cam0" / "data.csv")
        cam1_timestamps = read_timestamps(path / "cam1" / "data.csv")
        if cam0_timestamps != cam1_timestamps:
            raise ValueError("cam0 and cam1 timestamps do not match")
        return cls(path, cam0_timestamps)

    def get_cam0_image_path(self, timestamp: int) -> Path:
        return self.path / "cam0" / "data" / f"{timestamp}.png"

    def get_cam1_image_path(self, timestamp: int) -> Path:
        return self.path / "cam1" / "data" / f"{timestamp}.png"


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} frames")

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
