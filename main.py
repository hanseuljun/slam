import csv
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


def main():
    data_dir = Path("data/machine_hall/MH_01_easy/mav0")

    # Read cam0 timestamps
    cam0_csv = data_dir / "cam0" / "data.csv"
    cam0_timestamps = read_timestamps(cam0_csv)
    print(f"Found {len(cam0_timestamps)} frames in cam0")

    # Read cam1 timestamps
    cam1_csv = data_dir / "cam1" / "data.csv"
    cam1_timestamps = read_timestamps(cam1_csv)
    print(f"Found {len(cam1_timestamps)} frames in cam1")

    # Show first images from cam0 and cam1 side by side
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    cam0_image_path = data_dir / "cam0" / "data" / f"{cam0_timestamps[0]}.png"
    cam0_img = mpimg.imread(cam0_image_path)
    ax0.imshow(cam0_img, cmap="gray")
    ax0.set_title(f"cam0 - {cam0_timestamps[0]}")

    cam1_image_path = data_dir / "cam1" / "data" / f"{cam1_timestamps[0]}.png"
    cam1_img = mpimg.imread(cam1_image_path)
    ax1.imshow(cam1_img, cmap="gray")
    ax1.set_title(f"cam1 - {cam1_timestamps[0]}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
