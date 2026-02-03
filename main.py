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
    timestamps = read_timestamps(cam0_csv)

    print(f"Found {len(timestamps)} frames in cam0")
    print(f"First timestamp: {timestamps[0]} ns")
    print(f"Last timestamp: {timestamps[-1]} ns")

    # Show first cam0 image
    first_image_path = data_dir / "cam0" / "data" / f"{timestamps[0]}.png"
    img = mpimg.imread(first_image_path)
    plt.imshow(img, cmap="gray")
    plt.title(f"cam0 - {timestamps[0]}")
    plt.show()


if __name__ == "__main__":
    main()
