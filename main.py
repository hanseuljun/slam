import csv
from pathlib import Path


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
    print(f"\nFirst 10 timestamps:")
    for ts in timestamps[:10]:
        print(f"  {ts}")


if __name__ == "__main__":
    main()
