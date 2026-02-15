from pathlib import Path

import cv2

from slam.data import DataFolder

MAV0_DIR = Path("data/machine_hall/MH_01_easy/mav0")
OUTPUT_PATH = Path("tmp/MH_01_easy.mp4")


def images_to_video(data_folder: DataFolder, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(data_folder.get_cam0_image_path(data_folder.cam_timestamps_ns[0])))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, data_folder.cam0_rate_hz, (width, height))

    for i, timestamp_ns in enumerate(data_folder.cam_timestamps_ns):
        frame = cv2.imread(str(data_folder.get_cam0_image_path(timestamp_ns)))
        writer.write(frame)
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(data_folder.cam_timestamps_ns)} frames")

    writer.release()
    print(f"Saved video to {output_path} ({len(data_folder.cam_timestamps_ns)} frames, {data_folder.cam0_rate_hz} fps)")


if __name__ == "__main__":
    data_folder = DataFolder.load(MAV0_DIR)
    images_to_video(data_folder, OUTPUT_PATH)
