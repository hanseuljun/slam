from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")

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
