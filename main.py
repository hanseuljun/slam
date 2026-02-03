from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")

    sift = cv2.SIFT_create()

    # Show first 2 images from cam0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, ax in enumerate(axes):
        img = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[i])), cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(img, None)

        print(f"Image {i}:")
        print(f"  Keypoints: {len(keypoints)}")
        print(f"  Descriptors shape: {descriptors.shape}")
        print(f"  Descriptors dtype: {descriptors.dtype}")
        print(f"  Descriptors range: [{descriptors.min():.1f}, {descriptors.max():.1f}]")

        img_with_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ax.imshow(img_with_kp)
        ax.set_title(f"{len(keypoints)} kp")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
