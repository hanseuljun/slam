from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")

    # Load images using OpenCV
    cam0_img = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)
    cam1_img = cv2.imread(str(data.get_cam1_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)

    # Run SIFT on both images
    sift = cv2.SIFT_create()
    kp0, desc0 = sift.detectAndCompute(cam0_img, None)
    kp1, desc1 = sift.detectAndCompute(cam1_img, None)

    print(f"cam0: {len(kp0)} keypoints")
    print(f"cam1: {len(kp1)} keypoints")

    # Draw keypoints on images
    cam0_with_kp = cv2.drawKeypoints(cam0_img, kp0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cam1_with_kp = cv2.drawKeypoints(cam1_img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show images side by side
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

    ax0.imshow(cam0_with_kp)
    ax0.set_title(f"cam0 - {len(kp0)} keypoints")

    ax1.imshow(cam1_with_kp)
    ax1.set_title(f"cam1 - {len(kp1)} keypoints")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
