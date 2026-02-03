from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")

    sift = cv2.SIFT_create()

    # Load first two images and detect SIFT features
    img0 = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[1])), cv2.IMREAD_GRAYSCALE)

    kp0, desc0 = sift.detectAndCompute(img0, None)
    kp1, desc1 = sift.detectAndCompute(img1, None)

    print(f"Image 0: {len(kp0)} keypoints")
    print(f"Image 1: {len(kp1)} keypoints")

    # Match descriptors using BFMatcher with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc0, desc1, k=2)

    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    # Draw matches
    img_matches = cv2.drawMatches(
        img0, kp0, img1, kp1, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16, 8))
    plt.imshow(img_matches)
    plt.title(f"{len(good_matches)} matches")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
