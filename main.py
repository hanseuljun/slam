from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

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

    # Get first match and compute normalized coordinates
    first_match = good_matches[0]
    pt0 = kp0[first_match.queryIdx].pt  # (u, v) in image 0
    pt1 = kp1[first_match.trainIdx].pt  # (u, v) in image 1

    print(f"\nFirst match:")
    print(f"  Image 0 pixel: ({pt0[0]:.2f}, {pt0[1]:.2f})")
    print(f"  Image 1 pixel: ({pt1[0]:.2f}, {pt1[1]:.2f})")

    # Apply inverse intrinsic matrix to get normalized coordinates
    K = data.cam0_intrinsics.to_matrix()
    K_inv = np.linalg.inv(K)

    pt0_homog = np.array([pt0[0], pt0[1], 1.0])
    pt1_homog = np.array([pt1[0], pt1[1], 1.0])

    pt0_norm = K_inv @ pt0_homog
    pt1_norm = K_inv @ pt1_homog

    print(f"  Image 0 normalized: ({pt0_norm[0]:.4f}, {pt0_norm[1]:.4f})")
    print(f"  Image 1 normalized: ({pt1_norm[0]:.4f}, {pt1_norm[1]:.4f})")

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
