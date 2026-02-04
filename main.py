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

    keypoints0, descriptors0 = sift.detectAndCompute(img0, None)
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)

    print(f"Image 0: {len(keypoints0)} keypoints")
    print(f"Image 1: {len(keypoints1)} keypoints")

    # Match descriptors using BFMatcher with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors0, descriptors1, k=2)

    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    # Get first match and compute normalized coordinates
    first_match = good_matches[0]
    keypoint0 = keypoints0[first_match.queryIdx]
    keypoint1 = keypoints1[first_match.trainIdx]

    print(f"\nFirst match (DMatch):")
    print(f"  queryIdx: {first_match.queryIdx}")
    print(f"  trainIdx: {first_match.trainIdx}")
    print(f"  distance: {first_match.distance:.2f}")

    print(f"\nKeypoint 0:")
    print(f"  pt: ({keypoint0.pt[0]:.2f}, {keypoint0.pt[1]:.2f})")
    print(f"  size: {keypoint0.size:.2f}")
    print(f"  angle: {keypoint0.angle:.2f}")
    print(f"  response: {keypoint0.response:.4f}")
    print(f"  octave: {keypoint0.octave}")

    print(f"\nKeypoint 1:")
    print(f"  pt: ({keypoint1.pt[0]:.2f}, {keypoint1.pt[1]:.2f})")
    print(f"  size: {keypoint1.size:.2f}")
    print(f"  angle: {keypoint1.angle:.2f}")
    print(f"  response: {keypoint1.response:.4f}")
    print(f"  octave: {keypoint1.octave}")

    # Apply inverse intrinsic matrix to get normalized coordinates
    K = data.cam0_intrinsics.to_matrix()
    K_inv = np.linalg.inv(K)

    pt0_homog = np.array([keypoint0.pt[0], keypoint0.pt[1], 1.0])
    pt1_homog = np.array([keypoint1.pt[0], keypoint1.pt[1], 1.0])

    pt0_norm = K_inv @ pt0_homog
    pt1_norm = K_inv @ pt1_homog

    print(f"  Image 0 normalized: ({pt0_norm[0]:.4f}, {pt0_norm[1]:.4f})")
    print(f"  Image 1 normalized: ({pt1_norm[0]:.4f}, {pt1_norm[1]:.4f})")

    # Extract all matched points
    points0 = np.array([keypoints0[m.queryIdx].pt for m in good_matches])
    points1 = np.array([keypoints1[m.trainIdx].pt for m in good_matches])

    # Find essential matrix
    E, mask = cv2.findEssentialMat(points0, points1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    inliers = mask.ravel().sum()
    print(f"\nEssential Matrix:")
    print(f"  Inliers: {inliers} / {len(good_matches)}")
    print(f"  E:\n{E}")

    # Draw matches
    img_matches = cv2.drawMatches(
        img0, keypoints0, img1, keypoints1, good_matches, None,
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
