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

    # Load first frame from left and right cameras
    img0 = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(data.get_cam1_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)

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
    K0 = data.cam0_intrinsics.to_matrix()
    K0_inv = np.linalg.inv(K0)
    K1 = data.cam1_intrinsics.to_matrix()
    K1_inv = np.linalg.inv(K1)

    cam0_extrinsics = data.cam0_extrinsics
    cam1_extrinsics = data.cam1_extrinsics
    cam0_to_cam1 = np.linalg.inv(cam0_extrinsics) @ cam1_extrinsics

    cam0_translation = cam0_extrinsics[:3, 3]
    cam1_translation = cam1_extrinsics[:3, 3]

    print(f"\nCam0 Extrinsics:\n{cam0_extrinsics}")
    print(f"\nCam1 Extrinsics:\n{cam1_extrinsics}")
    print(f"\nCam0 to Cam1:\n{cam0_to_cam1}")
    print(f"\nCam0 Translation: {cam0_translation}")
    print(f"Cam1 Translation: {cam1_translation}")

    pt0_homog = np.array([keypoint0.pt[0], keypoint0.pt[1], 1.0])
    pt1_homog = np.array([keypoint1.pt[0], keypoint1.pt[1], 1.0])

    pt0_norm = K0_inv @ pt0_homog
    pt1_norm = K1_inv @ pt1_homog

    print(f"  Image 0 normalized: ({pt0_norm[0]:.4f}, {pt0_norm[1]:.4f})")
    print(f"  Image 1 normalized: ({pt1_norm[0]:.4f}, {pt1_norm[1]:.4f})")

    # Extract all matched points
    points0 = np.array([keypoints0[m.queryIdx].pt for m in good_matches])
    points1 = np.array([keypoints1[m.trainIdx].pt for m in good_matches])

    # Build projection matrices for triangulation
    # cam0 is the reference frame, so P0 = K0 @ [I | 0]
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # P1 = K1 @ [R | t] where transformation is from cam0 to cam1
    T_cam0_to_cam1 = np.linalg.inv(cam1_extrinsics) @ cam0_extrinsics
    P1 = K1 @ T_cam0_to_cam1[:3, :]

    # Triangulate points (cv2.triangulatePoints expects 2xN arrays)
    points_4d = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3, :] / points_4d[3, :]

    print(f"\nTriangulated {points_3d.shape[1]} points")
    print(f"First 5 3D points (in cam0 frame):")
    for i in range(min(5, points_3d.shape[1])):
        print(f"  Point {i}: ({points_3d[0, i]:.4f}, {points_3d[1, i]:.4f}, {points_3d[2, i]:.4f})")

    # Draw keypoints on both images
    img0_with_kp = cv2.drawKeypoints(img0, keypoints0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img1_with_kp = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    ax0.imshow(img0_with_kp)
    ax0.set_title(f"Image 0: {len(keypoints0)} keypoints")
    ax0.axis("off")
    ax1.imshow(img1_with_kp)
    ax1.set_title(f"Image 1: {len(keypoints1)} keypoints")
    ax1.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
