from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam import DataFolder


def visualize_3d_points(points_3d: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], c='b', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangulated 3D Points')
    plt.show()


def visualize_keypoints(img0: np.ndarray, keypoints0, img1: np.ndarray, keypoints1) -> None:
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


def triangulate_stereo_matches(
    data: DataFolder,
    cam0_keypoints,
    cam0_descriptors: np.ndarray,
    cam1_keypoints,
    cam1_descriptors: np.ndarray,
) -> np.ndarray:
    # Match descriptors using BFMatcher with ratio test
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(cam0_descriptors, cam1_descriptors, k=2)

    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    # Apply inverse intrinsic matrix to get normalized coordinates
    K0 = data.cam0_intrinsics.to_matrix()
    K1 = data.cam1_intrinsics.to_matrix()

    print(f"\nCam0 Extrinsics:\n{data.cam0_extrinsics}")
    print(f"\nCam1 Extrinsics:\n{data.cam1_extrinsics}")

    # Extract all matched points
    points0 = np.array([cam0_keypoints[m.queryIdx].pt for m in good_matches])
    points1 = np.array([cam1_keypoints[m.trainIdx].pt for m in good_matches])

    # Build projection matrices for triangulation
    # cam0 is the reference frame, so P0 = K0 @ [I | 0]
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # P1 = K1 @ [R | t] where transformation is from cam0 to cam1
    T_cam0_to_cam1 = np.linalg.inv(data.cam1_extrinsics) @ data.cam0_extrinsics
    P1 = K1 @ T_cam0_to_cam1[:3, :]

    print(f"\nT_cam0_to_cam1:\n{T_cam0_to_cam1}")
    print(f"\nCam0 Projection Matrix (P0):\n{P0}")
    print(f"\nCam1 Projection Matrix (P1):\n{P1}")

    # Triangulate points (cv2.triangulatePoints expects 2xN arrays)
    points_4d = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3, :] / points_4d[3, :]

    return points_3d


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")
    print(f"Cam0 distortion: k1={data.cam0_intrinsics.k1}, k2={data.cam0_intrinsics.k2}, p1={data.cam0_intrinsics.p1}, p2={data.cam0_intrinsics.p2}")
    print(f"Cam1 distortion: k1={data.cam1_intrinsics.k1}, k2={data.cam1_intrinsics.k2}, p1={data.cam1_intrinsics.p1}, p2={data.cam1_intrinsics.p2}")

    sift = cv2.SIFT_create()

    # Load first frame from left and right cameras
    cam0_img0 = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)
    cam1_img0 = cv2.imread(str(data.get_cam1_image_path(data.cam_timestamps[0])), cv2.IMREAD_GRAYSCALE)

    cam0_keypoints0, cam0_descriptors0 = sift.detectAndCompute(cam0_img0, None)
    cam1_keypoints0, cam1_descriptors0 = sift.detectAndCompute(cam1_img0, None)

    # Load second frame from left camera
    cam0_img1 = cv2.imread(str(data.get_cam0_image_path(data.cam_timestamps[1])), cv2.IMREAD_GRAYSCALE)
    cam0_keypoints1, cam0_descriptors1 = sift.detectAndCompute(cam0_img1, None)

    print(f"cam0_img0: {len(cam0_keypoints0)} keypoints")
    print(f"cam1_img0: {len(cam1_keypoints0)} keypoints")
    print(f"cam0_img1: {len(cam0_keypoints1)} keypoints")

    points_3d = triangulate_stereo_matches(
        data, cam0_keypoints0, cam0_descriptors0, cam1_keypoints0, cam1_descriptors0
    )

    print(f"\nTriangulated {points_3d.shape[1]} points")
    print(f"First 5 3D points (in cam0 frame):")
    for i in range(min(5, points_3d.shape[1])):
        print(f"  Point {i}: ({points_3d[0, i]:.4f}, {points_3d[1, i]:.4f}, {points_3d[2, i]:.4f})")

    # Compute and print average of all 3D points
    avg_point = np.mean(points_3d, axis=1)
    print(f"\nAverage 3D point: ({avg_point[0]:.4f}, {avg_point[1]:.4f}, {avg_point[2]:.4f})")

    # Get stereo matches again to know which cam0_keypoints0 indices have 3D points
    bf = cv2.BFMatcher()
    stereo_matches = bf.knnMatch(cam0_descriptors0, cam1_descriptors0, k=2)
    stereo_good_matches = []
    for m, n in stereo_matches:
        if m.distance < 0.75 * n.distance:
            stereo_good_matches.append(m)

    # Map from cam0_keypoints0 index to 3D point index
    cam0_idx_to_3d_idx = {m.queryIdx: i for i, m in enumerate(stereo_good_matches)}

    # Match cam0_img0 with cam0_img1 (temporal matching)
    temporal_matches = bf.knnMatch(cam0_descriptors0, cam0_descriptors1, k=2)
    temporal_good_matches = []
    for m, n in temporal_matches:
        if m.distance < 0.75 * n.distance:
            temporal_good_matches.append(m)

    print(f"\nTemporal matches (cam0_img0 -> cam0_img1): {len(temporal_good_matches)}")

    # Find correspondences: cam0_keypoints0 that have both 3D points AND matches in cam0_img1
    object_points = []
    image_points = []
    for m in temporal_good_matches:
        if m.queryIdx in cam0_idx_to_3d_idx:
            idx_3d = cam0_idx_to_3d_idx[m.queryIdx]
            object_points.append(points_3d[:, idx_3d])
            image_points.append(cam0_keypoints1[m.trainIdx].pt)

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    print(f"Points for PnP: {len(object_points)}")

    # Run solvePnP
    K0 = data.cam0_intrinsics.to_matrix()
    dist_coeffs = np.array([
        data.cam0_intrinsics.k1,
        data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1,
        data.cam0_intrinsics.p2,
    ])

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K0, dist_coeffs)
    print(f"\nsolvePnP success: {success}")
    print(f"Rotation vector:\n{rvec}")
    print(f"Translation vector:\n{tvec}")

    # Convert rvec and tvec to 4x4 transformation matrix
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    print(f"\nTransformation matrix (cam0_img0 -> cam0_img1):\n{T}")

    visualize_3d_points(points_3d)
    visualize_keypoints(cam0_img0, cam0_keypoints0, cam1_img0, cam1_keypoints0)


if __name__ == "__main__":
    main()
