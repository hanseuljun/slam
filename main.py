from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam import DataFolder


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

    # Apply inverse intrinsic matrix to get normalized coordinates
    K0 = data.cam0_intrinsics.to_matrix()
    K1 = data.cam1_intrinsics.to_matrix()

    # Extract all matched points
    points0 = np.array([cam0_keypoints[m.queryIdx].pt for m in good_matches])
    points1 = np.array([cam1_keypoints[m.trainIdx].pt for m in good_matches])

    # Build projection matrices for triangulation
    # cam0 is the reference frame, so P0 = K0 @ [I | 0]
    P0 = K0 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # P1 = K1 @ [R | t] where transformation is from cam0 to cam1
    T_cam0_to_cam1 = np.linalg.inv(data.cam1_extrinsics) @ data.cam0_extrinsics
    P1 = K1 @ T_cam0_to_cam1[:3, :]

    # Triangulate points (cv2.triangulatePoints expects 2xN arrays)
    points_4d = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3, :] / points_4d[3, :]

    return points_3d


def solve_pnp(
    data: DataFolder,
    points_3d: np.ndarray,
    cam0_descriptors0: np.ndarray,
    cam1_descriptors0: np.ndarray,
    cam0_keypoints1,
    cam0_descriptors1: np.ndarray,
) -> np.ndarray:
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

    # Run solvePnP
    K0 = data.cam0_intrinsics.to_matrix()
    dist_coeffs = np.array([
        data.cam0_intrinsics.k1,
        data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1,
        data.cam0_intrinsics.p2,
    ])

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, K0, dist_coeffs)
    if not success:
        raise RuntimeError("cv2.solvePnP failed")

    # Convert rvec and tvec to 4x4 transformation matrix
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    return T


def solve_step(
    data: DataFolder,
    sift,
    timestamp0_ns: int,
    timestamp1_ns: int,
) -> np.ndarray:
    # Load first frame from left and right cameras
    cam0_img0 = cv2.imread(str(data.get_cam0_image_path(timestamp0_ns)), cv2.IMREAD_GRAYSCALE)
    cam1_img0 = cv2.imread(str(data.get_cam1_image_path(timestamp0_ns)), cv2.IMREAD_GRAYSCALE)

    cam0_keypoints0, cam0_descriptors0 = sift.detectAndCompute(cam0_img0, None)
    cam1_keypoints0, cam1_descriptors0 = sift.detectAndCompute(cam1_img0, None)

    # Load second frame from left camera
    cam0_img1 = cv2.imread(str(data.get_cam0_image_path(timestamp1_ns)), cv2.IMREAD_GRAYSCALE)
    cam0_keypoints1, cam0_descriptors1 = sift.detectAndCompute(cam0_img1, None)

    points_3d = triangulate_stereo_matches(
        data, cam0_keypoints0, cam0_descriptors0, cam1_keypoints0, cam1_descriptors0
    )

    T = solve_pnp(
        data, points_3d, cam0_descriptors0,
        cam1_descriptors0, cam0_keypoints1, cam0_descriptors1
    )

    return T


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps_ns)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")
    print(f"Cam0 distortion: k1={data.cam0_intrinsics.k1}, k2={data.cam0_intrinsics.k2}, p1={data.cam0_intrinsics.p1}, p2={data.cam0_intrinsics.p2}")
    print(f"Cam1 distortion: k1={data.cam1_intrinsics.k1}, k2={data.cam1_intrinsics.k2}, p1={data.cam1_intrinsics.p1}, p2={data.cam1_intrinsics.p2}")

    sift = cv2.SIFT_create()

    keyframe_index = 0
    cam0_transforms = [data.cam0_extrinsics]
    for i in range(50):
        T = solve_step(data,
                       sift,
                       data.cam_timestamps_ns[keyframe_index],
                       data.cam_timestamps_ns[i + 1])
        cam0_transforms.append(cam0_transforms[keyframe_index] @ T)

    min_timestamp_ns = data.cam_timestamps_ns[0]
    for i, T in enumerate(cam0_transforms):
        t_seconds = (data.cam_timestamps_ns[i] - min_timestamp_ns) / 1e9
        print(f"\ncam0_transforms[{i}] (t={t_seconds:.3f}s):\n{T}")

    print("\nFirst 10 ground truth samples:")
    for i, sample in enumerate(data.ground_truth_samples[:10]):
        t_seconds = (sample.timestamp_ns - min_timestamp_ns) / 1e9
        print(f"  [{i}] t={t_seconds:.3f}s, pos={sample.position}, quat={sample.quaternion}")

    # Extract translations from cam0_transforms
    cam0_positions = np.array([T[:3, 3] for T in cam0_transforms])
    cam0_times = np.array([(data.cam_timestamps_ns[i] - min_timestamp_ns) / 1e9
                           for i in range(len(cam0_transforms))])

    # Extract ground truth positions
    gt_positions = np.array([sample.position for sample in data.ground_truth_samples[:200]])
    gt_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9
                         for s in data.ground_truth_samples[:len(gt_positions)]])

    # Plot cam0_transforms vs ground truth
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax1.plot(cam0_times, cam0_positions[:, 0], label='cam0 x')
    ax1.plot(gt_times, gt_positions[:, 0], label='gt x')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X [m]')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(cam0_times, cam0_positions[:, 1], label='cam0 y')
    ax2.plot(gt_times, gt_positions[:, 1], label='gt y')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(cam0_times, cam0_positions[:, 2], label='cam0 z')
    ax3.plot(gt_times, gt_positions[:, 2], label='gt z')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Z [m]')
    ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
