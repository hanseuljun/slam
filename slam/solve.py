import time

import cv2
import numpy as np

from slam.data import DataFolder


def triangulate_stereo_matches(
    data: DataFolder,
    cam0_keypoints,
    cam0_descriptors: np.ndarray,
    cam1_keypoints,
    cam1_descriptors: np.ndarray,
) -> np.ndarray:
    # Match descriptors using BFMatcher with ratio test
    # Use NORM_HAMMING for binary descriptors (ORB, BRIEF, BRISK)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(cam0_descriptors, cam1_descriptors, k=2)

    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    K0 = data.cam0_intrinsics.to_matrix()
    K1 = data.cam1_intrinsics.to_matrix()
    dist_coeffs0 = np.array([
        data.cam0_intrinsics.k1,
        data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1,
        data.cam0_intrinsics.p2,
    ])
    dist_coeffs1 = np.array([
        data.cam1_intrinsics.k1,
        data.cam1_intrinsics.k2,
        data.cam1_intrinsics.p1,
        data.cam1_intrinsics.p2,
    ])

    # Extract all matched points
    points0 = np.array([cam0_keypoints[m.queryIdx].pt for m in good_matches])
    points1 = np.array([cam1_keypoints[m.trainIdx].pt for m in good_matches])

    # Undistort points (returns undistorted points in pixel coordinates)
    points0 = cv2.undistortPoints(points0, K0, dist_coeffs0, P=K0).reshape(-1, 2)
    points1 = cv2.undistortPoints(points1, K1, dist_coeffs1, P=K1).reshape(-1, 2)

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
    # Use NORM_HAMMING for binary descriptors (ORB, BRIEF, BRISK)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
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
) -> tuple[np.ndarray, np.ndarray]:
    t_start = time.perf_counter()

    # Load first frame from left and right cameras, and the second frame from left camera
    cam0_img0 = cv2.imread(str(data.get_cam0_image_path(timestamp0_ns)), cv2.IMREAD_GRAYSCALE)
    cam1_img0 = cv2.imread(str(data.get_cam1_image_path(timestamp0_ns)), cv2.IMREAD_GRAYSCALE)
    cam0_img1 = cv2.imread(str(data.get_cam0_image_path(timestamp1_ns)), cv2.IMREAD_GRAYSCALE)
    t_imread = time.perf_counter()

    cam0_keypoints0, cam0_descriptors0 = sift.detectAndCompute(cam0_img0, None)
    cam1_keypoints0, cam1_descriptors0 = sift.detectAndCompute(cam1_img0, None)
    cam0_keypoints1, cam0_descriptors1 = sift.detectAndCompute(cam0_img1, None)
    t_sift = time.perf_counter()

    points_3d = triangulate_stereo_matches(
        data, cam0_keypoints0, cam0_descriptors0, cam1_keypoints0, cam1_descriptors0
    )
    t_triangulate = time.perf_counter()

    T = solve_pnp(
        data, points_3d, cam0_descriptors0,
        cam1_descriptors0, cam0_keypoints1, cam0_descriptors1
    )
    t_pnp = time.perf_counter()

    print(f"solve_step: {(t_pnp-t_start)*1000:.2f}ms "
          f"(imread: {(t_imread-t_start)*1000:.2f}ms, "
          f"sift: {(t_sift-t_imread)*1000:.2f}ms, "
          f"triangulate: {(t_triangulate-t_sift)*1000:.2f}ms, "
          f"solve_pnp: {(t_pnp-t_triangulate)*1000:.2f}ms)")

    return T, points_3d
