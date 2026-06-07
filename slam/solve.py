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
    stereo_matches: list,
    cam0_descriptors0: np.ndarray,
    cam0_keypoints1,
    cam0_descriptors1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    cam0_idx_to_3d_idx = {m.queryIdx: i for i, m in enumerate(stereo_matches)}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
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

    success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, K0, dist_coeffs)
    if not success:
        raise RuntimeError(f"cv2.solvePnPRansac failed. len(object_points): {len(object_points)}")

    # Compute mean reprojection error over inliers
    inlier_object_points = object_points[inliers.flatten()]
    inlier_image_points = image_points[inliers.flatten()]
    projected, _ = cv2.projectPoints(inlier_object_points, rvec, tvec, K0, dist_coeffs)
    reprojection_error = np.mean(np.linalg.norm(inlier_image_points - projected.reshape(-1, 2), axis=1))

    return rvec, tvec, len(temporal_good_matches), reprojection_error


def solve_stereo_pnp(
    data: DataFolder,
    cam0_descriptors0: np.ndarray,
    stereo_matches: list,
    points_3d: np.ndarray,
    cam0_keypoints1,
    cam0_descriptors1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    return solve_pnp(
        data, points_3d, stereo_matches,
        cam0_descriptors0, cam0_keypoints1, cam0_descriptors1
    )
