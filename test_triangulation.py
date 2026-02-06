from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam import DataFolder, triangulate_stereo_matches


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))

    orb = cv2.ORB_create(nfeatures=2000)

    timestamp_ns = data.cam_timestamps_ns[0]
    cam0_img = cv2.imread(str(data.get_cam0_image_path(timestamp_ns)), cv2.IMREAD_GRAYSCALE)
    cam1_img = cv2.imread(str(data.get_cam1_image_path(timestamp_ns)), cv2.IMREAD_GRAYSCALE)

    cam0_keypoints, cam0_descriptors = orb.detectAndCompute(cam0_img, None)
    cam1_keypoints, cam1_descriptors = orb.detectAndCompute(cam1_img, None)

    print(f"cam0: {len(cam0_keypoints)} keypoints")
    print(f"cam1: {len(cam1_keypoints)} keypoints")

    points_3d = triangulate_stereo_matches(
        data, cam0_keypoints, cam0_descriptors, cam1_keypoints, cam1_descriptors
    )

    print(f"Triangulated {points_3d.shape[1]} 3D points")
    print(f"X range: [{points_3d[0, :].min():.2f}, {points_3d[0, :].max():.2f}]")
    print(f"Y range: [{points_3d[1, :].min():.2f}, {points_3d[1, :].max():.2f}]")
    print(f"Z range: [{points_3d[2, :].min():.2f}, {points_3d[2, :].max():.2f}]")

    # Reproject 3D points back to cam0
    K0 = data.cam0_intrinsics.to_matrix()
    dist_coeffs0 = np.array([
        data.cam0_intrinsics.k1,
        data.cam0_intrinsics.k2,
        data.cam0_intrinsics.p1,
        data.cam0_intrinsics.p2,
    ])

    # Project 3D points (in cam0 frame) to 2D
    rvec = np.zeros(3)  # No rotation (points are already in cam0 frame)
    tvec = np.zeros(3)  # No translation
    projected_points, _ = cv2.projectPoints(
        points_3d.T, rvec, tvec, K0, dist_coeffs0
    )
    projected_points = projected_points.reshape(-1, 2)

    # Visualize keypoints and reprojected points side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: cam0_keypoints
    ax1.imshow(cam0_img, cmap='gray')
    kp_x = [kp.pt[0] for kp in cam0_keypoints]
    kp_y = [kp.pt[1] for kp in cam0_keypoints]
    ax1.scatter(kp_x, kp_y, c='red', s=20, alpha=0.5)
    ax1.set_title(f'cam0_keypoints ({len(cam0_keypoints)})')
    ax1.axis('off')

    # Right: reprojected points
    ax2.imshow(cam0_img, cmap='gray')
    ax2.scatter(projected_points[:, 0], projected_points[:, 1],
                c='green', s=20, alpha=0.5)
    ax2.set_title(f'Reprojected 3D Points ({len(projected_points)})')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    # Visualize 3D points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], s=1, alpha=0.5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Triangulated 3D Points')
    plt.show()


if __name__ == "__main__":
    main()
