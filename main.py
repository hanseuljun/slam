from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam import DataFolder, solve_step


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    print(f"Found {len(data.cam_timestamps_ns)} camera frames")
    print(f"Found {len(data.imu_samples)} IMU samples")
    print(f"Cam0 distortion: k1={data.cam0_intrinsics.k1}, k2={data.cam0_intrinsics.k2}, p1={data.cam0_intrinsics.p1}, p2={data.cam0_intrinsics.p2}")
    print(f"Cam1 distortion: k1={data.cam1_intrinsics.k1}, k2={data.cam1_intrinsics.k2}, p1={data.cam1_intrinsics.p1}, p2={data.cam1_intrinsics.p2}")
    print(f"Leica extrinsics:\n{data.leica_extrinsics}")
    print(f"Found {len(data.leica_samples)} Leica samples")
    for i, sample in enumerate(data.leica_samples[:5]):
        print(f"  [{i}] t={sample.timestamp_ns}, pos={sample.position}")

    orb = cv2.ORB_create(nfeatures=2000)

    keyframe_index = 0
    cam0_transforms = [data.cam0_extrinsics]
    viz_points_3d = None
    for i in range(100):
        T, points_3d = solve_step(data,
                                  orb,
                                  data.cam_timestamps_ns[keyframe_index],
                                  data.cam_timestamps_ns[i + 1])
        cam0_transforms.append(cam0_transforms[keyframe_index] @ T)
        if i == 10:
            viz_points_3d = points_3d

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
    gt_positions = np.array([sample.position for sample in data.ground_truth_samples[:400]])
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

    # Visualize 3D points
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(viz_points_3d[0, :], viz_points_3d[1, :], viz_points_3d[2, :], s=1, alpha=0.5)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Triangulated 3D Points')
    plt.show()


if __name__ == "__main__":
    main()
