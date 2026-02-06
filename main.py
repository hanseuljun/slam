from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam import DataFolder, solve_step


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    orb = cv2.ORB_create(nfeatures=2000)

    min_timestamp_ns = data.cam_timestamps_ns[0]
    max_timestamp_ns = min_timestamp_ns + int(5e9)  # 5 seconds

    first_leica_pos = data.leica_samples[0].position
    T_world_to_leica = np.eye(4)
    T_world_to_leica[:3, 3] = first_leica_pos
    initial_cam0_transform = T_world_to_leica @ np.linalg.inv(data.leica_extrinsics) @ data.cam0_extrinsics

    keyframe_index = 0
    cam0_transforms = [initial_cam0_transform]
    viz_points_3d = None
    i = 0
    while data.cam_timestamps_ns[i + 1] <= max_timestamp_ns:
        T, points_3d = solve_step(data,
                                  orb,
                                  data.cam_timestamps_ns[keyframe_index],
                                  data.cam_timestamps_ns[i + 1])
        cam0_transforms.append(cam0_transforms[keyframe_index] @ T)
        if i == 10:
            viz_points_3d = points_3d
        i += 1

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

    # Extract ground truth positions (within time range)
    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_timestamp_ns]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in gt_samples])

    # Extract leica positions (within time range)
    leica_samples = [s for s in data.leica_samples if s.timestamp_ns <= max_timestamp_ns]
    leica_positions = np.array([s.position for s in leica_samples])
    leica_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in leica_samples])

    # Plot cam0_transforms vs ground truth vs leica
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax1.plot(cam0_times, cam0_positions[:, 0], label='cam0 x')
    ax1.plot(gt_times, gt_positions[:, 0], label='gt x')
    ax1.plot(leica_times, leica_positions[:, 0], label='leica x')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X [m]')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(cam0_times, cam0_positions[:, 1], label='cam0 y')
    ax2.plot(gt_times, gt_positions[:, 1], label='gt y')
    ax2.plot(leica_times, leica_positions[:, 1], label='leica y')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(cam0_times, cam0_positions[:, 2], label='cam0 z')
    ax3.plot(gt_times, gt_positions[:, 2], label='gt z')
    ax3.plot(leica_times, leica_positions[:, 2], label='leica z')
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
