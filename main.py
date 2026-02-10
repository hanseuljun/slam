from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from slam import DataFolder, solve_step


def quaternion_to_rotation_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    orb = cv2.ORB_create(nfeatures=2000)

    min_timestamp_ns = data.cam_timestamps_ns[0]
    max_timestamp_ns = min_timestamp_ns + int(5e9)  # 5 seconds

    keyframe_index = 0
    estimated_transforms_in_cam0 = [np.eye(4)]
    viz_points_3d = None
    i = 0
    while data.cam_timestamps_ns[i + 1] <= max_timestamp_ns:
        T, points_3d = solve_step(data,
                                  orb,
                                  data.cam_timestamps_ns[keyframe_index],
                                  data.cam_timestamps_ns[i + 1])
        estimated_transforms_in_cam0.append(estimated_transforms_in_cam0[keyframe_index] @ T)
        if i == 10:
            viz_points_3d = points_3d
        i += 1

    # Get first ground truth sample as 4x4 transformation matrix
    first_gt = data.ground_truth_samples[0]
    first_gt_transform = np.eye(4)
    first_gt_transform[:3, :3] = quaternion_to_rotation_matrix(first_gt.quaternion)
    first_gt_transform[:3, 3] = first_gt.position
    print(f"First ground truth transform:\n{first_gt_transform}")

    # Find camera timestamp closest to first ground truth timestamp
    cam_timestamps = np.array(data.cam_timestamps_ns)
    closest_cam_index = np.argmin(np.abs(cam_timestamps - first_gt.timestamp_ns))
    print(f"First GT timestamp: {first_gt.timestamp_ns}")
    print(f"Closest cam timestamp: {data.cam_timestamps_ns[closest_cam_index]} (index {closest_cam_index})")
    print(f"Time diff: {(data.cam_timestamps_ns[closest_cam_index] - first_gt.timestamp_ns) / 1e6:.2f} ms")

    # Compute T_cam0_to_world matrix: first_gt_transform @ inv(leica_extrinsics) @ cam0_extrinsics = T_cam0_to_world @ estimated_transforms_in_cam0[closest_cam_index]
    T_cam0_to_world = first_gt_transform @ np.linalg.inv(data.leica_extrinsics) @ data.cam0_extrinsics @ np.linalg.inv(estimated_transforms_in_cam0[closest_cam_index])
    print(f"\T_cam0_to_world:\n{T_cam0_to_world}")

    # Transform estimated poses to world frame
    estimated_transforms_in_world = [T_cam0_to_world @ T for T in estimated_transforms_in_cam0]

    # for i, T in enumerate(estimated_transforms_in_world):
    #     t_seconds = (data.cam_timestamps_ns[i] - min_timestamp_ns) / 1e9
    #     print(f"\nestimated_transforms_in_world[{i}] (t={t_seconds:.3f}s):\n{T}")

    # print("\nFirst 10 ground truth samples:")
    # for i, sample in enumerate(data.ground_truth_samples[:10]):
    #     t_seconds = (sample.timestamp_ns - min_timestamp_ns) / 1e9
    #     print(f"  [{i}] t={t_seconds:.3f}s, pos={sample.position}, quat={sample.quaternion}")

    # Extract translations from estimated_transforms_in_world
    world_positions = np.array([T[:3, 3] for T in estimated_transforms_in_world])
    world_times = np.array([(data.cam_timestamps_ns[i] - min_timestamp_ns) / 1e9
                            for i in range(len(estimated_transforms_in_world))])

    # Extract ground truth positions (within time range)
    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_timestamp_ns]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in gt_samples])

    # Extract forward directions (z-axis of rotation) from estimated transforms
    estimated_forward = np.array([T[:3, 2] for T in estimated_transforms_in_world])
    # Extract up directions (y-axis of rotation) from estimated transforms
    estimated_up = np.array([T[:3, 1] for T in estimated_transforms_in_world])

    # Extract forward directions from ground truth quaternions
    gt_forward = np.array([quaternion_to_rotation_matrix(s.quaternion)[:, 2] for s in gt_samples])
    # Extract up directions from ground truth quaternions
    gt_up = np.array([quaternion_to_rotation_matrix(s.quaternion)[:, 1] for s in gt_samples])

    # Plot forward directions
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle('Forward Direction (z-axis)')

    ax1 = fig.add_subplot(131)
    ax1.plot(world_times, estimated_forward[:, 0], label='estimated')
    ax1.plot(gt_times, gt_forward[:, 0], label='gt')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Forward X')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(world_times, estimated_forward[:, 1], label='estimated')
    ax2.plot(gt_times, gt_forward[:, 1], label='gt')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Forward Y')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(world_times, estimated_forward[:, 2], label='estimated')
    ax3.plot(gt_times, gt_forward[:, 2], label='gt')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Forward Z')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Plot up directions
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle('Up Direction (y-axis)')

    ax1 = fig.add_subplot(131)
    ax1.plot(world_times, estimated_up[:, 0], label='estimated')
    ax1.plot(gt_times, gt_up[:, 0], label='gt')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Up X')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(world_times, estimated_up[:, 1], label='estimated')
    ax2.plot(gt_times, gt_up[:, 1], label='gt')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Up Y')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(world_times, estimated_up[:, 2], label='estimated')
    ax3.plot(gt_times, gt_up[:, 2], label='gt')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Up Z')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Plot estimated_transforms_in_cam0 vs ground truth
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax1.plot(world_times, world_positions[:, 0], label='estimated x')
    ax1.plot(gt_times, gt_positions[:, 0], label='gt x')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('X [m]')
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.plot(world_times, world_positions[:, 1], label='estimated y')
    ax2.plot(gt_times, gt_positions[:, 1], label='gt y')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.plot(world_times, world_positions[:, 2], label='estimated z')
    ax3.plot(gt_times, gt_positions[:, 2], label='gt z')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Z [m]')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Visualize 3D points
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(viz_points_3d[0, :], viz_points_3d[1, :], viz_points_3d[2, :], s=1, alpha=0.5)
    # ax.set_xlabel('X [m]')
    # ax.set_ylabel('Y [m]')
    # ax.set_zlabel('Z [m]')
    # ax.set_title('Triangulated 3D Points')
    # plt.show()


if __name__ == "__main__":
    main()
