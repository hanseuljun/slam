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
    max_timestamp_ns = min_timestamp_ns + int(6e9)  # 6 seconds

    keyframe_index = 0
    estimated_transforms_in_cam0 = [np.eye(4)]
    i = 0
    while data.cam_timestamps_ns[i + 1] <= max_timestamp_ns:
        T, _ = solve_step(data,
                          orb,
                          data.cam_timestamps_ns[keyframe_index],
                          data.cam_timestamps_ns[i + 1])
        estimated_transforms_in_cam0.append(estimated_transforms_in_cam0[keyframe_index] @ T)
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

    # Transform estimated poses to world frame
    # with first_gt_transform and inverse of estimated_transforms_in_cam0[closest_cam_index],
    # convert T into matching first_gt_transform at closest_cam_index.
    # apply data.leica_extrinsics and data.cam0_extrinsics to get the coordinate system right.
    # TODO: i think inverse of data.leica_extrinsics and data.cam0_extrinsics on the left side of T, but the opposite is working. figure out why.
    estimated_transforms_in_world = [first_gt_transform @ data.leica_extrinsics @ np.linalg.inv(data.cam0_extrinsics) @ np.linalg.inv(estimated_transforms_in_cam0[closest_cam_index]) @
                                     T @
                                     data.cam0_extrinsics @ np.linalg.inv(data.leica_extrinsics) for T in estimated_transforms_in_cam0]

    # Extract translations from estimated_transforms_in_world
    world_positions = np.array([T[:3, 3] for T in estimated_transforms_in_world])
    world_times = np.array([(data.cam_timestamps_ns[i] - min_timestamp_ns) / 1e9
                            for i in range(len(estimated_transforms_in_world))])

    # Extract ground truth positions (within time range)
    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_timestamp_ns]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in gt_samples])

    # Extract rotation axes from estimated transforms
    estimated_right = np.array([T[:3, 0] for T in estimated_transforms_in_world])
    estimated_up = np.array([T[:3, 1] for T in estimated_transforms_in_world])
    estimated_forward = np.array([T[:3, 2] for T in estimated_transforms_in_world])

    # Extract rotation axes from ground truth quaternions
    gt_right = np.array([quaternion_to_rotation_matrix(s.quaternion)[:, 0] for s in gt_samples])
    gt_up = np.array([quaternion_to_rotation_matrix(s.quaternion)[:, 1] for s in gt_samples])
    gt_forward = np.array([quaternion_to_rotation_matrix(s.quaternion)[:, 2] for s in gt_samples])

    # Plot rotation axes: rows = x/y/z axis, cols = x/y/z component
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle('Rotation Axes (estimated vs gt)')

    axis_names = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
    estimated_axes = [estimated_right, estimated_up, estimated_forward]
    gt_axes = [gt_right, gt_up, gt_forward]
    component_names = ['X', 'Y', 'Z']

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(world_times, estimated_axes[row][:, col], label='estimated')
            ax.plot(gt_times, gt_axes[row][:, col], label='gt')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{axis_names[row]} {component_names[col]}')
            ax.legend()

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


if __name__ == "__main__":
    main()
