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
    max_timestamp_ns = min_timestamp_ns + int(30e9)  # 30 seconds
    cam_timestamp_indices_in_range = [i for i, t in enumerate(data.cam_timestamps_ns) if t <= max_timestamp_ns]

    keyframe_indices = [0]
    keyframe_num_temporal_matches = None
    estimated_transforms_in_body = [data.cam0_extrinsics]
    estimated_angular_velocities_in_body = []
    num_temporal_matches_list = []
    for i in range(1, len(cam_timestamp_indices_in_range)):
        if i % 100 == 0:
            print(f"i={i}")
        try:
            rvec, tvec, num_temporal_matches = solve_step(data,
                                       orb,
                                       data.cam_timestamps_ns[cam_timestamp_indices_in_range[keyframe_indices[-1]]],
                                       data.cam_timestamps_ns[cam_timestamp_indices_in_range[i]])
        except Exception as e:
            print(f"solve_step failed at i={i}: {e}")
            continue
        if keyframe_num_temporal_matches is None:
            keyframe_num_temporal_matches = num_temporal_matches
        # TODO: i think the inverse of the extrinsics should be here but that is not the case based on data. figure out why.
        M = np.linalg.inv(data.cam0_extrinsics)
        rvec = M[:3, :3] @ rvec
        tvec = M[:3, :3] @ tvec
        estimated_angular_velocities_in_body.append(rvec.flatten() * data.cam0_rate_hz)
        num_temporal_matches_list.append(num_temporal_matches)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        estimated_transforms_in_body.append(estimated_transforms_in_body[keyframe_indices[-1]] @ T)
        if num_temporal_matches < keyframe_num_temporal_matches / 2:
            keyframe_indices.append(i)
            keyframe_num_temporal_matches = None

    # Get first ground truth sample as 4x4 transformation matrix
    first_gt = data.ground_truth_samples[0]
    first_gt_transform = np.eye(4)
    first_gt_transform[:3, :3] = quaternion_to_rotation_matrix(first_gt.quaternion)
    first_gt_transform[:3, 3] = first_gt.position
    print(f"First ground truth transform:\n{first_gt_transform}")

    # Find camera timestamp closest to first ground truth timestamp
    cam_timestamps_ns = np.array([data.cam_timestamps_ns[i] for i in cam_timestamp_indices_in_range])
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt.timestamp_ns))
    print(f"First GT timestamp: {first_gt.timestamp_ns}")
    print(f"Closest cam timestamp: {cam_timestamps_ns[closest_cam_index]} (index {closest_cam_index})")
    print(f"Time diff: {(cam_timestamps_ns[closest_cam_index] - first_gt.timestamp_ns) / 1e6:.2f} ms")

    # Transform estimated poses to world frame
    estimated_transforms_in_world = [first_gt_transform @ data.leica_extrinsics @ np.linalg.inv(estimated_transforms_in_body[closest_cam_index]) @
                                     T @
                                     np.linalg.inv(data.leica_extrinsics) for T in estimated_transforms_in_body]

    # Extract translations from estimated_transforms_in_world
    world_positions = np.array([T[:3, 3] for T in estimated_transforms_in_world])
    world_times = np.array([(data.cam_timestamps_ns[cam_timestamp_indices_in_range[i]] - min_timestamp_ns) / 1e9
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

    # Extract rotation axes from IMU attitudes (computed later, but referenced here)
    # imu_attitudes_in_body is computed in the angular velocity section below, so move it up
    imu_samples_in_range = [s for s in data.imu_samples if s.timestamp_ns <= max_timestamp_ns]
    imu_angular_velocities = np.array([s.angular_velocity for s in imu_samples_in_range])
    imu_rotations = imu_angular_velocities / data.imu0_rate_hz
    imu_attitudes_in_body = [np.eye(3)]
    for rot in imu_rotations:
        R, _ = cv2.Rodrigues(rot)
        imu_attitudes_in_body.append(imu_attitudes_in_body[-1] @ R)
    imu_attitudes_in_body = np.array(imu_attitudes_in_body)
    imu_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in imu_samples_in_range])
    imu_attitude_times = np.concatenate([[0.0], imu_times])

    # Transform IMU attitudes from body frame to world frame
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples_in_range])
    closest_imu_index = np.argmin(np.abs(imu_timestamps_ns - first_gt.timestamp_ns))
    # +1 because imu_attitudes_in_body has an extra identity at index 0
    closest_imu_attitude_index = closest_imu_index + 1
    R_gt = first_gt_transform[:3, :3]
    R_leica = data.leica_extrinsics[:3, :3]
    imu_attitudes_in_world = np.array([
        R_gt @ R_leica @ imu_attitudes_in_body[closest_imu_attitude_index].T @ att @ R_leica.T
        for att in imu_attitudes_in_body
    ])

    imu_right = imu_attitudes_in_world[:, :, 0]
    imu_up = imu_attitudes_in_world[:, :, 1]
    imu_forward = imu_attitudes_in_world[:, :, 2]

    # Plot rotation axes: rows = x/y/z axis, cols = x/y/z component
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle('Rotation Axes (estimated vs gt vs imu)')

    axis_names = ['Right (x-axis)', 'Up (y-axis)', 'Forward (z-axis)']
    estimated_axes = [estimated_right, estimated_up, estimated_forward]
    gt_axes = [gt_right, gt_up, gt_forward]
    imu_axes = [imu_right, imu_up, imu_forward]
    component_names = ['X', 'Y', 'Z']

    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.plot(world_times, estimated_axes[row][:, col], label='estimated')
            ax.plot(gt_times, gt_axes[row][:, col], label='gt')
            ax.plot(imu_attitude_times, imu_axes[row][:, col], label='imu', alpha=0.5)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(f'{axis_names[row]} {component_names[col]}')
            ax.legend()

    plt.tight_layout()
    plt.show()

    # Plot estimated_transforms_in_body vs ground truth
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

    # Compute ground truth angular velocities from consecutive quaternions
    gt_angular_velocities = []
    gt_angular_velocity_times = []
    for j in range(len(gt_samples) - 1):
        R0 = quaternion_to_rotation_matrix(gt_samples[j].quaternion)
        R1 = quaternion_to_rotation_matrix(gt_samples[j + 1].quaternion)
        R_rel = R0.T @ R1
        rvec_gt, _ = cv2.Rodrigues(R_rel)
        dt = (gt_samples[j + 1].timestamp_ns - gt_samples[j].timestamp_ns) / 1e9
        gt_angular_velocities.append(rvec_gt.flatten() / dt)
        gt_angular_velocity_times.append((gt_samples[j + 1].timestamp_ns - min_timestamp_ns) / 1e9)
    gt_angular_velocities = np.array(gt_angular_velocities)
    gt_angular_velocity_times = np.array(gt_angular_velocity_times)

    # Plot estimated angular velocities in body frame
    angular_velocities = np.array(estimated_angular_velocities_in_body)
    angular_velocity_times = np.array([(data.cam_timestamps_ns[cam_timestamp_indices_in_range[i + 1]] - min_timestamp_ns) / 1e9
                                       for i in range(len(angular_velocities))])

    fig, (ax_wx, ax_wy, ax_wz) = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle('Angular Velocity in Body Frame')

    ax_wx.plot(angular_velocity_times, angular_velocities[:, 0], label='estimated')
    ax_wx.plot(imu_times, imu_angular_velocities[:, 0], label='imu')
    ax_wx.plot(gt_angular_velocity_times, gt_angular_velocities[:, 0], label='gt')
    ax_wx.set_xlabel('Time [s]')
    ax_wx.set_ylabel('wx [rad/s]')
    ax_wx.legend()

    ax_wy.plot(angular_velocity_times, angular_velocities[:, 1], label='estimated')
    ax_wy.plot(imu_times, imu_angular_velocities[:, 1], label='imu')
    ax_wy.plot(gt_angular_velocity_times, gt_angular_velocities[:, 1], label='gt')
    ax_wy.set_xlabel('Time [s]')
    ax_wy.set_ylabel('wy [rad/s]')
    ax_wy.legend()

    ax_wz.plot(angular_velocity_times, angular_velocities[:, 2], label='estimated')
    ax_wz.plot(imu_times, imu_angular_velocities[:, 2], label='imu')
    ax_wz.plot(gt_angular_velocity_times, gt_angular_velocities[:, 2], label='gt')
    ax_wz.set_xlabel('Time [s]')
    ax_wz.set_ylabel('wz [rad/s]')
    ax_wz.legend()

    plt.tight_layout()
    plt.show()

    # Plot number of temporal matches over time
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(angular_velocity_times, num_temporal_matches_list)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Num Temporal Matches')
    ax.set_title('Number of Temporal Matches')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
