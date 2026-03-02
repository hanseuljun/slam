from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import least_squares

from slam import DataFolder, solve_stereo_pnp
from slam.plot import plot_angular_velocities, plot_positions, plot_rotation_axes


def quaternion_to_rotation_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def optimize_angular_velocities(imu_angular_velocities_in_body_at_cam_times, pnp_attitudes, cam_rate_hz):
    N = len(pnp_attitudes)
    x0 = imu_angular_velocities_in_body_at_cam_times[:-1].flatten()  # (N-1)*3

    def accumulate(x):
        omegas = x.reshape(N - 1, 3)
        attitudes = [pnp_attitudes[0]]
        for omega in omegas:
            R, _ = cv2.Rodrigues(omega / cam_rate_hz)
            attitudes.append(attitudes[-1] @ R)
        return np.array(attitudes)  # (N, 3, 3)

    def residuals(x):
        omegas = x.reshape(N - 1, 3)
        optimized_attitudes = accumulate(x)
        omega_res = (omegas - imu_angular_velocities_in_body_at_cam_times[:-1]).flatten()
        attitude_res = (optimized_attitudes[1:] - pnp_attitudes[1:]).flatten()
        return np.concatenate([omega_res, attitude_res])

    result = least_squares(residuals, x0, method='lm')
    optimized_angular_velocities = result.x.reshape(N - 1, 3)
    optimized_attitudes = accumulate(result.x)
    return optimized_angular_velocities, optimized_attitudes


def run_pnp(data, orb, cam_timestamp_indices_in_range):
    keyframe_indices = [0]
    keyframe_num_temporal_matches = None
    pnp_poses_in_body = [data.cam0_extrinsics]
    pnp_angular_velocities_from_rvec_in_body = []
    num_temporal_matches_list = []
    reprojection_errors = []
    for i in range(1, len(cam_timestamp_indices_in_range)):
        if i % 100 == 0:
            print(f"i={i}")
        try:
            rvec, tvec, num_temporal_matches, reprojection_error = solve_stereo_pnp(data,
                                       orb,
                                       data.cam_timestamps_ns[cam_timestamp_indices_in_range[keyframe_indices[-1]]],
                                       data.cam_timestamps_ns[cam_timestamp_indices_in_range[i]])
        except Exception as e:
            print(f"solve_stereo_pnp failed at i={i}: {e}")
            continue
        if keyframe_num_temporal_matches is None:
            keyframe_num_temporal_matches = num_temporal_matches
        # TODO: i think the inverse of the extrinsics should be here but that is not the case based on data. figure out why.
        M = np.linalg.inv(data.cam0_extrinsics)
        rvec = M[:3, :3] @ rvec
        tvec = M[:3, :3] @ tvec
        pnp_angular_velocities_from_rvec_in_body.append(rvec.flatten() * data.cam0_rate_hz)
        num_temporal_matches_list.append(num_temporal_matches)
        reprojection_errors.append(reprojection_error)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        pnp_poses_in_body.append(pnp_poses_in_body[keyframe_indices[-1]] @ T)
        if num_temporal_matches < keyframe_num_temporal_matches / 2:
            keyframe_indices.append(i)
            keyframe_num_temporal_matches = None
    return pnp_poses_in_body, pnp_angular_velocities_from_rvec_in_body, num_temporal_matches_list, reprojection_errors


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    orb = cv2.ORB_create(nfeatures=2000)

    min_timestamp_ns = data.cam_timestamps_ns[0]
    max_timestamp_ns = min_timestamp_ns + int(20e9)  # 20 seconds
    cam_timestamp_indices_in_range = [i for i, t in enumerate(data.cam_timestamps_ns) if t <= max_timestamp_ns]

    pnp_poses_in_body, pnp_angular_velocities_from_rvec_in_body, _, _ = \
        run_pnp(data, orb, cam_timestamp_indices_in_range)

    # Get first ground truth sample as 4x4 pose matrix
    first_gt = data.ground_truth_samples[0]
    first_gt_pose = np.eye(4)
    first_gt_pose[:3, :3] = quaternion_to_rotation_matrix(first_gt.quaternion)
    first_gt_pose[:3, 3] = first_gt.position
    print(f"First ground truth pose:\n{first_gt_pose}")

    # Find camera timestamp closest to first ground truth timestamp
    cam_timestamps_ns = np.array([data.cam_timestamps_ns[i] for i in cam_timestamp_indices_in_range])
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt.timestamp_ns))
    print(f"First GT timestamp: {first_gt.timestamp_ns}")
    print(f"Closest cam timestamp: {cam_timestamps_ns[closest_cam_index]} (index {closest_cam_index})")
    print(f"Time diff: {(cam_timestamps_ns[closest_cam_index] - first_gt.timestamp_ns) / 1e6:.2f} ms")

    # Transform estimated poses to world frame
    pnp_poses_in_world = [first_gt_pose @ data.leica_extrinsics @ np.linalg.inv(pnp_poses_in_body[closest_cam_index]) @
                                     T @
                                     np.linalg.inv(data.leica_extrinsics) for T in pnp_poses_in_body]

    # Extract translations from pnp_poses_in_world
    pnp_positions_in_world = np.array([T[:3, 3] for T in pnp_poses_in_world])
    pnp_times = np.array([(data.cam_timestamps_ns[cam_timestamp_indices_in_range[i]] - min_timestamp_ns) / 1e9
                            for i in range(len(pnp_poses_in_world))])

    # Extract ground truth positions (within time range)
    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_timestamp_ns]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in gt_samples])

    # Extract rotations
    pnp_attitudes = np.array([T[:3, :3] for T in pnp_poses_in_world])
    gt_attitudes = np.array([quaternion_to_rotation_matrix(s.quaternion) for s in gt_samples])

    # Extract rotation axes from IMU attitudes (computed later, but referenced here)
    # imu_attitudes_in_body is computed in the angular velocity section below, so move it up
    imu_samples_in_range = [s for s in data.imu_samples if s.timestamp_ns <= max_timestamp_ns]
    imu_angular_velocities_in_body = np.array([s.angular_velocity for s in imu_samples_in_range])
    imu_rotations = imu_angular_velocities_in_body / data.imu0_rate_hz
    imu_attitudes_in_body = [np.eye(3)]
    for rot in imu_rotations:
        R, _ = cv2.Rodrigues(rot)
        imu_attitudes_in_body.append(imu_attitudes_in_body[-1] @ R)
    imu_attitudes_in_body = np.array(imu_attitudes_in_body)
    imu_times = np.array([(s.timestamp_ns - min_timestamp_ns) / 1e9 for s in imu_samples_in_range])
    imu_attitude_times = np.concatenate([[0.0], imu_times])

    # Transform IMU attitudes from body frame to world frame
    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples_in_range])
    pnp_cam_timestamps_ns = np.array([
        data.cam_timestamps_ns[cam_timestamp_indices_in_range[i]]
        for i in range(len(pnp_poses_in_body))
    ])
    nearest_imu_indices = np.array([np.argmin(np.abs(imu_timestamps_ns - ts)) for ts in pnp_cam_timestamps_ns])
    nearest_imu_times = (imu_timestamps_ns[nearest_imu_indices] - min_timestamp_ns) / 1e9
    imu_angular_velocities_in_body_at_cam_times = imu_angular_velocities_in_body[nearest_imu_indices]

    print("Optimizing angular velocities...")
    optimized_angular_velocities, optimized_attitudes = optimize_angular_velocities(
        imu_angular_velocities_in_body_at_cam_times, pnp_attitudes, data.cam0_rate_hz
    )
    closest_imu_index = np.argmin(np.abs(imu_timestamps_ns - first_gt.timestamp_ns))
    # +1 because imu_attitudes_in_body has an extra identity at index 0
    closest_imu_attitude_index = closest_imu_index + 1
    R_gt = first_gt_pose[:3, :3]
    R_leica = data.leica_extrinsics[:3, :3]
    imu_attitudes_in_world = np.array([
        R_gt @ R_leica @ imu_attitudes_in_body[closest_imu_attitude_index].T @ att @ R_leica.T
        for att in imu_attitudes_in_body
    ])

    plot_rotation_axes(
        series=[
            (pnp_times, pnp_attitudes, 'pnp'),
            # (imu_attitude_times, imu_attitudes_in_world, 'imu'),
            (gt_times, gt_attitudes, 'gt'),
            (pnp_times, optimized_attitudes, 'opt'),
        ],
    )

    plot_positions(
        series=[
            (pnp_times, pnp_positions_in_world, 'pnp'),
            (gt_times, gt_positions, 'gt'),
        ],
    )

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
        # The angular velocity coming from attitude difference from frame t + 1 and frame t
        # goes to frame t for angular velocity.
        gt_angular_velocity_times.append((gt_samples[j].timestamp_ns - min_timestamp_ns) / 1e9)
    gt_angular_velocities = np.array(gt_angular_velocities)
    gt_angular_velocity_times = np.array(gt_angular_velocity_times)

    pnp_angular_velocities = np.array(pnp_angular_velocities_from_rvec_in_body)
    pnp_angular_velocity_times = np.array([(data.cam_timestamps_ns[cam_timestamp_indices_in_range[i]] - min_timestamp_ns) / 1e9
                                             for i in range(len(pnp_angular_velocities))])

    plot_angular_velocities(
        series=[
            # (pnp_angular_velocity_times, pnp_angular_velocities, 'pnp'),
            (imu_times, imu_angular_velocities_in_body, 'imu'),
            (pnp_times, imu_angular_velocities_in_body_at_cam_times, 'imu@cam'),
            (gt_angular_velocity_times, gt_angular_velocities, 'gt'),
            (pnp_angular_velocity_times, optimized_angular_velocities, 'opt'),
        ],
    )

    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax.plot(pnp_times, pnp_times, label='pnp')
    # ax.plot(pnp_times, nearest_imu_times, label='nearest imu', linestyle='--')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Time [s]')
    # ax.set_title('Nearest IMU Time vs Slam Time')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
