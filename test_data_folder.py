from pathlib import Path

from slam import DataFolder


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))

    print(f"path: {data.path}")

    print(f"\ncam_timestamps_ns ({len(data.cam_timestamps_ns)} total):")
    for i, ts in enumerate(data.cam_timestamps_ns[:5]):
        print(f"  [{i}] {ts}")

    print(f"\nimu_samples ({len(data.imu_samples)} total):")
    for i, sample in enumerate(data.imu_samples[:5]):
        print(f"  [{i}] t={sample.timestamp_ns}, ang_vel={sample.angular_velocity}, lin_acc={sample.linear_acceleration}")

    print(f"\nground_truth_samples ({len(data.ground_truth_samples)} total):")
    for i, sample in enumerate(data.ground_truth_samples[:5]):
        print(f"  [{i}] t={sample.timestamp_ns}, pos={sample.position}, quat={sample.quaternion}")

    print(f"\nleica_samples ({len(data.leica_samples)} total):")
    for i, sample in enumerate(data.leica_samples[:5]):
        print(f"  [{i}] t={sample.timestamp_ns}, pos={sample.position}")

    print(f"\ncam0_extrinsics:\n{data.cam0_extrinsics}")
    print(f"\ncam1_extrinsics:\n{data.cam1_extrinsics}")
    print(f"\nleica_extrinsics:\n{data.leica_extrinsics}")

    print(f"\ncam0_intrinsics:")
    print(f"  fx={data.cam0_intrinsics.fx}, fy={data.cam0_intrinsics.fy}")
    print(f"  cx={data.cam0_intrinsics.cx}, cy={data.cam0_intrinsics.cy}")
    print(f"  k1={data.cam0_intrinsics.k1}, k2={data.cam0_intrinsics.k2}")
    print(f"  p1={data.cam0_intrinsics.p1}, p2={data.cam0_intrinsics.p2}")

    print(f"\ncam1_intrinsics:")
    print(f"  fx={data.cam1_intrinsics.fx}, fy={data.cam1_intrinsics.fy}")
    print(f"  cx={data.cam1_intrinsics.cx}, cy={data.cam1_intrinsics.cy}")
    print(f"  k1={data.cam1_intrinsics.k1}, k2={data.cam1_intrinsics.k2}")
    print(f"  p1={data.cam1_intrinsics.p1}, p2={data.cam1_intrinsics.p2}")


if __name__ == "__main__":
    main()
