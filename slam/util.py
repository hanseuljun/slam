import numpy as np

from slam.data import DataFolder


def from_body_to_cam0(data: DataFolder, pose: np.ndarray) -> np.ndarray:
    # return data.cam0_extrinsics @ pose
    # return pose
    return np.linalg.inv(data.cam0_extrinsics) @ pose


def quaternion_to_rotation_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
