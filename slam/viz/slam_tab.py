import io
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.optimize import least_squares
from imgui_bundle import imgui, hello_imgui

from slam import DataFolder, solve_stereo_pnp
from slam.plot import plot_positions, plot_attitudes_and_angular_velocities


def _to_texture(image: np.ndarray) -> hello_imgui.TextureGpu:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return hello_imgui.create_texture_gpu_from_rgba_data(rgba)


def _quaternion_to_rotation_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _optimize_angular_velocities(
    imu_angular_velocities_at_cam_times: np.ndarray,
    pnp_attitudes: np.ndarray,
    cam_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(pnp_attitudes)
    x0 = imu_angular_velocities_at_cam_times[:-1].flatten()

    def accumulate(x: np.ndarray) -> np.ndarray:
        omegas = x.reshape(N - 1, 3)
        attitudes = [pnp_attitudes[0]]
        for omega in omegas:
            R, _ = cv2.Rodrigues(omega / cam_rate_hz)
            attitudes.append(attitudes[-1] @ R)
        return np.array(attitudes)

    def residuals(x: np.ndarray) -> np.ndarray:
        omegas = x.reshape(N - 1, 3)
        optimized_attitudes = accumulate(x)
        omega_res = (omegas - imu_angular_velocities_at_cam_times[:-1]).flatten()
        attitude_res = (optimized_attitudes[1:] - pnp_attitudes[1:]).flatten()
        return np.concatenate([omega_res, attitude_res])

    result = least_squares(residuals, x0, method='lm')
    return result.x.reshape(N - 1, 3), accumulate(result.x)


def _run_pnp(
    data: DataFolder,
    orb: cv2.ORB,
    indices_in_range: list[int],
    on_progress: Callable[[float], None],
) -> tuple[np.ndarray, np.ndarray]:
    keyframe_indices: list[int] = [0]
    keyframe_num_temporal_matches: Optional[int] = None
    pnp_poses: list[np.ndarray] = [np.eye(4)]
    pnp_angular_velocities: list[np.ndarray] = []
    n = len(indices_in_range)
    for i in range(1, n):
        on_progress(i / n)
        try:
            rvec, tvec, num_temporal_matches, _ = solve_stereo_pnp(
                data, orb,
                data.cam_timestamps_ns[indices_in_range[keyframe_indices[-1]]],
                data.cam_timestamps_ns[indices_in_range[i]],
            )
        except Exception as e:
            print(f"solve_stereo_pnp failed at i={i}: {e}")
            continue
        if keyframe_num_temporal_matches is None:
            keyframe_num_temporal_matches = num_temporal_matches
        M = np.linalg.inv(data.cam0_extrinsics)
        rvec = M[:3, :3] @ rvec
        tvec = M[:3, :3] @ tvec
        pnp_angular_velocities.append(rvec.flatten() * data.cam0_rate_hz)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        pnp_poses.append(pnp_poses[keyframe_indices[-1]] @ T)
        if num_temporal_matches < keyframe_num_temporal_matches / 2:
            keyframe_indices.append(i)
            keyframe_num_temporal_matches = None
    return np.array(pnp_poses), np.array(pnp_angular_velocities)


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return img


@dataclass
class _Plots:
    positions: np.ndarray
    attitudes_and_angular_velocities: np.ndarray


def _compute_plots(
    data: DataFolder,
    set_progress: Callable[[float, str], None],
) -> _Plots:
    orb = cv2.ORB_create(nfeatures=2000)
    min_ts = data.cam_timestamps_ns[0]
    max_ts = min_ts + int(20e9)
    indices_in_range = [i for i, t in enumerate(data.cam_timestamps_ns) if t <= max_ts]

    set_progress(0.0, "Running PnP...")
    pnp_poses_without_initial, pnp_angular_velocities_from_rvec = _run_pnp(
        data, orb, indices_in_range,
        on_progress=lambda p: set_progress(p * 0.8, "Running PnP..."),
    )

    set_progress(0.8, "Transforming poses...")
    first_gt = data.ground_truth_samples[0]
    first_gt_pose = np.eye(4)
    first_gt_pose[:3, :3] = _quaternion_to_rotation_matrix(first_gt.quaternion)
    first_gt_pose[:3, 3] = first_gt.position

    cam_timestamps_ns = np.array([data.cam_timestamps_ns[i] for i in indices_in_range])
    closest_cam_index = np.argmin(np.abs(cam_timestamps_ns - first_gt.timestamp_ns))

    pnp_poses_in_world = np.array([
        first_gt_pose @ data.leica_extrinsics
        @ np.linalg.inv(data.cam0_extrinsics @ pnp_poses_without_initial[closest_cam_index])
        @ data.cam0_extrinsics @ T
        @ np.linalg.inv(data.leica_extrinsics)
        for T in pnp_poses_without_initial
    ])

    pnp_times = np.array([
        (data.cam_timestamps_ns[indices_in_range[i]] - min_ts) / 1e9
        for i in range(len(pnp_poses_in_world))
    ])
    pnp_positions_in_world = pnp_poses_in_world[:, :3, 3]
    pnp_attitudes = pnp_poses_in_world[:, :3, :3]
    pnp_angular_velocity_times = np.array([
        (data.cam_timestamps_ns[indices_in_range[i]] - min_ts) / 1e9
        for i in range(len(pnp_angular_velocities_from_rvec))
    ])

    gt_samples = [s for s in data.ground_truth_samples if s.timestamp_ns <= max_ts]
    gt_positions = np.array([s.position for s in gt_samples])
    gt_times = np.array([(s.timestamp_ns - min_ts) / 1e9 for s in gt_samples])
    gt_attitudes = np.array([_quaternion_to_rotation_matrix(s.quaternion) for s in gt_samples])

    imu_samples = [s for s in data.imu_samples if s.timestamp_ns <= max_ts]
    imu_angular_velocities = np.array([s.angular_velocity for s in imu_samples])
    imu_rotations = imu_angular_velocities / data.imu0_rate_hz
    imu_attitudes = [np.eye(3)]
    for rot in imu_rotations:
        R, _ = cv2.Rodrigues(rot)
        imu_attitudes.append(imu_attitudes[-1] @ R)
    imu_attitudes = np.array(imu_attitudes)
    imu_times = np.array([(s.timestamp_ns - min_ts) / 1e9 for s in imu_samples])
    imu_attitude_times = np.concatenate([[0.0], imu_times])

    imu_timestamps_ns = np.array([s.timestamp_ns for s in imu_samples])
    pnp_cam_timestamps_ns = np.array([
        data.cam_timestamps_ns[indices_in_range[i]]
        for i in range(len(pnp_poses_without_initial))
    ])
    nearest_imu_indices = np.array([
        np.argmin(np.abs(imu_timestamps_ns - ts)) for ts in pnp_cam_timestamps_ns
    ])
    imu_angular_velocities_at_cam_times = imu_angular_velocities[nearest_imu_indices]

    set_progress(0.85, "Optimizing angular velocities...")
    optimized_angular_velocities, optimized_attitudes = _optimize_angular_velocities(
        imu_angular_velocities_at_cam_times, pnp_attitudes, data.cam0_rate_hz
    )

    closest_imu_index = np.argmin(np.abs(imu_timestamps_ns - first_gt.timestamp_ns))
    R_gt = first_gt_pose[:3, :3]
    R_leica = data.leica_extrinsics[:3, :3]
    imu_attitudes_in_world = np.array([
        R_gt @ R_leica @ imu_attitudes[closest_imu_index + 1].T @ att @ R_leica.T
        for att in imu_attitudes
    ])

    gt_angular_velocities = []
    gt_angular_velocity_times = []
    for j in range(len(gt_samples) - 1):
        R0 = _quaternion_to_rotation_matrix(gt_samples[j].quaternion)
        R1 = _quaternion_to_rotation_matrix(gt_samples[j + 1].quaternion)
        R_rel = R0.T @ R1
        rvec_gt, _ = cv2.Rodrigues(R_rel)
        dt = (gt_samples[j + 1].timestamp_ns - gt_samples[j].timestamp_ns) / 1e9
        gt_angular_velocities.append(rvec_gt.flatten() / dt)
        gt_angular_velocity_times.append((gt_samples[j].timestamp_ns - min_ts) / 1e9)
    gt_angular_velocities = np.array(gt_angular_velocities)
    gt_angular_velocity_times = np.array(gt_angular_velocity_times)

    set_progress(0.95, "Rendering plots...")
    fig_pos = plot_positions(series=[
        (pnp_times, pnp_positions_in_world, 'pnp'),
        (gt_times, gt_positions, 'gt'),
    ])
    fig_att = plot_attitudes_and_angular_velocities(
        attitude_series=[
            (pnp_times, pnp_attitudes, 'pnp'),
            (imu_attitude_times, imu_attitudes_in_world, 'imu'),
            (gt_times, gt_attitudes, 'gt'),
            (pnp_times, optimized_attitudes, 'opt'),
        ],
        angular_velocity_series=[
            (pnp_angular_velocity_times, pnp_angular_velocities_from_rvec, 'pnp'),
            (imu_times, imu_angular_velocities, 'imu'),
            (pnp_times, imu_angular_velocities_at_cam_times, 'imu@cam'),
            (gt_angular_velocity_times, gt_angular_velocities, 'gt'),
            (pnp_angular_velocity_times, optimized_angular_velocities, 'opt'),
        ],
    )

    return _Plots(
        positions=_fig_to_image(fig_pos),
        attitudes_and_angular_velocities=_fig_to_image(fig_att),
    )


class SlamTabState:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._plots: Optional[_Plots] = None
        self._tex_positions: Optional[hello_imgui.TextureGpu] = None
        self._tex_attitudes: Optional[hello_imgui.TextureGpu] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._started: bool = False
        self._progress: float = 0.0
        self._progress_label: str = ""

    def _set_progress(self, value: float, label: str) -> None:
        self._progress = value
        self._progress_label = label

    def _compute(self) -> None:
        try:
            self._plots = _compute_plots(self._data, self._set_progress)
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._loading = True
        threading.Thread(target=self._compute, daemon=True).start()


def slam_tab(state: SlamTabState) -> None:

    if state._loading:
        imgui.text(state._progress_label)
        imgui.progress_bar(state._progress, (-1, 0))
        return
    if state._error:
        imgui.text(f"Error: {state._error}")
        return
    if state._plots is None:
        return

    if state._tex_positions is None:
        state._tex_positions = _to_texture(state._plots.positions)
        state._tex_attitudes = _to_texture(state._plots.attitudes_and_angular_velocities)

    imgui.begin_child("##slam_scroll", (0, 0), False)
    imgui.image(imgui.ImTextureRef(state._tex_positions.texture_id()), (state._tex_positions.width, state._tex_positions.height))
    imgui.image(imgui.ImTextureRef(state._tex_attitudes.texture_id()), (state._tex_attitudes.width, state._tex_attitudes.height))
    imgui.end_child()
