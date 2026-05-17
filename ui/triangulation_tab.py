import io
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui

from slam import DataFolder, triangulate_stereo_matches
from ui._utils import array_to_data_uri


def _fig_to_image(fig: plt.Figure) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return img


@dataclass
class _Results:
    n_cam0_keypoints: int
    n_cam1_keypoints: int
    n_3d_points: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    median_reprojection_error: float
    mean_reprojection_error: float
    keypoints_plot: np.ndarray
    points_3d_plot: np.ndarray


class TriangulationTabState:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._results: Optional[_Results] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._started: bool = False

    def _compute(self) -> None:
        try:
            data = self._data
            orb = cv2.ORB_create(nfeatures=2000)
            timestamp_ns = data.cam_timestamps_ns[0]
            cam0_img = cv2.imread(str(data.get_cam0_image_path(timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            cam1_img = cv2.imread(str(data.get_cam1_image_path(timestamp_ns)), cv2.IMREAD_GRAYSCALE)

            cam0_keypoints, cam0_descriptors = orb.detectAndCompute(cam0_img, None)
            cam1_keypoints, cam1_descriptors = orb.detectAndCompute(cam1_img, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(cam0_descriptors, cam1_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            points_3d = triangulate_stereo_matches(
                data, cam0_keypoints, cam0_descriptors, cam1_keypoints, cam1_descriptors
            )

            matched_cam0_pts = np.array([cam0_keypoints[m.queryIdx].pt for m in good_matches])
            K0 = data.cam0_intrinsics.to_matrix()
            dist_coeffs0 = np.array([
                data.cam0_intrinsics.k1,
                data.cam0_intrinsics.k2,
                data.cam0_intrinsics.p1,
                data.cam0_intrinsics.p2,
            ])
            projected_points, _ = cv2.projectPoints(
                points_3d.T, np.zeros(3), np.zeros(3), K0, dist_coeffs0
            )
            projected_points = projected_points.reshape(-1, 2)
            reprojection_errors = np.linalg.norm(matched_cam0_pts - projected_points, axis=1)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            ax1.imshow(cam0_img, cmap='gray')
            ax1.scatter([kp.pt[0] for kp in cam0_keypoints], [kp.pt[1] for kp in cam0_keypoints],
                        c='red', s=20, alpha=0.5)
            ax1.set_title(f'cam0 keypoints ({len(cam0_keypoints)})')
            ax1.axis('off')
            ax2.imshow(cam1_img, cmap='gray')
            ax2.scatter([kp.pt[0] for kp in cam1_keypoints], [kp.pt[1] for kp in cam1_keypoints],
                        c='blue', s=20, alpha=0.5)
            ax2.set_title(f'cam1 keypoints ({len(cam1_keypoints)})')
            ax2.axis('off')
            ax3.imshow(cam0_img, cmap='gray')
            ax3.scatter(projected_points[:, 0], projected_points[:, 1], c='green', s=20, alpha=0.5)
            ax3.set_title(f'Reprojected 3D points ({len(projected_points)})')
            ax3.axis('off')
            plt.tight_layout()
            keypoints_plot = _fig_to_image(fig)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points_3d[0, :], points_3d[1, :], points_3d[2, :], s=1, alpha=0.5)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title('Triangulated 3D points')
            plt.tight_layout()
            points_3d_plot = _fig_to_image(fig)

            self._results = _Results(
                n_cam0_keypoints=len(cam0_keypoints),
                n_cam1_keypoints=len(cam1_keypoints),
                n_3d_points=points_3d.shape[1],
                x_range=(float(points_3d[0].min()), float(points_3d[0].max())),
                y_range=(float(points_3d[1].min()), float(points_3d[1].max())),
                z_range=(float(points_3d[2].min()), float(points_3d[2].max())),
                median_reprojection_error=float(np.median(reprojection_errors)),
                mean_reprojection_error=float(np.mean(reprojection_errors)),
                keypoints_plot=keypoints_plot,
                points_3d_plot=points_3d_plot,
            )
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


def triangulation_tab(state: TriangulationTabState) -> None:
    with ui.column().classes('w-full'):
        loading_label = ui.label('Computing triangulation...')
        error_label = ui.label('').classes('text-red-500').set_visibility(False)
        stats_col = ui.column().set_visibility(False)
        img_keypoints = ui.image('').classes('w-full').set_visibility(False)
        img_3d = ui.image('').classes('w-full').set_visibility(False)

        def poll() -> None:
            if state._loading:
                return
            elif state._error:
                loading_label.set_visibility(False)
                error_label.text = f'Error: {state._error}'
                error_label.set_visibility(True)
                timer.cancel()
            elif state._results is not None:
                r = state._results
                loading_label.set_visibility(False)
                with stats_col:
                    ui.label(f'cam0 keypoints: {r.n_cam0_keypoints}')
                    ui.label(f'cam1 keypoints: {r.n_cam1_keypoints}')
                    ui.label(f'Triangulated 3D points: {r.n_3d_points}')
                    ui.label(f'X range: [{r.x_range[0]:.2f}, {r.x_range[1]:.2f}]')
                    ui.label(f'Y range: [{r.y_range[0]:.2f}, {r.y_range[1]:.2f}]')
                    ui.label(f'Z range: [{r.z_range[0]:.2f}, {r.z_range[1]:.2f}]')
                    ui.label(f'Median reprojection error: {r.median_reprojection_error:.2f} px')
                    ui.label(f'Mean reprojection error: {r.mean_reprojection_error:.2f} px')
                img_keypoints.source = array_to_data_uri(r.keypoints_plot)
                img_3d.source = array_to_data_uri(r.points_3d_plot)
                stats_col.set_visibility(True)
                img_keypoints.set_visibility(True)
                img_3d.set_visibility(True)
                timer.cancel()

        timer = ui.timer(0.5, poll)
