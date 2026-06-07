import io
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from imgui_bundle import imgui, hello_imgui

from slam import DataFolder, triangulate_stereo_matches


def _to_texture(image: np.ndarray) -> hello_imgui.TextureGpu:
    rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return hello_imgui.create_texture_gpu_from_rgba_data(rgba)


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


class TriangulationViewState:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._results: Optional[_Results] = None
        self._tex_keypoints: Optional[hello_imgui.TextureGpu] = None
        self._tex_3d: Optional[hello_imgui.TextureGpu] = None
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


def triangulation_view(state: TriangulationViewState) -> None:
    if state._loading:
        imgui.text("Computing triangulation...")
        return
    if state._error:
        imgui.text(f"Error: {state._error}")
        return
    if state._results is None:
        return

    r = state._results
    if state._tex_keypoints is None:
        state._tex_keypoints = _to_texture(r.keypoints_plot)
        state._tex_3d = _to_texture(r.points_3d_plot)

    imgui.begin_child("##tri_scroll", (0, 0), False)
    imgui.text(f"cam0 keypoints: {r.n_cam0_keypoints}")
    imgui.text(f"cam1 keypoints: {r.n_cam1_keypoints}")
    imgui.text(f"Triangulated 3D points: {r.n_3d_points}")
    imgui.text(f"X range: [{r.x_range[0]:.2f}, {r.x_range[1]:.2f}]")
    imgui.text(f"Y range: [{r.y_range[0]:.2f}, {r.y_range[1]:.2f}]")
    imgui.text(f"Z range: [{r.z_range[0]:.2f}, {r.z_range[1]:.2f}]")
    imgui.text(f"Median reprojection error: {r.median_reprojection_error:.2f} px")
    imgui.text(f"Mean reprojection error: {r.mean_reprojection_error:.2f} px")
    imgui.separator()
    tex = state._tex_keypoints
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    tex = state._tex_3d
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.end_child()
