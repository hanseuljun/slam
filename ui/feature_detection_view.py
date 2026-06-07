import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import hello_imgui, imgui

from slam.data import DataFolder
from slam.feature import FeatureDetectionResult, detect_features
from ui.utils import figure_to_image, image_to_texture


@dataclass
class _Results:
    cam0: FeatureDetectionResult
    cam1: FeatureDetectionResult
    plot: np.ndarray


class FeatureDetectionViewModel:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._results: Optional[_Results] = None
        self._tex: Optional[hello_imgui.TextureGpu] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._started: bool = False

    def _compute(self) -> None:
        try:
            data = self._data
            timestamp_ns = data.cam_timestamps_ns[0]
            cam0_img = cv2.imread(str(data.get_cam0_image_path(timestamp_ns)), cv2.IMREAD_GRAYSCALE)
            cam1_img = cv2.imread(str(data.get_cam1_image_path(timestamp_ns)), cv2.IMREAD_GRAYSCALE)

            cam0_result = detect_features(cam0_img)
            cam1_result = detect_features(cam1_img)

            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 5))
            cam0_drawn = cv2.drawKeypoints(
                cam0_img, cam0_result.keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            cam1_drawn = cv2.drawKeypoints(
                cam1_img, cam1_result.keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )
            ax0.imshow(cam0_drawn)
            ax0.set_title(f"cam0 — {len(cam0_result.keypoints)} keypoints")
            ax0.axis("off")
            ax1.imshow(cam1_drawn)
            ax1.set_title(f"cam1 — {len(cam1_result.keypoints)} keypoints")
            ax1.axis("off")
            plt.tight_layout()
            plot = figure_to_image(fig)

            self._results = _Results(cam0=cam0_result, cam1=cam1_result, plot=plot)
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


def feature_detection_view(model: FeatureDetectionViewModel) -> None:
    if model._loading:
        imgui.text("Detecting features...")
        return
    if model._error:
        imgui.text(f"Error: {model._error}")
        return
    if model._results is None:
        return

    r = model._results
    if model._tex is None:
        model._tex = image_to_texture(r.plot)

    imgui.begin_child("##fd_scroll", (0, 0), False)
    imgui.text(f"cam0 keypoints: {len(r.cam0.keypoints)}")
    imgui.text(f"cam1 keypoints: {len(r.cam1.keypoints)}")
    imgui.separator()
    tex = model._tex
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.end_child()
