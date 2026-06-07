import threading
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import imgui, hello_imgui

from slam.coordinate_mapping_check import (
    CoordinateMappingCheckResult,
    CoordinateMappingChecker,
)
from slam.data import DataFolder
from slam.feature_detection import FeatureDetectionResult
from slam.stereo_matching import StereoMatchingResult
from ui.utils import figure_to_image, image_to_texture


def _plot_result(result: CoordinateMappingCheckResult) -> plt.Figure:
    times = result.times
    mean_errors = np.array([np.mean(f.projection_errors) if f.projection_errors else float('nan') for f in result.frames])
    num_matches = np.array([len(f.matches) for f in result.frames])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Coordinate Mapping Check (Ground Truth Poses)')

    ax1.plot(times, mean_errors)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Projection Error [px]')
    ax1.set_title('Mean Projection Error per Adjacent Frame Pair')

    ax2.plot(times, num_matches, color='orange')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Count')
    ax2.set_title('Number of Matched Features per Adjacent Frame Pair')

    plt.tight_layout()
    return fig


class CoordinateMappingViewModel:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._checker: Optional[CoordinateMappingChecker] = None
        self._result: Optional[CoordinateMappingCheckResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._texture: Optional[hello_imgui.TextureGpu] = None

    def start(
        self,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
    ) -> None:
        self._checker = CoordinateMappingChecker(
            self._data, feature_detection_result, stereo_matching_result,
        )
        self._result = None
        self._loading = True
        self._error = None
        self._texture = None
        threading.Thread(target=self._compute, daemon=True).start()

    def _compute(self) -> None:
        try:
            self._result = self._checker.run()
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False


def coordinate_mapping_view(model: CoordinateMappingViewModel) -> None:
    if model._checker is None:
        imgui.text("Waiting for stereo matching...")
        return
    if model._loading:
        imgui.text("Computing coordinate mapping check...")
        imgui.progress_bar(model._checker.progress, (-1, 0))
        return
    if model._error:
        imgui.text(f"Error: {model._error}")
        return
    if model._result is None:
        return

    if model._texture is None:
        model._texture = image_to_texture(figure_to_image(_plot_result(model._result)))

    tex = model._texture
    imgui.begin_child("##cmc_scroll", (0, 0), False)
    imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
    imgui.end_child()
