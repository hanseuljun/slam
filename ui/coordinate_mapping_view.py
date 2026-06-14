import json
import threading
from typing import Optional

import cv2
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


def _match_color(i: int, total: int) -> tuple[int, int, int]:
    hue = int(i * 180 / max(total, 1)) % 180
    color_hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))


def _plot_result(result: CoordinateMappingCheckResult, enabled: dict[str, bool]) -> plt.Figure:
    times = result.times
    mean_errors = np.array([np.mean(f.projection_errors) if f.projection_errors else float('nan') for f in result.frames])
    mean_icp_errors = np.array([np.mean(f.icp_projection_errors) if f.icp_projection_errors else float('nan') for f in result.frames])
    num_matches = np.array([len(f.matches) for f in result.frames])

    show_error = enabled.get('gt') or enabled.get('icp')
    show_matches = enabled.get('matches', False)
    active = [s for s in ['errors', 'matches']
              if (s == 'errors' and show_error) or (s == 'matches' and show_matches)]

    if not active:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
        ax.axis('off')
        plt.tight_layout()
        return fig

    fig, axes = plt.subplots(len(active), 1, figsize=(12, 4 * len(active)))
    if len(active) == 1:
        axes = [axes]
    fig.suptitle('Coordinate Mapping Check (Ground Truth Poses)')
    ax_iter = iter(axes)

    if show_error:
        ax = next(ax_iter)
        if enabled.get('gt'):
            ax.plot(times, mean_errors, label='GT')
        if enabled.get('icp'):
            ax.plot(times, mean_icp_errors, label='ICP', color='green')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Projection Error [px]')
        ax.set_title('Mean Projection Error per Adjacent Frame Pair')
        ax.legend()

    if show_matches:
        ax = next(ax_iter)
        ax.plot(times, num_matches, color='orange')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Count')
        ax.set_title('Number of Matched Features per Adjacent Frame Pair')

    plt.tight_layout()
    return fig


def _checkboxes(enabled: dict[str, bool], id_suffix: str) -> bool:
    changed = False
    labels = list(enabled)
    for i, label in enumerate(labels):
        c, enabled[label] = imgui.checkbox(f"{label}##{id_suffix}", enabled[label])
        changed = changed or c
        if i < len(labels) - 1:
            imgui.same_line()
    return changed


class CoordinateMappingViewModel:
    def __init__(self, data: DataFolder) -> None:
        self._data = data
        self._checker: Optional[CoordinateMappingChecker] = None
        self._feature_detection_result: Optional[FeatureDetectionResult] = None
        self._stereo_matching_result: Optional[StereoMatchingResult] = None
        self._result: Optional[CoordinateMappingCheckResult] = None
        self._loading: bool = False
        self._error: Optional[str] = None
        self._plot_texture: Optional[hello_imgui.TextureGpu] = None
        self.frame_index: int = 0
        self.match_index_min: int = 0
        self.match_index_max: int = 0
        self.show_projected: bool = True
        self._cache_key: tuple = ()
        self._match_texture: Optional[hello_imgui.TextureGpu] = None
        self.plot_enabled: dict[str, bool] = {'gt': True, 'icp': True, 'matches': True}

    def start(
        self,
        feature_detection_result: FeatureDetectionResult,
        stereo_matching_result: StereoMatchingResult,
    ) -> None:
        self._feature_detection_result = feature_detection_result
        self._stereo_matching_result = stereo_matching_result
        self._checker = CoordinateMappingChecker(
            self._data, feature_detection_result, stereo_matching_result,
        )
        self._result = None
        self._loading = True
        self._error = None
        self._plot_texture = None
        self.frame_index = 0
        self.match_index_min = 0
        self.match_index_max = 0
        self.show_projected = True
        self._cache_key = ()
        self._match_texture = None
        threading.Thread(target=self._compute, daemon=True).start()

    def _compute(self) -> None:
        try:
            self._result = self._checker.run()
            self.match_index_max = max(0, len(self._result.frames[0].matches) - 1)
        except Exception as e:
            self._error = str(e)
        finally:
            self._loading = False

    def current_match_texture(self) -> Optional[hello_imgui.TextureGpu]:
        if self._result is None or self._feature_detection_result is None or self._stereo_matching_result is None:
            return None
        cache_key = (self.frame_index, self.match_index_min, self.match_index_max, self.show_projected)
        if self._cache_key != cache_key:
            self._match_texture = None
            frame = self._result.frames[self.frame_index]
            sm_k = self._stereo_matching_result.frames[self.frame_index]
            sm_k1 = self._stereo_matching_result.frames[self.frame_index + 1]
            fd_k = self._feature_detection_result.frames[self.frame_index]
            fd_k1 = self._feature_detection_result.frames[self.frame_index + 1]

            cam0_img_k = cv2.imread(
                str(self._data.get_cam0_image_path(sm_k.timestamp_ns)),
                cv2.IMREAD_GRAYSCALE,
            )
            cam0_img_k1 = cv2.imread(
                str(self._data.get_cam0_image_path(sm_k1.timestamp_ns)),
                cv2.IMREAD_GRAYSCALE,
            )
            kps_k = [fd_k.cam0_keypoints[m.queryIdx] for m in sm_k.matches]
            matches = frame.matches[self.match_index_min:self.match_index_max + 1]
            projected = frame.projected_points[self.match_index_min:self.match_index_max + 1]

            total = len(matches)
            img_matches = None
            for i, m in enumerate(matches):
                color = _match_color(i, total)
                flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                if img_matches is not None:
                    flags |= cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                img_matches = cv2.drawMatches(
                    cam0_img_k, kps_k,
                    cam0_img_k1, fd_k1.cam0_keypoints,
                    [m], img_matches,
                    matchColor=color,
                    flags=flags,
                )

            if img_matches is None:
                h = max(cam0_img_k.shape[0], cam0_img_k1.shape[0])
                img_matches = np.zeros((h, cam0_img_k.shape[1] + cam0_img_k1.shape[1], 3), dtype=np.uint8)
                img_matches[:cam0_img_k.shape[0], :cam0_img_k.shape[1]] = cv2.cvtColor(cam0_img_k, cv2.COLOR_GRAY2BGR)
                img_matches[:cam0_img_k1.shape[0], cam0_img_k.shape[1]:] = cv2.cvtColor(cam0_img_k1, cv2.COLOR_GRAY2BGR)

            if self.show_projected:
                offset_x = cam0_img_k.shape[1]
                for i, pt in enumerate(projected):
                    color = _match_color(i, total)
                    x, y = int(round(pt[0])) + offset_x, int(round(pt[1]))
                    cv2.drawMarker(img_matches, (x, y), color, cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=1)

            self._match_texture = image_to_texture(img_matches)
            self._cache_key = cache_key
        return self._match_texture


def _download_transforms(result: CoordinateMappingCheckResult) -> None:
    records = []
    for f in result.frames:
        records.append({
            "timestamp_ns": f.timestamp_ns,
            "gt_transform": f.gt_transform.tolist(),
            "icp_transform": f.icp_transform.tolist() if f.icp_transform is not None else None,
        })
    import os
    os.makedirs("tmp", exist_ok=True)
    path = "tmp/inter_frame_transforms.json"
    with open(path, "w") as fp:
        json.dump(records, fp, indent=2)
    print(f"Saved {len(records)} inter-frame transforms to {path}")


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

    n = len(model._result.frames)
    first_ts_ns = model._data.cam_timestamps_ns[0]

    frame = model._result.frames[model.frame_index]
    num_matches = len(frame.matches)

    tex = model.current_match_texture()
    if tex is not None:
        imgui.image(imgui.ImTextureRef(tex.texture_id()), (tex.width, tex.height))
        imgui.set_next_item_width(tex.width)
        changed, new_index = imgui.slider_int("##cm_slider", model.frame_index, 0, n - 1)
        if changed:
            model.frame_index = new_index
            model.match_index_min = 0
            model.match_index_max = max(0, len(model._result.frames[new_index].matches) - 1)

    imgui.text("Frame")
    imgui.same_line()
    imgui.set_next_item_width(200)
    changed, new_index = imgui.input_int("##cm_frame_input", model.frame_index, step=1)
    if changed:
        model.frame_index = max(0, min(n - 1, new_index))
        model.match_index_min = 0
        model.match_index_max = max(0, len(model._result.frames[model.frame_index].matches) - 1)

    if num_matches > 0:
        max_idx = num_matches - 1
        imgui.text("Match range")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_min, new_min = imgui.input_int("##cm_match_min", model.match_index_min, step=1)
        if changed_min:
            model.match_index_min = max(0, min(model.match_index_max, new_min))
        imgui.same_line()
        imgui.text("to")
        imgui.same_line()
        imgui.set_next_item_width(150)
        changed_max, new_max = imgui.input_int("##cm_match_max", model.match_index_max, step=1)
        if changed_max:
            model.match_index_max = max(model.match_index_min, min(max_idx, new_max))
        imgui.same_line()
        _, model.show_projected = imgui.checkbox("Show projected", model.show_projected)

    imgui.text(f"Matches: {num_matches}")
    mean_err = np.mean(frame.projection_errors) if frame.projection_errors else float('nan')
    imgui.text(f"Mean projection error (GT):  {mean_err:.2f} px")
    mean_icp_err = np.mean(frame.icp_projection_errors) if frame.icp_projection_errors else float('nan')
    imgui.text(f"Mean projection error (ICP): {mean_icp_err:.2f} px")
    imgui.text(f"Timestamp: {frame.timestamp_ns} ns")
    imgui.text(f"Time since first frame: {(frame.timestamp_ns - first_ts_ns) / 1e9:.3f} s")

    imgui.separator()

    if imgui.button("Download inter-frame transforms"):
        _download_transforms(model._result)

    imgui.separator()

    if _checkboxes(model.plot_enabled, "cm_plot"):
        model._plot_texture = None
    if model._plot_texture is None:
        model._plot_texture = image_to_texture(figure_to_image(_plot_result(model._result, model.plot_enabled)))

    plot_tex = model._plot_texture
    imgui.image(imgui.ImTextureRef(plot_tex.texture_id()), (plot_tex.width, plot_tex.height))
