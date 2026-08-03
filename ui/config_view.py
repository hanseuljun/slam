from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from imgui_bundle import imgui


@dataclass
class ConfigViewModel:
    data_paths: list[str] = field(default_factory=list)
    selected_index: int = 0
    start_s: float = 0.0
    duration_s: float = 200.0
    # Off by default: SLAM doesn't need any of these three anymore -- SlamSolver computes its own
    # feature detection/stereo matching/optical flow internally (see _compute in slam.py) -- so
    # they exist purely for their own diagnostic tabs, at the cost of redundant compute duplicating
    # what SLAM already does itself. Opt in explicitly, same as the other diagnostics below.
    run_feature_detection: bool = False
    run_stereo_matching: bool = False
    run_optical_flow: bool = False
    run_coordinate_mapping_check: bool = False
    run_imu_initialization: bool = False
    # Unlike the diagnostics above, this drives SLAM's own solve (SlamSolver's enable_loop_closure)
    # rather than a redundant side computation -- on by default now that it's been validated as a
    # net win; see SlamSolver.__init__'s enable_loop_closure for the remaining known tradeoff.
    run_loop_closure: bool = True

    @property
    def data_path_str(self) -> str:
        return self.data_paths[self.selected_index]

    def sanitize(self) -> None:
        """Enforce the pipeline's real dependency chain (feature detection -> stereo matching ->
        optical flow / coordinate mapping check) regardless of how these flags got set -- so a
        stage's checkbox can never end up True while something it needs is False.
        """
        if not self.run_feature_detection:
            self.run_stereo_matching = False
        if not self.run_stereo_matching:
            self.run_optical_flow = False
            self.run_coordinate_mapping_check = False


def _dependent_checkbox(label: str, value: bool, enabled: bool, requires: str) -> bool:
    """A checkbox that's forced off and greyed out whenever its prerequisite stage is disabled,
    with a tooltip explaining why -- rather than letting the user check a box that would silently
    do nothing.
    """
    if not enabled:
        value = False
    imgui.begin_disabled(not enabled)
    _, value = imgui.checkbox(label, value)
    imgui.end_disabled()
    if not enabled and imgui.is_item_hovered():
        imgui.set_tooltip(f"Requires {requires}")
    return value


def config_view(model: ConfigViewModel, on_run: Callable[[], None]) -> None:
    model.sanitize()

    # Stacked vertically (rather than same_line()-chained) since this now renders in the
    # collapsible left sidebar (see root_view in main.py) instead of a full-width top bar.
    labels = [Path(p).parent.name for p in model.data_paths]
    imgui.text("Dataset")
    imgui.set_next_item_width(-1)
    _, model.selected_index = imgui.combo("##data_path", model.selected_index, labels)

    imgui.text("Start (s)")
    imgui.set_next_item_width(-1)
    _, model.start_s = imgui.input_float("##start_s", model.start_s, step=1.0)

    imgui.text("Duration (s)")
    imgui.set_next_item_width(-1)
    _, model.duration_s = imgui.input_float("##duration_s", model.duration_s, step=1.0)

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    # Each stage's checkbox is only interactive while the stage(s) it depends on are enabled --
    # see ConfigViewModel.sanitize for the dependency chain this mirrors.
    _, model.run_feature_detection = imgui.checkbox("Feature Detection", model.run_feature_detection)
    model.run_stereo_matching = _dependent_checkbox(
        "Stereo Matching", model.run_stereo_matching, model.run_feature_detection, "Feature Detection")
    model.run_optical_flow = _dependent_checkbox(
        "Optical Flow", model.run_optical_flow, model.run_stereo_matching, "Stereo Matching")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    model.run_coordinate_mapping_check = _dependent_checkbox(
        "Coordinate Mapping Check", model.run_coordinate_mapping_check, model.run_stereo_matching,
        "Stereo Matching")
    _, model.run_imu_initialization = imgui.checkbox(
        "IMU Initialization", model.run_imu_initialization
    )
    # Off by default: loop closure's candidate search is O(K^2) brute-force descriptor matching
    # (not yet cheap enough to run unconditionally) and its candidate-consolidation step is
    # validated-but-not-tuned (see slam.py's _LoopClosureDetector docstring) --
    # a clear win on one regression-check sequence, a real if smaller localized regression on
    # another.
    _, model.run_loop_closure = imgui.checkbox(
        "Loop Closure", model.run_loop_closure
    )

    imgui.spacing()
    if imgui.button("Run Again", (-1, 0)):
        on_run()
