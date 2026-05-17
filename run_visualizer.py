from pathlib import Path

import cv2
from imgui_bundle import hello_imgui, immapp, immvision, imgui

from slam import DataFolder


class VisualizerState:
    def __init__(self, data: DataFolder):
        self.data = data
        self.frame_index = 0
        self._cached_index = -1
        self._cached_image = None

    def current_image(self):
        if self._cached_index != self.frame_index:
            ts = self.data.cam_timestamps_ns[self.frame_index]
            self._cached_image = cv2.imread(
                str(self.data.get_cam0_image_path(ts)), cv2.IMREAD_GRAYSCALE
            )
            self._cached_index = self.frame_index
        return self._cached_image


def make_gui(state: VisualizerState):
    def gui():
        viewport = imgui.get_main_viewport()
        imgui.set_next_window_pos(viewport.work_pos)
        imgui.set_next_window_size(viewport.work_size)
        imgui.begin(
            "##main",
            flags=imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar,
        )

        if imgui.begin_tab_bar("##tabs"):
            if imgui.begin_tab_item("Overview")[0]:
                imgui.end_tab_item()

            if imgui.begin_tab_item("Camera Frames")[0]:
                n = len(state.data.cam_timestamps_ns)
                _, state.frame_index = imgui.slider_int(
                    "Frame", state.frame_index, 0, n - 1
                )
                image = state.current_image()
                if image is not None:
                    immvision.image("##cam0", image, immvision.ImageParams())
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.end()

    return gui


def main():
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    state = VisualizerState(data)

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "SLAM Visualizer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "visualizer.ini"
    runner_params.callbacks.show_gui = make_gui(state)

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
