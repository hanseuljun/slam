from pathlib import Path

from imgui_bundle import hello_imgui, immapp

from slam import DataFolder
from app import App


def main() -> None:
    data = DataFolder.load(Path("data/machine_hall/MH_01_easy/mav0"))
    app = App(data)

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "SLAM Visualizer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "main.ini"
    runner_params.callbacks.show_gui = app.render

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
