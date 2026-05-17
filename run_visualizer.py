from imgui_bundle import imgui, immapp, hello_imgui


def gui():
    viewport = imgui.get_main_viewport()
    imgui.set_next_window_pos(viewport.get_center(), imgui.Cond_.always, (0.5, 0.5))
    imgui.set_next_window_size((0, 0))
    imgui.begin(
        "##placeholder",
        flags=imgui.WindowFlags_.no_title_bar
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_move,
    )
    imgui.text("SLAM Visualizer")
    imgui.end()


runner_params = hello_imgui.RunnerParams()
runner_params.app_window_params.window_title = "SLAM Visualizer"
runner_params.app_window_params.window_geometry.size = (1280, 720)
runner_params.ini_filename = "visualizer.ini"
runner_params.callbacks.show_gui = gui

immapp.run(runner_params)
