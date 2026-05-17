from imgui_bundle import imgui, immapp, hello_imgui


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
            imgui.end_tab_item()
        imgui.end_tab_bar()

    imgui.end()


def main():
    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "SLAM Visualizer"
    runner_params.app_window_params.window_geometry.size = (1280, 720)
    runner_params.ini_filename = "visualizer.ini"
    runner_params.callbacks.show_gui = gui

    immapp.run(runner_params)


if __name__ == "__main__":
    main()
