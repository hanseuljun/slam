from imgui_bundle import imgui, immapp


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


immapp.run(gui, window_title="SLAM Visualizer", window_size=(1280, 720))
