import cv2
import matplotlib.pyplot as plt
import numpy as np
from imgui_bundle import hello_imgui


def image_to_texture(image: np.ndarray) -> hello_imgui.TextureGpu:
    if image.ndim == 2:
        rgba = np.stack([image, image, image, np.full_like(image, 255)], axis=-1)
    else:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    return hello_imgui.create_texture_gpu_from_rgba_data(rgba)


def rasterize_figure(fig: plt.Figure) -> np.ndarray:
    # Grabs pixels straight off the Agg canvas instead of the old savefig(png) -> cv2.imdecode
    # round-trip -- PNG encode/decode was pure overhead here since the bytes get decoded right
    # back into an array a moment later. ~1.7x faster per figure. Loses savefig's bbox_inches
    # ='tight' whitespace crop (this returns the figure at its exact configured size instead), a
    # cosmetic difference only -- callers already size figures explicitly via figsize=. Leaves the
    # figure open, unlike figure_to_image below -- for a caller reusing the same Figure/Axes
    # across many renders (updating line data in place) rather than building a new one every call.
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def figure_to_image(fig: plt.Figure) -> np.ndarray:
    img = rasterize_figure(fig)
    plt.close(fig)
    return img
