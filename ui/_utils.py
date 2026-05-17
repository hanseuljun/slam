import base64

import cv2
import numpy as np


def array_to_data_uri(image: np.ndarray) -> str:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode('.png', image)
    return f'data:image/png;base64,{base64.b64encode(buf).decode()}'
