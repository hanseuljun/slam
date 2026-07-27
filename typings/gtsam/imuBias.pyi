# Same situation as noiseModel.pyi: gtsam.gtsam.imuBias is a pybind11 submodule nested inside
# the compiled extension that the stub generator never captured. Declared by hand.
from typing import overload

import numpy as np

class ConstantBias:
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, biasAcc: np.ndarray, biasGyro: np.ndarray) -> None: ...
    def vector(self) -> np.ndarray: ...
