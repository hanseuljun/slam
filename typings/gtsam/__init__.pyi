from typing import overload

import numpy as np

from gtsam.gtsam import *
from gtsam import gtsam as gtsam, utils as utils
from gtsam import noiseModel as noiseModel, imuBias as imuBias
from gtsam.utils import findExampleDataFile as findExampleDataFile

# gtsam/__init__.py's _init() defines these two at runtime via `global Point2/Point3` --
# shims for the C++ Point2/Point3 types after GTSAM deleted them in favor of plain numpy
# vectors. The auto-generated stub above can't see a name injected into globals() from
# inside a function, so it's declared by hand here to match the real behavior/signature.
@overload
def Point2(x: np.ndarray) -> np.ndarray: ...
@overload
def Point2(x: float = ..., y: float = ...) -> np.ndarray: ...
@overload
def Point3(x: np.ndarray) -> np.ndarray: ...
@overload
def Point3(x: float = ..., y: float = ..., z: float = ...) -> np.ndarray: ...
