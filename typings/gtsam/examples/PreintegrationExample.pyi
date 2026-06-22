import gtsam
import numpy as np
from _typeshed import Incomplete
from gtsam.utils.plot import plot_pose3 as plot_pose3
from mpl_toolkits.mplot3d import Axes3D as Axes3D
from typing import Sequence

IMU_FIG: int
POSES_FIG: int
GRAVITY: int

class PreintegrationExample:
    @staticmethod
    def defaultParams(g: float): ...
    scenario: Incomplete
    dt: Incomplete
    maxDim: int
    labels: Incomplete
    colors: Incomplete
    params: Incomplete
    actualBias: Incomplete
    runner: Incomplete
    def __init__(self, twist: np.ndarray | None = None, bias: gtsam.imuBias.ConstantBias | None = None, params: gtsam.PreintegrationParams | None = None, dt: float = 0.01) -> None: ...
    def plotImu(self, t: float, measuredOmega: Sequence, measuredAcc: Sequence): ...
    def plotGroundTruthPose(self, t: float, scale: float = 0.3, time_interval: float = 0.01): ...
    def run(self, T: int = 12): ...
