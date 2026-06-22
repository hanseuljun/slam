import argparse
import gtsam
from PreintegrationExample import PreintegrationExample
from _typeshed import Incomplete
from gtsam.symbol_shorthand import B as B, V as V, X as X
from gtsam.utils.plot import plot_pose3 as plot_pose3
from mpl_toolkits.mplot3d import Axes3D as Axes3D

BIAS_KEY: Incomplete
GRAVITY: float

def parse_args() -> argparse.Namespace: ...

class ImuFactorExample(PreintegrationExample):
    velocity: Incomplete
    priorNoise: Incomplete
    velNoise: Incomplete
    def __init__(self, twist_scenario: str = 'sick_twist') -> None: ...
    def addPrior(self, i: int, graph: gtsam.NonlinearFactorGraph): ...
    def optimize(self, graph: gtsam.NonlinearFactorGraph, initial: gtsam.Values): ...
    def plot(self, values: gtsam.Values, title: str = 'Estimated Trajectory', fignum: int = ..., show: bool = False): ...
    def run(self, T: int = 12, compute_covariances: bool = False, verbose: bool = True): ...
