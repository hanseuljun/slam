from _typeshed import Incomplete
from gtsam import Cal3_S2 as Cal3_S2, PinholeCameraCal3_S2 as PinholeCameraCal3_S2, Point3 as Point3, Pose3 as Pose3

class Options:
    triangle: Incomplete
    nrCameras: Incomplete
    def __init__(self, triangle: bool = False, nrCameras: int = 3, K=...) -> None: ...

class GroundTruth:
    K: Incomplete
    cameras: Incomplete
    points: Incomplete
    def __init__(self, K=..., nrCameras: int = 3, nrPoints: int = 4) -> None: ...
    def print(self, s: str = '') -> None: ...

class Data:
    class NoiseModels: ...
    K: Incomplete
    Z: Incomplete
    J: Incomplete
    odometry: Incomplete
    noiseModels: Incomplete
    def __init__(self, K=..., nrCameras: int = 3, nrPoints: int = 4) -> None: ...

def generate_data(options) -> tuple[Data, GroundTruth]: ...
