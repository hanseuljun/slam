from gtsam import symbol as symbol

class Options:
    hardConstraint: bool
    pointPriors: bool
    batchInitialization: bool
    reorderInterval: int
    alwaysRelinearize: bool
    def __init__(self) -> None: ...

def initialize(data, truth, options): ...
def step(data, isam, result, truth, currPoseIndex): ...
