import argparse
import gtsam
from gtsam import GeneralSFMFactorCal3Bundler as GeneralSFMFactorCal3Bundler, PriorFactorPinholeCameraCal3Bundler as PriorFactorPinholeCameraCal3Bundler, PriorFactorPoint3 as PriorFactorPoint3, SfmData as SfmData
from gtsam.symbol_shorthand import P as P
from gtsam.utils import plot as plot

DEFAULT_BAL_DATASET: str

def plot_scene(scene_data: SfmData, result: gtsam.Values) -> None: ...
def run(args: argparse.Namespace) -> None: ...
def main() -> None: ...
