# gtsam.gtsam.noiseModel is a pybind11 submodule nested inside the compiled gtsam.gtsam
# extension (confirmed at runtime: gtsam.gtsam.noiseModel has no __file__, only a __spec__),
# so the stub generator that produced gtsam.pyi never captured its members -- `from
# .gtsam.noiseModel import *` (the auto-generated original of this file) resolved to nothing.
# Declared here by hand instead, covering the members actually used in this project plus their
# immediate siblings for completeness.
import numpy as np

class Base: ...

class Gaussian(Base):
    @staticmethod
    def Covariance(covariance: np.ndarray, smart: bool = ...) -> "Gaussian": ...

class Diagonal(Gaussian):
    @staticmethod
    def Sigmas(sigmas: np.ndarray, smart: bool = ...) -> "Diagonal": ...
    @staticmethod
    def Variances(variances: np.ndarray, smart: bool = ...) -> "Diagonal": ...
    @staticmethod
    def Precisions(precisions: np.ndarray, smart: bool = ...) -> "Diagonal": ...

class Constrained(Diagonal):
    @staticmethod
    def MixedSigmas(sigmas: np.ndarray) -> "Constrained": ...

class Isotropic(Diagonal):
    @staticmethod
    def Sigma(dim: int, sigma: float, smart: bool = ...) -> "Isotropic": ...
    @staticmethod
    def Variance(dim: int, variance: float, smart: bool = ...) -> "Isotropic": ...

class Unit(Isotropic):
    @staticmethod
    def Create(dim: int) -> "Unit": ...

class mEstimator:
    class Base: ...
    class Huber(Base):
        @staticmethod
        def Create(k: float) -> "mEstimator.Huber": ...
    class Cauchy(Base):
        @staticmethod
        def Create(k: float) -> "mEstimator.Cauchy": ...
    class Tukey(Base):
        @staticmethod
        def Create(k: float) -> "mEstimator.Tukey": ...

class Robust(Base):
    @staticmethod
    def Create(robust: "mEstimator.Base", noise: Base) -> "Robust": ...
