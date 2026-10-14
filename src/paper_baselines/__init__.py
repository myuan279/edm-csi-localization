"""Implementations of the eight comparison methods in Table 1."""

from .cnn import OpenCSICNNRegressor
from .ddpm_loc import DDPMLocDenoiseNet
from .gaussian import FullCovarianceGaussianMLP, HomoscedasticGaussianMLP
from .knn import TrainRPPrototypeKNN
from .mlp import MLPRegressor

__all__ = [
    "DDPMLocDenoiseNet",
    "FullCovarianceGaussianMLP",
    "HomoscedasticGaussianMLP",
    "MLPRegressor",
    "OpenCSICNNRegressor",
    "TrainRPPrototypeKNN",
]
