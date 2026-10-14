"""Frozen OpenCSI-style CNN architecture and canonical feature layout."""

from __future__ import annotations

import torch
import torch.nn as nn


N_CHANNELS = 2
N_SUBCARRIERS = 39
N_PORTS = 4
FULL312_DIM = N_CHANNELS * N_SUBCARRIERS * N_PORTS


def full312_to_csi_image(features: torch.Tensor) -> torch.Tensor:
    """Restore C-order full312 vectors as (channel, subcarrier, port)."""
    if features.ndim != 2 or features.shape[1] != FULL312_DIM:
        raise ValueError(f"Expected (B, 312), got {tuple(features.shape)}")
    return features.reshape(-1, N_CHANNELS, N_SUBCARRIERS, N_PORTS)


class OpenCSICNNRegressor(nn.Module):
    """OpenCSI Table-I CNN with normalized x/y regression output."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=100,
            kernel_size=(39, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=100,
            out_channels=1000,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(4000, 400, bias=True)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(400, 2, bias=True)

    def forward_with_shapes(
        self, full312_features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        image = full312_to_csi_image(full312_features)
        conv1 = self.relu1(self.conv1(image))
        conv2 = self.relu2(self.conv2(conv1))
        flattened = self.flatten(conv2)
        hidden = self.dropout(self.relu3(self.fc1(flattened)))
        output = self.fc2(hidden)
        return output, {
            "input_image": tuple(image.shape),
            "conv1": tuple(conv1.shape),
            "conv2": tuple(conv2.shape),
            "flatten": tuple(flattened.shape),
            "hidden": tuple(hidden.shape),
            "output": tuple(output.shape),
        }

    def forward(self, full312_features: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_shapes(full312_features)
        return output
