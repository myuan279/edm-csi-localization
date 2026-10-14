"""312D RI-CSI MLP point-regression baseline for the EDM-Loc paper."""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ── MLP baseline ───────────────────────────────────────────────────────
class MLPRegressor(nn.Module):
    """
    与论文 baseline 一致 : 6 层 FC ,
    feature dims = [128, 64, 32, 32, 64, 128].
    输入 8 维 CSI, 输出 2 维归一化坐标.
    """
    def __init__(
        self,
        in_dim:    int = 312,
        hidden:    list = (128, 64, 32, 32, 64, 128),
        out_dim:   int = 2,
    ):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.GELU(), nn.LayerNorm(b)]
        layers += [nn.Linear(dims[-1], out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp(
    model:     MLPRegressor,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> float:
    model.train()
    mse, running, n = nn.MSELoss(), 0.0, 0
    for csi, pos in loader:
        csi, pos = csi.to(device), pos.to(device)
        optimizer.zero_grad()
        loss = mse(model(csi), pos)
        loss.backward()
        optimizer.step()
        running += loss.item()
        n += 1
    return running / max(n, 1)


@torch.no_grad()
def eval_mlp(
    model:        MLPRegressor,
    loader:       DataLoader,
    full_dataset,
    device:       torch.device,
    max_batches:  Optional[int] = None,
) -> dict:
    model.eval()
    all_pred, all_true = [], []
    for i, (csi, pos) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        pred = model(csi.to(device)).cpu()
        all_pred.append(pred)
        all_true.append(pos)
    return full_dataset.error_cm(torch.cat(all_pred), torch.cat(all_true))
