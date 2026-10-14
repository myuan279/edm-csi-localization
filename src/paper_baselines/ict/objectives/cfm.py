from __future__ import annotations

import torch


def flow_matching_loss(model, position: torch.Tensor, condition: torch.Tensor, *, generator=None) -> torch.Tensor:
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    time = torch.rand((len(position),), device=position.device, dtype=position.dtype, generator=generator)
    state = (1.0 - time[:, None]) * position + time[:, None] * noise
    target = noise - position
    prediction = model(state, time, condition)
    return (prediction - target).square().sum(dim=1).mean()
