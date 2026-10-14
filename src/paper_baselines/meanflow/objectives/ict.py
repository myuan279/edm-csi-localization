from __future__ import annotations

import torch

from .ct import consistency_loss


def pseudo_huber(residual: torch.Tensor, *, delta: float = 1.0) -> torch.Tensor:
    if delta <= 0:
        raise ValueError("Pseudo-Huber delta must be positive")
    return delta**2 * (torch.sqrt(1.0 + (residual / delta).square()) - 1.0)


def ict_loss(online, target, position: torch.Tensor, condition: torch.Tensor, *, generator=None, levels: int = 18, delta: float = 1.0) -> torch.Tensor:
    # iCT keeps the improved noise/target recipe separate from CT.  The target
    # branch is detached; target parameters are updated by the caller using the
    # registered iCT target rule.
    schedule = torch.linspace(0.002, 80.0, levels, device=position.device, dtype=position.dtype).flip(0)
    high_index = torch.randint(1, levels, (len(position),), device=position.device, generator=generator)
    low_index = high_index - 1
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    high_sigma, low_sigma = schedule[high_index], schedule[low_index]
    high_state = position + high_sigma[:, None] * noise
    low_state = position + low_sigma[:, None] * noise
    online_prediction = online.forward_from_condition(high_state, high_sigma, online.encode_condition(condition))
    with torch.no_grad():
        target_prediction = target.forward_from_condition(low_state, low_sigma, target.encode_condition(condition))
    return pseudo_huber(online_prediction - target_prediction, delta=delta).sum(dim=1).mean()
