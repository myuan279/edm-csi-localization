from __future__ import annotations

import torch


def sigma_schedule(count: int, *, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, device=None, dtype=None) -> torch.Tensor:
    if count < 2 or not (0.0 < sigma_min < sigma_max) or rho <= 0:
        raise ValueError("invalid consistency sigma schedule")
    ramp = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype or torch.float32)
    high = sigma_max ** (1.0 / rho)
    low = sigma_min ** (1.0 / rho)
    return (high + ramp * (low - high)).pow(rho)


def consistency_boundary(state: torch.Tensor, raw: torch.Tensor, sigma: torch.Tensor, *, sigma_data: float = 1.0) -> torch.Tensor:
    sigma = sigma.reshape(-1).to(dtype=state.dtype)
    c_skip = sigma_data**2 / (sigma.square() + sigma_data**2)
    c_out = sigma * sigma_data / torch.sqrt(sigma.square() + sigma_data**2)
    return c_skip[:, None] * state + c_out[:, None] * raw


def consistency_loss(online, target, position: torch.Tensor, condition: torch.Tensor, *, generator=None, levels: int = 18) -> torch.Tensor:
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must be [B,2]")
    schedule = sigma_schedule(levels, device=position.device, dtype=position.dtype)
    high_index = torch.randint(1, levels, (len(position),), device=position.device, generator=generator)
    low_index = high_index - 1
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    high_sigma = schedule[high_index]
    low_sigma = schedule[low_index]
    high_state = position + high_sigma[:, None] * noise
    low_state = position + low_sigma[:, None] * noise
    online_embedding = online.encode_condition(condition)
    online_prediction = online.forward_from_condition(high_state, high_sigma, online_embedding)
    with torch.no_grad():
        target_embedding = target.encode_condition(condition)
        target_prediction = target.forward_from_condition(low_state, low_sigma, target_embedding)
    return (online_prediction - target_prediction).square().sum(dim=1).mean()
