from __future__ import annotations

import math

import torch


def sigma_schedule(
    count: int,
    *,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Ascending Karras sigma grid used by CT/iCT pair construction."""
    if count < 2 or not (0.0 < sigma_min < sigma_max) or rho <= 0:
        raise ValueError("invalid consistency sigma schedule")
    ramp = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype or torch.float32)
    low = sigma_min ** (1.0 / rho)
    high = sigma_max ** (1.0 / rho)
    return (low + ramp * (high - low)).pow(rho)


def ct_curriculum(
    update_index: int,
    total_updates: int,
    *,
    s0: int = 2,
    s1: int = 150,
    mu0: float = 0.9,
) -> tuple[int, float]:
    """Return the current CT interval count and target EMA decay."""
    if total_updates <= 0 or s0 < 2 or s1 < s0 or not (0.0 < mu0 < 1.0):
        raise ValueError("invalid CT curriculum")
    fraction = min(max(float(update_index) / float(total_updates), 0.0), 1.0)
    radicand = fraction * ((s1 + 1) ** 2 - s0**2) + s0**2
    levels = int(math.ceil(math.sqrt(radicand) - 1.0) + 1)
    levels = max(2, min(s1, levels))
    decay = math.exp(s0 * math.log(mu0) / levels)
    return levels, decay


def consistency_boundary(
    state: torch.Tensor,
    raw: torch.Tensor,
    sigma: torch.Tensor,
    *,
    sigma_min: float = 0.002,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    sigma = sigma.reshape(-1).to(dtype=state.dtype)
    shifted_sigma = (sigma - sigma_min).clamp_min(0.0)
    denominator = torch.sqrt(sigma.square() + sigma_data**2)
    c_skip = sigma_data**2 / (shifted_sigma.square() + sigma_data**2)
    c_out = shifted_sigma * sigma_data / denominator
    return c_skip[:, None] * state + c_out[:, None] * raw


def consistency_loss(
    online,
    target,
    position: torch.Tensor,
    condition: torch.Tensor,
    *,
    generator=None,
    levels: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must be [B,2]")
    if target is None:
        raise ValueError("CT requires an EMA target network")
    schedule = sigma_schedule(
        levels,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=position.device,
        dtype=position.dtype,
    )
    low_index = torch.randint(0, levels - 1, (len(position),), device=position.device, generator=generator)
    high_index = low_index + 1
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    low_sigma = schedule[low_index]
    high_sigma = schedule[high_index]
    low_state = position + low_sigma[:, None] * noise
    high_state = position + high_sigma[:, None] * noise
    online_prediction = online.forward_from_condition(
        high_state,
        high_sigma,
        online.encode_condition(condition),
    )
    with torch.no_grad():
        target_prediction = target.forward_from_condition(
            low_state,
            low_sigma,
            target.encode_condition(condition),
        )
    return (online_prediction - target_prediction).square().sum(dim=1).mean()
