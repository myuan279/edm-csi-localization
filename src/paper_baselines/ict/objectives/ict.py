from __future__ import annotations

import math

import torch

from .ct import sigma_schedule


def ict_curriculum(
    update_index: int,
    total_updates: int,
    *,
    s0: int = 10,
    s1: int = 1280,
) -> tuple[int, int]:
    if total_updates <= 0 or s0 < 2 or s1 < s0:
        raise ValueError("invalid iCT curriculum")
    ratio = max(2, int(s1 / s0))
    k_prime = max(1, int(total_updates / (math.log2(ratio) + 1.0)))
    phase = max(0, int(update_index) // k_prime)
    levels = min(s0 * (2**phase), s1) + 1
    return levels, k_prime


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def lognormal_interval_weights(
    sigma_low: torch.Tensor,
    sigma_high: torch.Tensor,
    *,
    p_mean: float = -1.1,
    p_std: float = 2.0,
) -> torch.Tensor:
    if p_std <= 0:
        raise ValueError("p_std must be positive")
    low_z = (torch.log(sigma_low.clamp_min(1e-12)) - p_mean) / p_std
    high_z = (torch.log(sigma_high.clamp_min(1e-12)) - p_mean) / p_std
    return (_normal_cdf(high_z) - _normal_cdf(low_z)).clamp_min(1e-12)


def radial_pseudo_huber(residual: torch.Tensor, *, c: float = 0.00054 * math.sqrt(2.0)) -> torch.Tensor:
    if c <= 0:
        raise ValueError("Pseudo-Huber c must be positive")
    return torch.sqrt(residual.square().sum(dim=1) + c**2) - c


def ict_loss(
    online,
    position: torch.Tensor,
    condition: torch.Tensor,
    *,
    generator=None,
    levels: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 1.0,
    p_mean: float = -1.1,
    p_std: float = 2.0,
    c: float = 0.00054 * math.sqrt(2.0),
) -> torch.Tensor:
    """Improved CT objective with weighted log-normal interval sampling."""
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must be [B,2]")
    schedule = sigma_schedule(
        levels,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=position.device,
        dtype=position.dtype,
    )
    sigma_low_grid = schedule[:-1]
    sigma_high_grid = schedule[1:]
    weights = lognormal_interval_weights(
        sigma_low_grid,
        sigma_high_grid,
        p_mean=p_mean,
        p_std=p_std,
    )
    weights = weights / weights.sum()
    indices = torch.multinomial(weights, len(position), replacement=True, generator=generator)
    sigma_low = sigma_low_grid[indices]
    sigma_high = sigma_high_grid[indices]
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    low_state = position + sigma_low[:, None] * noise
    high_state = position + sigma_high[:, None] * noise
    high_prediction = online.forward_from_condition(
        high_state,
        sigma_high,
        online.encode_condition(condition),
    )
    with torch.no_grad():
        low_prediction = online.forward_from_condition(
            low_state,
            sigma_low,
            online.encode_condition(condition),
        )
    residual = high_prediction - low_prediction
    return (radial_pseudo_huber(residual, c=c) / (sigma_high - sigma_low).clamp_min(1e-8)).mean()
