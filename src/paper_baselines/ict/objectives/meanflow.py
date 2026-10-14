from __future__ import annotations

import torch


def _logit_normal(
    count: int,
    *,
    mean: float,
    std: float,
    device,
    dtype,
    generator=None,
) -> torch.Tensor:
    if std <= 0:
        raise ValueError("logit-normal std must be positive")
    logits = torch.randn((count,), device=device, dtype=dtype, generator=generator) * std + mean
    return torch.sigmoid(logits)


def sample_meanflow_times(
    count: int,
    *,
    mean: float = -0.4,
    std: float = 1.0,
    diagonal_mix: float = 0.75,
    device,
    dtype,
    generator=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0.0 <= diagonal_mix <= 1.0):
        raise ValueError("diagonal_mix must lie in [0,1]")
    first = _logit_normal(count, mean=mean, std=std, device=device, dtype=dtype, generator=generator)
    second = _logit_normal(count, mean=mean, std=std, device=device, dtype=dtype, generator=generator)
    low = torch.minimum(first, second)
    high = torch.maximum(first, second)
    diagonal = torch.rand((count,), device=device, dtype=dtype, generator=generator) < diagonal_mix
    reference = torch.where(diagonal, high, low)
    time = torch.where(diagonal, high, low + (high - low))
    return reference, time


def meanflow_loss(
    model,
    position: torch.Tensor,
    condition: torch.Tensor,
    *,
    generator=None,
    logit_normal_mean: float = -0.4,
    logit_normal_std: float = 1.0,
    diagonal_mix: float = 0.75,
    weight_epsilon: float = 0.01,
) -> torch.Tensor:
    """Full MeanFlow identity loss over arbitrary local intervals (r,t)."""
    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must be [B,2]")
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    reference, time = sample_meanflow_times(
        len(position),
        mean=logit_normal_mean,
        std=logit_normal_std,
        diagonal_mix=diagonal_mix,
        device=position.device,
        dtype=position.dtype,
        generator=generator,
    )
    state = (1.0 - time[:, None]) * position + time[:, None] * noise
    velocity = noise - position
    condition_embedding = model.encode_condition(condition)
    prediction = model.forward_from_condition(state, reference, time, condition_embedding)

    zeros = torch.zeros_like(reference)

    def field_fn(current_state: torch.Tensor, current_reference: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        return model.forward_from_condition(
            current_state,
            current_reference,
            current_time,
            condition_embedding,
        )

    _, directional = torch.autograd.functional.jvp(
        field_fn,
        (state, reference, time),
        (velocity, zeros, torch.ones_like(time)),
        create_graph=True,
        strict=True,
    )
    target = velocity - (time - reference)[:, None] * directional
    residual_sq = (prediction - target.detach()).square().sum(dim=1)
    return (residual_sq / (residual_sq + weight_epsilon)).mean()
