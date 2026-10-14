from __future__ import annotations

import torch


def meanflow_loss(model, position: torch.Tensor, condition: torch.Tensor, *, generator=None) -> torch.Tensor:
    """MeanFlow identity loss with a condition-frozen directional derivative."""

    if position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("position must be [B,2]")
    noise = torch.randn(position.shape, device=position.device, dtype=position.dtype, generator=generator)
    time = torch.rand((len(position),), device=position.device, dtype=position.dtype, generator=generator)
    reference = torch.zeros_like(time)
    state = (1.0 - time[:, None]) * position + time[:, None] * noise
    velocity = noise - position
    condition_embedding = model.encode_condition(condition)
    prediction = model.forward_from_condition(state, reference, time, condition_embedding)

    # JVP follows the straight path in (state,time) while keeping the CSI
    # embedding and the reference time fixed, as required by MeanFlow.
    def field_fn(current_state: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        return model.forward_from_condition(current_state, reference, current_time, condition_embedding)

    _, directional = torch.autograd.functional.jvp(
        field_fn, (state, time), (velocity, torch.ones_like(time)), create_graph=True, strict=True
    )
    target = velocity - (time - reference)[:, None] * directional
    return (prediction - target.detach()).square().sum(dim=1).mean()
