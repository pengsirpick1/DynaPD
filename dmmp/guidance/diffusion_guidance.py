"""Differentiable utility guidance and continuous pre-rounding refinement for DMMPv3 V4."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..guidance.strong_surrogates import (
    StrongAttackContext,
    StrongSurrogateBundle,
    ensemble_target_risk,
    ensemble_utility,
)


def soft_allocation(logits: torch.Tensor, candidate_mask: torch.Tensor, target_counts: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(len(logits), -1)
    flat_mask = candidate_mask.reshape(len(candidate_mask), -1).float()
    masked = flat_logits.masked_fill(flat_mask <= 0, -1e9)
    probabilities = torch.softmax(masked, dim=1) * flat_mask
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (probabilities * target_counts.reshape(-1, 1)).reshape_as(logits)


def defense_guidance_loss(
    policy_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_counts: torch.Tensor,
    attack_bundle: StrongSurrogateBundle,
    attack_context: StrongAttackContext,
    targets: torch.Tensor,
    robust_weight: float,
    soft_utility_weight: float = 0.10,
) -> torch.Tensor:
    allocation = soft_allocation(policy_logits, candidate_mask, target_counts)
    attack_logits = attack_bundle.logits_from_allocation(allocation, attack_context)
    hard_risk, _ = ensemble_target_risk(attack_logits, targets)
    utility, _ = ensemble_utility(attack_logits, targets, attack_bundle.weights, robust_weight)
    return hard_risk.mean() - float(soft_utility_weight) * utility.mean()


def defense_target_risk(
    policy_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_counts: torch.Tensor,
    attack_bundle: StrongSurrogateBundle,
    attack_context: StrongAttackContext,
    targets: torch.Tensor,
) -> torch.Tensor:
    allocation = soft_allocation(policy_logits, candidate_mask, target_counts)
    attack_logits = attack_bundle.logits_from_allocation(allocation, attack_context)
    hard_risk, _ = ensemble_target_risk(attack_logits, targets)
    return hard_risk.mean()


def policy_diversity_loss(first_logits: torch.Tensor, second_logits: torch.Tensor, candidate_mask: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    first = torch.softmax(first_logits.reshape(len(first_logits), -1), dim=1)
    second = torch.softmax(second_logits.reshape(len(second_logits), -1), dim=1)
    mask = candidate_mask.reshape(len(candidate_mask), -1)
    distance = ((first - second).abs() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.relu(float(margin) - distance).mean()


def differentiable_ddim_sample(
    diffusion,
    condition: torch.Tensor,
    bandwidth: torch.Tensor,
    *,
    sampler_steps: int,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = condition.shape[0]
    x = torch.randn(batch, diffusion.template_dim, device=condition.device, dtype=condition.dtype) if initial_noise is None else initial_noise
    requested_steps = max(1, min(int(sampler_steps), diffusion.diffusion_steps))
    timesteps = torch.unique_consecutive(
        torch.linspace(diffusion.diffusion_steps - 1, 0, requested_steps, device=condition.device).round().long()
    )
    for index, timestep in enumerate(timesteps.tolist()):
        t = torch.full((batch,), int(timestep), device=condition.device, dtype=torch.long)
        predicted_noise = diffusion.denoiser(x, t, condition, bandwidth)
        alpha_t = diffusion.alpha_cumprod[int(timestep)]
        predicted_x0 = ((x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)).clamp(
            -diffusion.x0_clip,
            diffusion.x0_clip,
        )
        next_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
        if next_timestep < 0:
            x = predicted_x0
            break
        alpha_next = diffusion.alpha_cumprod[next_timestep]
        direction = torch.sqrt(torch.clamp(1.0 - alpha_next, min=0.0)) * predicted_noise
        x = torch.sqrt(alpha_next) * predicted_x0 + direction
    return x - x.mean(dim=1, keepdim=True)


def guided_ddim_sample(
    diffusion,
    condition: torch.Tensor,
    bandwidth: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_counts: torch.Tensor,
    attack_bundle: StrongSurrogateBundle,
    attack_context: StrongAttackContext,
    targets: torch.Tensor,
    *,
    sampler_steps: int,
    guidance_weight: float,
    guidance_last_steps: int,
    robust_weight: float,
    soft_utility_weight: float,
    risk_tolerance: float,
    generator: torch.Generator,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    batch = condition.shape[0]
    if initial_noise is None:
        x = torch.randn(batch, diffusion.template_dim, device=condition.device, dtype=condition.dtype, generator=generator)
    else:
        x = initial_noise.to(device=condition.device, dtype=condition.dtype)
    requested_steps = max(1, min(int(sampler_steps), diffusion.diffusion_steps))
    timesteps = torch.unique_consecutive(torch.linspace(diffusion.diffusion_steps - 1, 0, requested_steps, device=condition.device).round().long())
    for index, timestep in enumerate(timesteps.tolist()):
        with torch.no_grad():
            t = torch.full((batch,), int(timestep), device=condition.device, dtype=torch.long)
            predicted_noise = diffusion.denoiser(x, t, condition, bandwidth)
            alpha_t = diffusion.alpha_cumprod[int(timestep)]
            predicted_x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            predicted_x0 = predicted_x0.clamp(-diffusion.x0_clip, diffusion.x0_clip)
        if float(guidance_weight) > 0 and index >= max(0, len(timesteps) - int(guidance_last_steps)):
            with torch.enable_grad():
                guided = predicted_x0.detach().requires_grad_(True)
                allocation = soft_allocation(guided.reshape(batch, *candidate_mask.shape[1:]), candidate_mask, target_counts)
                attack_logits = attack_bundle.logits_from_allocation(allocation, attack_context)
                risk_before, _ = ensemble_target_risk(attack_logits, targets)
                utility_before, _ = ensemble_utility(attack_logits, targets, attack_bundle.weights, robust_weight)
                hard_objective = risk_before.sum() - float(soft_utility_weight) * utility_before.sum()
                gradient = torch.autograd.grad(hard_objective, guided)[0]
                candidate = (guided - float(guidance_weight) * gradient).detach().clamp(
                    -diffusion.x0_clip,
                    diffusion.x0_clip,
                )
                with torch.no_grad():
                    candidate_allocation = soft_allocation(
                        candidate.reshape(batch, *candidate_mask.shape[1:]),
                        candidate_mask,
                        target_counts,
                    )
                    candidate_logits = attack_bundle.logits_from_allocation(candidate_allocation, attack_context)
                    risk_after, _ = ensemble_target_risk(candidate_logits, targets)
                    accepted = (risk_after <= risk_before.detach() + float(risk_tolerance)).reshape(-1, 1)
                    predicted_x0 = torch.where(accepted, candidate, guided.detach())
        next_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
        if next_timestep < 0:
            x = predicted_x0
            break
        with torch.no_grad():
            alpha_next = diffusion.alpha_cumprod[next_timestep]
            direction = torch.sqrt(torch.clamp(1.0 - alpha_next, min=0.0)) * predicted_noise
            x = torch.sqrt(alpha_next) * predicted_x0 + direction
    return x - x.mean(dim=1, keepdim=True)


def continuous_refine_logits(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_counts: torch.Tensor,
    attack_bundle: StrongSurrogateBundle,
    attack_context: StrongAttackContext,
    targets: torch.Tensor,
    *,
    keep_ratio: float,
    steps: int = 6,
    robust_weight: float = 0.35,
    soft_utility_weight: float = 0.10,
    risk_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if float(keep_ratio) >= 0.999 and int(steps) <= 0:
        return logits, {
            "keep_ratio": 1.0,
            "gate_mean": 1.0,
            "utility_before": 0.0,
            "utility_after": 0.0,
            "risk_before": 0.0,
            "risk_after": 0.0,
            "accepted_fraction": 1.0,
        }
    gate_logits = torch.full_like(logits, 2.0, requires_grad=True)
    opt = torch.optim.Adam([gate_logits], lr=0.15)
    effective_counts = target_counts * float(np.clip(keep_ratio, 0.0, 1.0))
    with torch.no_grad():
        before_allocation = soft_allocation(logits, candidate_mask, effective_counts)
        before_logits = attack_bundle.logits_from_allocation(before_allocation, attack_context)
        before_risk, _ = ensemble_target_risk(before_logits, targets)
        before_utility, _ = ensemble_utility(before_logits, targets, attack_bundle.weights, robust_weight)
    for _ in range(max(int(steps), 0)):
        gate = torch.sigmoid(gate_logits) * candidate_mask
        refined = logits + torch.log(gate.clamp_min(1e-4))
        allocation = soft_allocation(refined, candidate_mask, effective_counts)
        attack_logits = attack_bundle.logits_from_allocation(allocation, attack_context)
        risk, _ = ensemble_target_risk(attack_logits, targets)
        utility, _ = ensemble_utility(attack_logits, targets, attack_bundle.weights, robust_weight)
        sparsity = gate.mean()
        loss = risk.mean() - float(soft_utility_weight) * utility.mean() + 0.02 * sparsity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    gate = torch.sigmoid(gate_logits.detach()) * candidate_mask
    refined = logits.detach() + torch.log(gate.clamp_min(1e-4))
    with torch.no_grad():
        after_allocation = soft_allocation(refined, candidate_mask, effective_counts)
        after_logits = attack_bundle.logits_from_allocation(after_allocation, attack_context)
        after_risk, _ = ensemble_target_risk(after_logits, targets)
        after_utility, _ = ensemble_utility(after_logits, targets, attack_bundle.weights, robust_weight)
        accepted = after_risk <= before_risk + float(risk_tolerance)
        accept_shape = (-1,) + (1,) * (logits.ndim - 1)
        selected = torch.where(accepted.reshape(accept_shape), refined, logits.detach())
        selected_utility = torch.where(accepted, after_utility, before_utility)
        selected_risk = torch.where(accepted, after_risk, before_risk)
    return selected, {
        "keep_ratio": float(keep_ratio),
        "gate_mean": float(gate.sum().cpu() / candidate_mask.sum().clamp_min(1.0).cpu()),
        "utility_before": float(before_utility.mean().cpu()),
        "utility_after": float(selected_utility.mean().cpu()),
        "risk_before": float(before_risk.mean().cpu()),
        "risk_after": float(selected_risk.mean().cpu()),
        "accepted_fraction": float(accepted.float().mean().cpu()),
    }


def enforce_defense_risk_guard(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    target_counts: torch.Tensor,
    attack_bundle: StrongSurrogateBundle,
    attack_context: StrongAttackContext,
    targets: torch.Tensor,
    *,
    risk_tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        baseline_allocation = soft_allocation(baseline_logits, candidate_mask, target_counts)
        baseline_attack_logits = attack_bundle.logits_from_allocation(baseline_allocation, attack_context)
        baseline_risk, _ = ensemble_target_risk(baseline_attack_logits, targets)
        candidate_allocation = soft_allocation(candidate_logits, candidate_mask, target_counts)
        candidate_attack_logits = attack_bundle.logits_from_allocation(candidate_allocation, attack_context)
        candidate_risk, _ = ensemble_target_risk(candidate_attack_logits, targets)
        accepted = candidate_risk <= baseline_risk + float(risk_tolerance)
        accept_shape = (-1,) + (1,) * (candidate_logits.ndim - 1)
        selected = torch.where(accepted.reshape(accept_shape), candidate_logits, baseline_logits)
        selected_risk = torch.where(accepted, candidate_risk, baseline_risk)
    return selected, {
        "preference_guard_risk_before": float(baseline_risk.mean().cpu()),
        "preference_guard_risk_after": float(selected_risk.mean().cpu()),
        "preference_guard_accepted_fraction": float(accepted.float().mean().cpu()),
    }

