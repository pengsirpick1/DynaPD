from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.diffusion.models import build_policy_diffusion
from dmmp.target_policy.config import load_target_policy_config
from dmmp.target_policy.losses import allocation_kl_loss, categorical_kl_loss, masked_smooth_l1, symmetric_kl_from_logits
from dmmp.target_policy.target_pool import TargetPolicyPool


class TargetConditionProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(max(1, int(num_layers))):
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.SiLU()])
            in_dim = int(hidden_dim)
        layers.pop()
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class TargetAuxHeads(nn.Module):
    def __init__(self, hidden_dim: int, patch_num: int):
        super().__init__()
        self.family = nn.Linear(int(hidden_dim), 5)
        self.primitive = nn.Linear(int(hidden_dim), 5)
        self.effect = nn.Linear(int(hidden_dim), 2 * int(patch_num))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.family(hidden), self.primitive(hidden), self.effect(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/smoke x0* target diffusion v1.")
    parser.add_argument("--pool_dir", required=True)
    parser.add_argument("--config", default="configs/x0_target_diffusion_v1.yaml")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _default_output_dir(smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "target_policy_diffusion_v1_smoke" if smoke else "target_policy_diffusion_v1"
    return PROJECT_ROOT / "results" / f"{stamp}_{suffix}"


def _resolve_output_dir(value: str, *, smoke: bool, overwrite: bool) -> Path:
    output = Path(value) if value else _default_output_dir(smoke)
    sentinel_names = {"target_diffusion_checkpoint.pt", "training_log.json", "training_summary_zh.md"}
    existing = [name for name in sentinel_names if (output / name).exists()]
    if existing and not bool(overwrite):
        raise SystemExit(
            f"Refusing to overwrite existing training artifacts in {output}. "
            f"Existing files: {', '.join(sorted(existing))}. Pass --overwrite to replace them."
        )
    return output


def _condition_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    prefix = np.asarray(arrays["prefix_vector"], dtype=np.float32)
    budget = np.asarray(arrays["budget_ratio"], dtype=np.float32).reshape(-1, 1)
    family = np.asarray(arrays["family_weights"], dtype=np.float32)
    primitive = np.asarray(arrays["primitive_weights"], dtype=np.float32)
    mask = np.asarray(arrays["allowed_mask"], dtype=np.float32).reshape(len(prefix), -1)
    return np.concatenate([prefix, budget, family, primitive, mask], axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    cfg = load_target_policy_config(args.config)
    if str(cfg.beta_schedule).lower() != "linear":
        raise ValueError("Only beta_schedule=linear is currently supported by dmmp.diffusion.models.build_policy_diffusion.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output = _resolve_output_dir(args.output_dir, smoke=bool(args.smoke), overwrite=bool(args.overwrite))
    arrays = TargetPolicyPool(args.pool_dir).load_training_arrays()
    x0 = np.asarray(arrays["x0_star"], dtype=np.float32).reshape(len(arrays["x0_star"]), -1)
    condition_np = _condition_matrix(arrays)
    allocation_np = np.asarray(arrays["policy_allocation"], dtype=np.float32)
    mask_np = np.asarray(arrays["allowed_mask"], dtype=np.float32)
    row_count = len(x0)
    if row_count == 0:
        raise ValueError("Empty target pool")
    if args.batch_size is not None:
        requested_batch_size = int(args.batch_size)
    elif str(cfg.batch_size).lower() == "auto":
        requested_batch_size = 4 if args.smoke else min(64, row_count)
    else:
        requested_batch_size = int(cfg.batch_size)
    batch_size = min(max(1, requested_batch_size), row_count)
    hidden_dim = min(int(cfg.hidden_dim), 128) if args.smoke else int(cfg.hidden_dim)
    projector = TargetConditionProjector(condition_np.shape[1], hidden_dim, num_layers=int(cfg.num_layers)).to(device)
    aux_heads = TargetAuxHeads(hidden_dim, int(cfg.strategy_horizon)).to(device)
    diffusion = build_policy_diffusion(
        hidden_dim,
        patch_num=int(cfg.strategy_horizon),
        hidden_dim=hidden_dim,
        diffusion_steps=int(cfg.diffusion_steps),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(aux_heads.parameters()) + list(diffusion.parameters()),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    rng = np.random.default_rng(int(cfg.seed))
    losses = []
    max_steps = int(args.max_steps) if int(args.max_steps) > 0 else int(np.ceil(row_count / max(batch_size, 1)))
    max_steps = max(1, max_steps)
    epochs = max(1, int(args.epochs if args.epochs is not None else cfg.epochs))
    use_amp = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        for step in range(max_steps):
            rows = rng.choice(row_count, size=batch_size, replace=row_count < batch_size)
            x0_t = torch.as_tensor(x0[rows], dtype=torch.float32, device=device)
            cond_t = projector(torch.as_tensor(condition_np[rows], dtype=torch.float32, device=device))
            family_logits, primitive_logits, effect_logits_flat = aux_heads(cond_t)
            mask_t = torch.as_tensor(mask_np[rows], dtype=torch.float32, device=device)
            alloc_t = torch.as_tensor(allocation_np[rows], dtype=torch.float32, device=device)
            family_t = torch.as_tensor(np.asarray(arrays["family_weights"], dtype=np.float32)[rows], dtype=torch.float32, device=device)
            primitive_t = torch.as_tensor(np.asarray(arrays["primitive_weights"], dtype=np.float32)[rows], dtype=torch.float32, device=device)
            effect_t = torch.as_tensor(np.asarray(arrays["effect_map"], dtype=np.float32)[rows], dtype=torch.float32, device=device)
            budget_t = torch.as_tensor(np.asarray(arrays["budget_ratio"], dtype=np.float32)[rows], dtype=torch.float32, device=device)
            timestep = torch.randint(0, int(cfg.diffusion_steps), (batch_size,), device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=use_amp):
                noisy, target_noise = diffusion.q_sample(x0_t, timestep)
                predicted_noise = diffusion.denoiser(noisy, timestep, cond_t, budget_t)
                eps_loss = torch.nn.functional.mse_loss(predicted_noise, target_noise)
                alpha = diffusion.alpha_cumprod[timestep].reshape(-1, 1)
                x0_pred = ((noisy - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)).reshape_as(mask_t)
                x0_target = x0_t.reshape_as(mask_t)
                x0_loss = masked_smooth_l1(x0_pred, x0_target, mask_t)
                alloc_loss = allocation_kl_loss(alloc_t, x0_pred, mask_t)
                effect_logits = effect_logits_flat.reshape_as(mask_t)
                effect_loss = symmetric_kl_from_logits(effect_logits, torch.log(effect_t.clamp_min(1.0e-8)), mask_t)
                family_loss = categorical_kl_loss(family_t, family_logits)
                primitive_loss = categorical_kl_loss(primitive_t, primitive_logits)
                fusion_loss = symmetric_kl_from_logits(x0_pred, effect_logits, mask_t)
                smooth_loss = ((x0_pred[:, :, 1:] - x0_pred[:, :, :-1]) ** 2).mean()
                struct_loss = ((effect_logits[:, :, 1:] - effect_logits[:, :, :-1]) ** 2).mean()
                loss = (
                    float(cfg.lambda_eps) * eps_loss
                    + float(cfg.lambda_x0) * x0_loss
                    + float(cfg.lambda_alloc) * alloc_loss
                    + float(cfg.lambda_effect) * effect_loss
                    + float(cfg.lambda_family) * family_loss
                    + float(cfg.lambda_primitive) * primitive_loss
                    + float(cfg.lambda_struct) * struct_loss
                    + float(cfg.lambda_fusion) * fusion_loss
                    + float(cfg.lambda_smooth) * smooth_loss
                )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(projector.parameters()) + list(aux_heads.parameters()) + list(diffusion.parameters()),
                float(cfg.gradient_clip),
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(
                {
                    "epoch": int(epoch + 1),
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu()),
                    "eps": float(eps_loss.detach().cpu()),
                    "x0": float(x0_loss.detach().cpu()),
                    "alloc": float(alloc_loss.detach().cpu()),
                    "effect": float(effect_loss.detach().cpu()),
                    "family": float(family_loss.detach().cpu()),
                    "primitive": float(primitive_loss.detach().cpu()),
                    "struct": float(struct_loss.detach().cpu()),
                    "fusion": float(fusion_loss.detach().cpu()),
                    "smooth": float(smooth_loss.detach().cpu()),
                }
            )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector_state": projector.state_dict(),
            "aux_heads_state": aux_heads.state_dict(),
            "diffusion_state": diffusion.state_dict(),
            "condition_dim": int(condition_np.shape[1]),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(cfg.num_layers),
            "config": cfg.__dict__,
        },
        output / "target_diffusion_checkpoint.pt",
    )
    summary = {
        "pool_dir": str(Path(args.pool_dir).resolve()),
        "rows": int(row_count),
        "batch_size": int(batch_size),
        "steps": int(len(losses)),
        "last": losses[-1],
        "checkpoint": str((output / "target_diffusion_checkpoint.pt").resolve()),
    }
    (output / "training_log.json").write_text(json.dumps(losses, indent=2), encoding="utf-8")
    (output / "training_summary_zh.md").write_text(
        "# target diffusion v1 smoke 训练摘要\n\n"
        f"- rows: {row_count}\n"
        f"- steps: {len(losses)}\n"
        f"- last loss: {losses[-1]['loss']:.6f}\n"
        f"- checkpoint: {summary['checkpoint']}\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
