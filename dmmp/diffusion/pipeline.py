"""Self-contained three-stage DMMPv3 defense pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from ..utils.config import DefenseConfig
from ..data import load_cw_data
from ..encoders.leakage import profile_prefix_leakage
from ..diffusion.models import TopKLeakageEncoder, build_policy_diffusion, leakage_encoder_loss
from ..projection.padding import (
    PaddingTemplate,
    aggregate_template_stats,
    normalized_template_entropy,
    refine_counts,
    render_batch,
    renderer_options_from_config,
    target_padding_count,
)
from ..diffusion.policy import (
    analytic_leakage,
    diffusion_condition,
    encoder_feature,
    make_prior_logits,
    normalize_map,
    sample_policy_logits,
)
from ..constraints.preferences import PreferencePool, RandomPreferenceMixer, canonical_preference
from ..encoders.prefix import extract_prefix_condition, nonzero_trace
from ..utils import as_serializable_config, log, resolve_device, save_npz, set_seed, write_csv, write_json


def condition_for_trace(trace: np.ndarray, cfg: DefenseConfig):
    return extract_prefix_condition(
        trace,
        int(cfg.prefix_n),
        int(cfg.patch_num),
        max_trace_length=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        early_fraction=float(cfg.early_fraction),
    )


def load_stage1(run_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    stage_dir = run_dir / "stage1_leakage_profile"
    s_cell = np.load(stage_dir / "cell_leakage.npy")
    topk = np.load(stage_dir / "topk_candidate_mask.npy")
    payload = json.loads((stage_dir / "view_leakage.json").read_text(encoding="utf-8"))
    return s_cell.astype(np.float32), topk.astype(np.float32), payload


def select_generation_indices(splits: dict[str, np.ndarray], cfg: DefenseConfig) -> np.ndarray:
    indices = splits.get(str(cfg.generation_split), splits["test"]).astype(np.int64)
    if int(cfg.max_generation_traces) > 0 and len(indices) > int(cfg.max_generation_traces):
        rng = np.random.default_rng(int(cfg.seed) + 17)
        indices = np.asarray(sorted(rng.choice(indices, size=int(cfg.max_generation_traces), replace=False)), dtype=np.int64)
    return indices


def run_stage1(raw: np.ndarray, y: np.ndarray, splits: dict[str, np.ndarray], run_dir: Path, cfg: DefenseConfig) -> dict[str, Any]:
    stage_dir = run_dir / "stage1_leakage_profile"
    train_idx = splits["train"]
    log(
        f"Stage 1 start: train={len(train_idx)}, prefix_n={cfg.prefix_n}, patch_num={cfg.patch_num}",
        cfg.progress,
    )
    result = profile_prefix_leakage(
        raw[train_idx],
        y[train_idx],
        stage_dir,
        prefix_n=int(cfg.prefix_n),
        patch_num=int(cfg.patch_num),
        max_trace_length=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        seed=int(cfg.seed),
        topk_cells=int(cfg.topk_cells),
        mi_bins=int(cfg.mi_bins),
        masking_max_samples=int(cfg.masking_max_samples),
        command="DMMPv3 run_defense.py",
    )
    log(f"Stage 1 done: {stage_dir}", cfg.progress)
    return result


def train_encoder(
    model: TopKLeakageEncoder,
    train_features: np.ndarray,
    s_cell: np.ndarray,
    topk_mask: np.ndarray,
    cfg: DefenseConfig,
    device: torch.device,
) -> dict[str, float]:
    epochs = int(cfg.encoder_epochs)
    if epochs <= 0 or train_features.size == 0:
        log("Stage 2 encoder: skip training, using analytic leakage fallback.", cfg.progress)
        return {"loss": 0.0, "mode": "analytic_fallback_no_pretrain"}
    log(f"Stage 2 encoder: epochs={epochs}, samples={train_features.shape[0]}", cfg.progress)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.encoder_lr))
    x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    target = torch.as_tensor(np.repeat(s_cell[None, :, :], x.shape[0], axis=0), dtype=torch.float32, device=device)
    mask = torch.as_tensor(np.repeat(topk_mask[None, :, :], x.shape[0], axis=0), dtype=torch.float32, device=device)
    batch_size = min(int(cfg.batch_size), x.shape[0])
    last = {"loss": 0.0}
    for epoch in range(1, epochs + 1):
        order = torch.randperm(x.shape[0], device=device)
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            _, c_leakage = model(x[idx])
            loss, last = leakage_encoder_loss(c_leakage, target[idx], mask[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        log(f"Stage 2 encoder: epoch {epoch}/{epochs}, loss={last.get('loss', 0.0):.6f}", cfg.progress)
    model.eval()
    last["mode"] = "trained"
    return last


def _build_policy_context(
    condition,
    feature: np.ndarray,
    encoder: TopKLeakageEncoder,
    s_cell: np.ndarray,
    topk_mask: np.ndarray,
    cfg: DefenseConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if int(cfg.encoder_epochs) > 0:
        with torch.no_grad():
            feature_t = torch.as_tensor(feature.reshape(1, -1), dtype=torch.float32, device=device)
            c_global_t, c_leak_t = encoder(feature_t)
        c_global = c_global_t.detach().cpu().numpy().reshape(-1)
        c_leak = normalize_map(c_leak_t.detach().cpu().numpy().reshape(2, int(cfg.patch_num)), condition.allowed_mask)
    else:
        c_global = np.zeros(int(cfg.hidden_dim), dtype=np.float32)
        c_leak = analytic_leakage(condition, s_cell, topk_mask)
    return c_global.astype(np.float32), c_leak.astype(np.float32)


def _project_one_policy(
    trace: np.ndarray,
    label: int,
    trace_id: str,
    condition,
    feature: np.ndarray,
    s_cell: np.ndarray,
    topk_mask: np.ndarray,
    pool: PreferencePool,
    mixer: RandomPreferenceMixer,
    encoder: TopKLeakageEncoder,
    diffusion,
    rng: np.random.Generator,
    cfg: DefenseConfig,
    device: torch.device,
    budget: float,
    local_index: int,
    preference_keys: Sequence[str],
    encoder_metrics: dict[str, Any],
    diffusion_metrics: dict[str, Any],
):
    maps_all = pool.compute_all(condition, topk_mask, s_cell, condition.allowed_mask)
    maps = {key: maps_all[key] for key in preference_keys}
    pref, record = mixer.sample(maps, sample_id=str(trace_id), budget=float(budget))
    c_global, c_leak = _build_policy_context(condition, feature, encoder, s_cell, topk_mask, cfg, device)
    prior = make_prior_logits(s_cell, pref, condition.allowed_mask, rng=rng)
    if int(cfg.diffusion_train_steps) > 0:
        cond_vec = torch.as_tensor(diffusion_condition(c_global, c_leak, pref, float(budget)).reshape(1, -1), dtype=torch.float32, device=device)
        budget_tensor = torch.as_tensor([float(budget)], dtype=torch.float32, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(cfg.seed) + int(local_index) + int(round(float(budget) * 10000)))
        sampled = sample_policy_logits(diffusion, cond_vec, budget_tensor, sampling_steps=int(cfg.sampling_steps), generator=generator)
        logits = 0.50 * sampled.detach().cpu().numpy().reshape(2, int(cfg.patch_num)) + 0.50 * prior
    else:
        logits = prior
    metadata = {
        "trace_id": str(trace_id),
        "label": int(label),
        "preference_subset": record["subset"],
        "preference_weights": record["weights"],
        "preference_subset_label": record["subset_label"],
        "encoder_mode": encoder_metrics.get("mode", ""),
        "diffusion_mode": diffusion_metrics.get("mode", ""),
    }
    template = project_policy_to_template_local(logits, condition, trace, float(budget), c_leak, pref, metadata)
    return template, logits, c_leak, pref, record


def project_policy_to_template_local(
    logits: np.ndarray,
    condition,
    trace: np.ndarray,
    budget: float,
    c_leak: np.ndarray,
    pref: np.ndarray,
    metadata: dict[str, Any],
) -> PaddingTemplate:
    from ..projection.padding import project_policy_to_template

    return project_policy_to_template(
        logits,
        condition,
        trace,
        float(budget),
        c_leakage=c_leak,
        preference=pref,
        method="topk_random_preference_mixer",
        metadata=metadata,
    )


def run_stage2(
    raw: np.ndarray,
    y: np.ndarray,
    trace_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    run_dir: Path,
    cfg: DefenseConfig,
    device: torch.device,
) -> dict[str, Any]:
    stage_dir = run_dir / "stage2_generation"
    stage_dir.mkdir(parents=True, exist_ok=True)
    log("Stage 2 start: random preference-guided policy generation.", cfg.progress)
    s_cell, topk_mask, _ = load_stage1(run_dir)
    budgets = cfg.budget_values
    preference_keys = [canonical_preference(item) for item in cfg.preference_values]
    rng = np.random.default_rng(int(cfg.seed) + 2000)
    pool = PreferencePool(patch_num=int(cfg.patch_num))
    mixer = RandomPreferenceMixer(combination_sizes=cfg.combination_size_values, seed=int(cfg.seed) + 3000)

    gen_indices = select_generation_indices(splits, cfg)
    gen_raw, gen_y, gen_ids = raw[gen_indices], y[gen_indices], trace_ids[gen_indices]
    log(f"Stage 2: build generation conditions, split={cfg.generation_split}, traces={len(gen_raw)}", cfg.progress)
    conditions = [condition_for_trace(trace, cfg) for trace in gen_raw]

    train_indices = splits["train"][: min(len(splits["train"]), int(cfg.encoder_train_samples))]
    train_conditions = [condition_for_trace(trace, cfg) for trace in raw[train_indices]]
    train_features = np.stack([encoder_feature(cond, topk_mask, s_cell) for cond in train_conditions], axis=0) if train_conditions else np.zeros((0, 1), dtype=np.float32)
    input_dim = int(train_features.shape[1]) if train_features.size else int(conditions[0].vector.size + 4 * int(cfg.patch_num))
    encoder = TopKLeakageEncoder(input_dim=input_dim, patch_num=int(cfg.patch_num), hidden_dim=int(cfg.hidden_dim)).to(device)
    encoder_metrics = train_encoder(encoder, train_features, s_cell, topk_mask, cfg, device)
    torch.save(
        {
            "model_state": encoder.state_dict(),
            "config": {"input_dim": input_dim, "patch_num": int(cfg.patch_num), "hidden_dim": int(cfg.hidden_dim), "prefix_n": int(cfg.prefix_n), "seed": int(cfg.seed)},
            "metrics": encoder_metrics,
        },
        stage_dir / "encoder_checkpoint.pt",
    )

    condition_dim = int(cfg.hidden_dim) + 2 * int(cfg.patch_num) + 2 * int(cfg.patch_num) + 1
    diffusion = build_policy_diffusion(condition_dim, int(cfg.patch_num), int(cfg.hidden_dim), int(cfg.diffusion_steps)).to(device)
    diffusion_metrics = {"denoising_loss": 0.0, "mode": "prior_only_no_pretrain"}
    if int(cfg.diffusion_train_steps) > 0 and train_features.size:
        log(f"Stage 2 diffusion: steps={cfg.diffusion_train_steps}, batch={cfg.batch_size}", cfg.progress)
        opt = torch.optim.AdamW(diffusion.parameters(), lr=float(cfg.diffusion_lr))
        diffusion.train()
        for step in range(int(cfg.diffusion_train_steps)):
            batch_n = min(int(cfg.batch_size), len(train_conditions))
            chosen = rng.choice(len(train_conditions), size=batch_n, replace=len(train_conditions) < batch_n)
            cond_vecs, priors, budget_values = [], [], []
            for item in chosen.tolist():
                cond = train_conditions[item]
                maps_all = pool.compute_all(cond, topk_mask, s_cell, cond.allowed_mask)
                maps = {key: maps_all[key] for key in preference_keys}
                budget = float(rng.choice(budgets))
                pref, _ = mixer.sample(maps, sample_id=f"train_{step}_{item}", budget=budget)
                c_leak = analytic_leakage(cond, s_cell, topk_mask)
                c_global = np.zeros(int(cfg.hidden_dim), dtype=np.float32)
                cond_vecs.append(diffusion_condition(c_global, c_leak, pref, budget))
                priors.append(make_prior_logits(s_cell, pref, cond.allowed_mask, rng=rng).reshape(-1))
                budget_values.append(budget)
            x0 = torch.as_tensor(np.stack(priors, axis=0), dtype=torch.float32, device=device)
            cond_tensor = torch.as_tensor(np.stack(cond_vecs, axis=0), dtype=torch.float32, device=device)
            budget_tensor = torch.as_tensor(np.asarray(budget_values), dtype=torch.float32, device=device)
            loss = diffusion.training_loss(x0, cond_tensor, budget_tensor)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            diffusion_metrics = {"denoising_loss": float(loss.detach().cpu()), "mode": "trained"}
            step_no = step + 1
            if step_no == 1 or step_no == int(cfg.diffusion_train_steps) or step_no % max(1, int(cfg.log_every)) == 0:
                log(f"Stage 2 diffusion: step {step_no}/{cfg.diffusion_train_steps}, loss={diffusion_metrics['denoising_loss']:.6f}", cfg.progress)
        diffusion.eval()
    else:
        log("Stage 2 diffusion: skip training, using x0 prior fallback.", cfg.progress)
    torch.save(
        {
            "model_state": diffusion.state_dict(),
            "config": {"condition_dim": condition_dim, "patch_num": int(cfg.patch_num), "hidden_dim": int(cfg.hidden_dim), "diffusion_steps": int(cfg.diffusion_steps), "sampling_steps": int(cfg.sampling_steps), "seed": int(cfg.seed)},
            "metrics": diffusion_metrics,
        },
        stage_dir / "diffusion_checkpoint.pt",
    )

    templates: list[PaddingTemplate] = []
    logits_rows, count_rows, c_rows, pref_rows, allowed_rows = [], [], [], [], []
    budget_rows, trace_index_rows, label_rows, id_rows = [], [], [], []
    records = []
    total = len(budgets) * len(conditions)
    done = 0
    with torch.no_grad():
        for budget in budgets:
            for local_index, (trace, label, trace_id, cond) in enumerate(zip(gen_raw, gen_y, gen_ids, conditions)):
                feature = encoder_feature(cond, topk_mask, s_cell)
                template, logits, c_leak, pref, record = _project_one_policy(
                    trace,
                    int(label),
                    str(trace_id),
                    cond,
                    feature,
                    s_cell,
                    topk_mask,
                    pool,
                    mixer,
                    encoder,
                    diffusion,
                    rng,
                    cfg,
                    device,
                    float(budget),
                    local_index,
                    preference_keys,
                    encoder_metrics,
                    diffusion_metrics,
                )
                templates.append(template)
                records.append({**record, **template.metadata})
                logits_rows.append(logits.astype(np.float32))
                count_rows.append(template.counts.astype(np.int32))
                c_rows.append(c_leak.astype(np.float32))
                pref_rows.append(pref.astype(np.float32))
                allowed_rows.append(np.asarray(cond.allowed_mask, dtype=np.float32))
                budget_rows.append(float(budget))
                trace_index_rows.append(int(gen_indices[local_index]))
                label_rows.append(int(label))
                id_rows.append(str(trace_id))
                done += 1
                if done == 1 or done == total or done % max(1, int(cfg.log_every)) == 0:
                    log(f"Stage 2 generate: {done}/{total} policies complete", cfg.progress)

    clean = raw[np.asarray(trace_index_rows, dtype=np.int64)]
    defended, stats = render_batch(
        clean,
        templates,
        seed=int(cfg.seed) + 5000,
        max_trace_length=int(cfg.max_trace_length),
        **renderer_options_from_config(cfg),
    )
    save_npz(
        stage_dir / "defended_traces" / "defended_all.npz",
        X=defended.astype(np.float32),
        y=np.asarray(label_rows, dtype=np.int64),
        trace_id=np.asarray(id_rows).astype(str),
        budget=np.asarray(budget_rows, dtype=np.float32),
        clean_index=np.asarray(trace_index_rows, dtype=np.int64),
    )
    np.save(stage_dir / "generated_policies.npy", np.asarray(count_rows, dtype=np.int32))
    np.save(stage_dir / "generated_policy_logits.npy", np.asarray(logits_rows, dtype=np.float32))
    np.save(stage_dir / "c_leakage_maps.npy", np.asarray(c_rows, dtype=np.float32))
    np.save(stage_dir / "preference_maps.npy", np.asarray(pref_rows, dtype=np.float32))
    np.save(stage_dir / "allowed_masks.npy", np.asarray(allowed_rows, dtype=np.float32))
    np.save(stage_dir / "policy_budgets.npy", np.asarray(budget_rows, dtype=np.float32))
    np.save(stage_dir / "generation_trace_indices.npy", np.asarray(trace_index_rows, dtype=np.int64))
    np.save(stage_dir / "generation_labels.npy", np.asarray(label_rows, dtype=np.int64))
    mixer.save(stage_dir / "preference_records.json")
    write_json(stage_dir / "generation_records.json", records)
    metrics = aggregate_template_stats(stats, templates)
    if count_rows:
        metrics["policy_diversity_std"] = float(np.mean(np.std(np.asarray(count_rows, dtype=np.float32).reshape(len(count_rows), -1), axis=0)))
    else:
        metrics["policy_diversity_std"] = 0.0
    metrics["preference_combination_entropy"] = mixer.stats()["combination_entropy"]
    write_json(stage_dir / "stage2_metrics.json", metrics)
    (stage_dir / "summary_zh.md").write_text(
        "\n".join(
            [
                "# Stage 2: Random preference-guided policy generation",
                "",
                f"- encoder loss: {float(encoder_metrics.get('loss', 0.0)):.6f} ({encoder_metrics.get('mode', '')})",
                f"- diffusion loss: {float(diffusion_metrics.get('denoising_loss', 0.0)):.6f} ({diffusion_metrics.get('mode', '')})",
                f"- bandwidth overhead: {metrics['bandwidth_overhead']:.6f}",
                f"- allowed violation: {metrics['allowed_mask_violation_rate']:.8f}",
                f"- generated traces: {metrics['generated_traces']}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"Stage 2 done: {stage_dir / 'stage2_metrics.json'}", cfg.progress)
    return metrics


def run_stage3(raw: np.ndarray, y: np.ndarray, splits: dict[str, np.ndarray], run_dir: Path, cfg: DefenseConfig) -> dict[str, Any]:
    stage2_dir = run_dir / "stage2_generation"
    stage3_dir = run_dir / "stage3_refinement"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    log("Stage 3 start: policy shrink/refinement.", cfg.progress)
    counts = np.load(stage2_dir / "generated_policies.npy")
    c_maps = np.load(stage2_dir / "c_leakage_maps.npy")
    pref_maps = np.load(stage2_dir / "preference_maps.npy")
    allowed = np.load(stage2_dir / "allowed_masks.npy")
    budgets = np.load(stage2_dir / "policy_budgets.npy")
    trace_indices = np.load(stage2_dir / "generation_trace_indices.npy")
    labels = np.load(stage2_dir / "generation_labels.npy")
    clean = raw[trace_indices.astype(np.int64)]
    refined_rows, templates, report_rows = [], [], []
    for index, (count, c_leak, pref, mask, budget, trace) in enumerate(zip(counts, c_maps, pref_maps, allowed, budgets, clean)):
        refined, report = refine_counts(count, c_leak, pref, method=str(cfg.shrink_method), keep_ratio=float(cfg.shrink_keep_ratio))
        violation = int(refined[np.asarray(mask) <= 0].sum())
        target = target_padding_count(trace, float(budget))
        template = PaddingTemplate(
            counts=refined.astype(np.int32),
            target_n_pad=int(target),
            actual_n_pad=int(refined.sum()),
            target_bandwidth=float(budget),
            metadata={
                "method": f"topk_random_preference_mixer_shrink_{cfg.shrink_method}",
                "allowed_violation_count": violation,
                "allowed_violation_rate": float(violation / max(int(refined.sum()), 1)),
                "template_entropy": normalized_template_entropy(refined),
                "budget": float(budget),
            },
        )
        refined_rows.append(refined.astype(np.int32))
        templates.append(template)
        report_rows.append(
            {
                "index": int(index),
                "budget": float(budget),
                "packets_before": int(count.sum()),
                "packets_after": int(refined.sum()),
                "bandwidth_before": float(count.sum() / max(nonzero_trace(trace).size, 1)),
                "bandwidth_after": float(refined.sum() / max(nonzero_trace(trace).size, 1)),
                "allowed_violation_rate": float(template.metadata["allowed_violation_rate"]),
                **report,
            }
        )
        done = index + 1
        if done == 1 or done == len(counts) or done % max(1, int(cfg.log_every)) == 0:
            log(f"Stage 3 refine: {done}/{len(counts)} policies complete", cfg.progress)
    defended, stats = render_batch(
        clean,
        templates,
        seed=int(cfg.seed) + 7000,
        max_trace_length=int(cfg.max_trace_length),
        **renderer_options_from_config(cfg),
    )
    save_npz(
        stage3_dir / "refined_defended_traces" / "refined_all.npz",
        X=defended.astype(np.float32),
        y=labels.astype(np.int64),
        clean_index=trace_indices.astype(np.int64),
        budget=budgets.astype(np.float32),
    )
    np.save(stage3_dir / "refined_policies.npy", np.asarray(refined_rows, dtype=np.int32))
    write_csv(stage3_dir / "shrink_report.csv", report_rows)
    template_metrics = aggregate_template_stats(stats, templates)
    shrink_before = float(np.mean([row["bandwidth_before"] for row in report_rows])) if report_rows else 0.0
    shrink_after = float(np.mean([row["bandwidth_after"] for row in report_rows])) if report_rows else 0.0
    metrics = {
        "bandwidth_before": shrink_before,
        "bandwidth_after": shrink_after,
        "shrink_ratio": float(1.0 - shrink_after / max(shrink_before, 1e-12)) if shrink_before > 0 else 0.0,
        "allowed_mask_violation_rate": template_metrics["allowed_mask_violation_rate"],
        "better_method": str(cfg.shrink_method),
    }
    write_json(stage3_dir / "stage3_metrics.json", metrics)
    (stage3_dir / "summary_zh.md").write_text(
        "\n".join(
            [
                "# Stage 3: Budget-performance-aware policy refinement",
                "",
                f"- bandwidth before: {metrics['bandwidth_before']:.6f}",
                f"- bandwidth after: {metrics['bandwidth_after']:.6f}",
                f"- shrink ratio: {metrics['shrink_ratio']:.6f}",
                f"- allowed violation: {metrics['allowed_mask_violation_rate']:.8f}",
            ]
        ),
        encoding="utf-8",
    )
    log(f"Stage 3 done: {stage3_dir / 'stage3_metrics.json'}", cfg.progress)
    return metrics


def load_trained_defense(run_dir: Path, cfg: DefenseConfig, device: torch.device):
    s_cell, topk_mask, _ = load_stage1(run_dir)
    stage2_dir = run_dir / "stage2_generation"
    encoder_payload = torch.load(stage2_dir / "encoder_checkpoint.pt", map_location=device, weights_only=False)
    encoder_cfg = encoder_payload.get("config", {})
    encoder = TopKLeakageEncoder(
        input_dim=int(encoder_cfg.get("input_dim", 1)),
        patch_num=int(encoder_cfg.get("patch_num", cfg.patch_num)),
        hidden_dim=int(encoder_cfg.get("hidden_dim", cfg.hidden_dim)),
    ).to(device)
    encoder.load_state_dict(encoder_payload["model_state"])
    encoder.eval()

    diffusion_payload = torch.load(stage2_dir / "diffusion_checkpoint.pt", map_location=device, weights_only=False)
    diffusion_cfg = diffusion_payload.get("config", {})
    diffusion = build_policy_diffusion(
        int(diffusion_cfg.get("condition_dim", int(cfg.hidden_dim) + 4 * int(cfg.patch_num) + 1)),
        int(diffusion_cfg.get("patch_num", cfg.patch_num)),
        int(diffusion_cfg.get("hidden_dim", cfg.hidden_dim)),
        int(diffusion_cfg.get("diffusion_steps", cfg.diffusion_steps)),
    ).to(device)
    diffusion.load_state_dict(diffusion_payload["model_state"])
    diffusion.eval()
    return s_cell, topk_mask, encoder, diffusion, encoder_payload.get("metrics", {}), diffusion_payload.get("metrics", {})


def generate_defended_dataset_from_pool(
    raw: np.ndarray,
    y: np.ndarray,
    trace_ids: np.ndarray,
    indices: np.ndarray,
    run_dir: Path,
    cfg: DefenseConfig,
    *,
    defense_seed: int,
    output_npz: str | Path | None = None,
    policy_variant: str = "stage3",
    budget: float | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate a fresh defended set from the trained DMMPv3 defense pool."""

    device = device or resolve_device(str(cfg.device))
    local_cfg = replace(cfg, seed=int(defense_seed))
    s_cell, topk_mask, encoder, diffusion, encoder_metrics, diffusion_metrics = load_trained_defense(run_dir, cfg, device)
    rng = np.random.default_rng(int(defense_seed) + 2000)
    pool = PreferencePool(patch_num=int(cfg.patch_num))
    mixer = RandomPreferenceMixer(combination_sizes=cfg.combination_size_values, seed=int(defense_seed) + 3000)
    preference_keys = [canonical_preference(item) for item in cfg.preference_values]
    chosen_budget = float(budget if budget is not None else cfg.budget_values[0])
    selected = np.asarray(indices, dtype=np.int64)
    clean = raw[selected]
    labels = y[selected].astype(np.int64)
    ids = trace_ids[selected]

    templates: list[PaddingTemplate] = []
    for local_index, (trace, label, trace_id) in enumerate(zip(clean, labels, ids)):
        cond = condition_for_trace(trace, cfg)
        feature = encoder_feature(cond, topk_mask, s_cell)
        template, _, c_leak, pref, _ = _project_one_policy(
            trace,
            int(label),
            str(trace_id),
            cond,
            feature,
            s_cell,
            topk_mask,
            pool,
            mixer,
            encoder,
            diffusion,
            rng,
            local_cfg,
            device,
            chosen_budget,
            local_index,
            preference_keys,
            encoder_metrics,
            diffusion_metrics,
        )
        if str(policy_variant).lower() == "stage3":
            refined, report = refine_counts(
                template.counts,
                c_leak,
                pref,
                method=str(cfg.shrink_method),
                keep_ratio=float(cfg.shrink_keep_ratio),
            )
            template = PaddingTemplate(
                counts=refined.astype(np.int32),
                target_n_pad=template.target_n_pad,
                actual_n_pad=int(refined.sum()),
                target_bandwidth=float(chosen_budget),
                metadata={
                    **template.metadata,
                    **report,
                    "method": f"{template.metadata.get('method', 'DMMPv3')}_fresh_stage3",
                    "template_entropy": normalized_template_entropy(refined),
                },
            )
        templates.append(template)
        done = local_index + 1
        if done == 1 or done == len(selected) or done % max(1, int(cfg.log_every)) == 0:
            log(f"Fresh defense sampling: {done}/{len(selected)} traces complete", cfg.progress)

    defended, stats = render_batch(
        clean,
        templates,
        seed=int(defense_seed) + 7000,
        max_trace_length=int(cfg.max_trace_length),
        **renderer_options_from_config(cfg),
    )
    metrics = aggregate_template_stats(stats, templates)
    metrics.update({"defense_seed": int(defense_seed), "budget": chosen_budget, "policy_variant": str(policy_variant)})
    if output_npz is not None:
        save_npz(
            output_npz,
            X=defended.astype(np.float32),
            y=labels.astype(np.int64),
            clean_index=selected.astype(np.int64),
            budget=np.full(labels.shape[0], chosen_budget, dtype=np.float32),
        )
        write_json(Path(output_npz).with_name(Path(output_npz).stem + "_metrics.json"), metrics)
    return defended.astype(np.float32), labels.astype(np.int64), metrics


def write_final_summary(run_dir: Path, cfg: DefenseConfig, data_source: str, stage_metrics: dict[str, Any]) -> None:
    lines = [
        "# DMMPv3 Top-K random preference diffusion defense",
        "",
        f"- dataset: CW ({data_source})",
        f"- seed: {int(cfg.seed)}",
        f"- prefix_n: {int(cfg.prefix_n)}",
        f"- patch_num: {int(cfg.patch_num)}",
        f"- budgets: {cfg.budgets}",
        "",
        "## Stage Summary",
        "",
    ]
    if (run_dir / "stage1_leakage_profile" / "topk_views.json").is_file():
        topk = json.loads((run_dir / "stage1_leakage_profile" / "topk_views.json").read_text(encoding="utf-8"))
        lines.append("- Stage 1 leakage ranking: " + ", ".join(f"{row['view']}={row['score']:.4f}" for row in topk["ranked_views"]))
    if "stage2" in stage_metrics:
        m = stage_metrics["stage2"]
        lines.append(f"- Stage 2: bandwidth={m['bandwidth_overhead']:.4f}, violation={m['allowed_mask_violation_rate']:.8f}, traces={m['generated_traces']}")
    if "stage3" in stage_metrics:
        m = stage_metrics["stage3"]
        lines.append(f"- Stage 3: bandwidth {m['bandwidth_before']:.4f}->{m['bandwidth_after']:.4f}, shrink={m['shrink_ratio']:.4f}")
    lines.extend(
        [
            "",
            "## Decoupling Note",
            "",
            "- DMMPv3 does not import project-level `experiments`, `defenses`, `models`, or old `defence/DMMP` modules.",
            "- Prefix condition extraction, padding projection/rendering, preference pool, diffusion model, and refinement are implemented inside `defence/DMMPv3/dmmp`.",
        ]
    )
    (run_dir / "final_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")


def run_defense_pipeline(cfg: DefenseConfig) -> Path:
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    log(f"Load data start: data_root={cfg.data_root}, seed={cfg.seed}", cfg.progress)
    raw, y, trace_ids, splits, data_source = load_cw_data(cfg)
    log(
        f"Load data done: samples={len(y)}, classes={len(np.unique(y))}, train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, device={device}",
        cfg.progress,
    )
    run_name = cfg.run_name or f"dmmpv3_cw_seed{int(cfg.seed)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg.output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"DMMPv3 refuses to overwrite a non-empty result directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_config.json",
        {
            **as_serializable_config(cfg),
            "device_resolved": str(device),
            "data_source": data_source,
            "num_samples": int(len(y)),
            "num_classes": int(len(np.unique(y))),
            "split_sizes": {key: int(len(value)) for key, value in splits.items()},
            "implementation": "defence/DMMPv3",
        },
    )
    write_json(run_dir / "split_indices.json", {key: value.astype(int).tolist() for key, value in splits.items()})

    stage_metrics: dict[str, Any] = {}
    if cfg.stage in {"1", "all"}:
        run_stage1(raw, y, splits, run_dir, cfg)
    if cfg.stage in {"2", "all"}:
        if not (run_dir / "stage1_leakage_profile" / "cell_leakage.npy").is_file():
            run_stage1(raw, y, splits, run_dir, cfg)
        stage_metrics["stage2"] = run_stage2(raw, y, trace_ids, splits, run_dir, cfg, device)
    if cfg.stage in {"3", "all"}:
        if not (run_dir / "stage2_generation" / "generated_policies.npy").is_file():
            stage_metrics["stage2"] = run_stage2(raw, y, trace_ids, splits, run_dir, cfg, device)
        stage_metrics["stage3"] = run_stage3(raw, y, splits, run_dir, cfg)
    write_final_summary(run_dir, cfg, data_source, stage_metrics)
    log(f"[done] results saved to: {run_dir}", cfg.progress)
    return run_dir

