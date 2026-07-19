"""Executable future-cell utility supervision and candidate scoring for DMMPv3 V4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from ..data import choose_stratified_subset
from ..encoders.leakage import build_view_features, score_feature_matrix
from ..encoders.prefix import extract_prefix_condition, nonzero_trace, tam_patch_ids
from ..guidance.strong_surrogates import (
    ensemble_finite_difference_maps,
    ensemble_utility_maps,
    load_strong_surrogates,
    train_strong_surrogates,
)
from ..utils import log, write_json


VIEW_NAMES = ("V_raw", "V_count", "V_interval", "V_burst", "V_rate", "V_cumul", "V_patch")


def full_patch_counts(
    raw: np.ndarray,
    patch_num: int = 200,
    max_trace_length: int = 5000,
    *,
    max_load_time: float = 80.0,
) -> np.ndarray:
    rows = np.asarray(raw)
    result = np.zeros((len(rows), 2, int(patch_num)), dtype=np.float32)
    for row_index, trace in enumerate(rows):
        values = nonzero_trace(trace)[: int(max_trace_length)]
        if values.size == 0:
            continue
        patch_ids = tam_patch_ids(values, int(patch_num), max_load_time=float(max_load_time))
        np.add.at(result[row_index, 0], patch_ids[values > 0], 1.0)
        np.add.at(result[row_index, 1], patch_ids[values < 0], 1.0)
    return result


def sample_allowed_masks(raw: np.ndarray, cfg) -> np.ndarray:
    return np.stack(
        [
            extract_prefix_condition(
                trace,
                int(cfg.prefix_n),
                int(cfg.patch_num),
                max_trace_length=int(cfg.max_trace_length),
                max_load_time=float(cfg.surrogate_rf_max_load_time),
                early_fraction=float(cfg.early_fraction),
            ).allowed_mask
            for trace in raw
        ],
        axis=0,
    ).astype(np.float32)


def _centroid_cv(features: np.ndarray, labels: np.ndarray, folds: int, seed: int) -> tuple[float, float]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    classes = np.unique(y)
    rng = np.random.default_rng(int(seed))
    assignments = np.zeros(len(y), dtype=np.int64)
    for label in classes:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        assignments[idx] = np.arange(len(idx), dtype=np.int64) % max(int(folds), 2)
    correct = 0
    log_loss_sum = 0.0
    total = 0
    for fold in range(max(int(folds), 2)):
        train = assignments != fold
        test = ~train
        if not np.any(test):
            continue
        mean = x[train].mean(axis=0, keepdims=True)
        scale = x[train].std(axis=0, keepdims=True) + 1e-4
        train_x = np.clip((x[train] - mean) / scale, -8.0, 8.0)
        test_x = np.clip((x[test] - mean) / scale, -8.0, 8.0)
        centroids = np.stack([train_x[y[train] == label].mean(axis=0) for label in classes], axis=0)
        true_pos = {int(label): pos for pos, label in enumerate(classes.tolist())}
        test_y = y[test]
        for start in range(0, len(test_x), 256):
            batch = test_x[start : start + 256]
            distance = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + np.sum(centroids * centroids, axis=1)[None, :]
                - 2.0 * batch @ centroids.T
            ) / max(batch.shape[1], 1)
            logits = -distance
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(np.clip(logits, -40.0, 0.0))
            probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            truth = np.asarray([true_pos[int(item)] for item in test_y[start : start + len(batch)]], dtype=np.int64)
            correct += int(np.sum(np.argmax(probs, axis=1) == truth))
            log_loss_sum += float(-np.log(np.maximum(probs[np.arange(len(batch)), truth], 1e-12)).sum())
            total += len(batch)
    return float(correct / max(total, 1)), float(log_loss_sum / max(total, 1))


def profile_views_v4(raw: np.ndarray, labels: np.ndarray, cfg, output_dir: str | Path) -> dict:
    max_samples = min(int(cfg.view_profile_samples), len(labels)) if int(cfg.view_profile_samples) > 0 else len(labels)
    subset = choose_stratified_subset(labels, max_samples, int(cfg.seed) + 101)
    x = np.asarray(raw[subset])
    y = np.asarray(labels[subset], dtype=np.int64)
    features = build_view_features(
        x,
        int(cfg.prefix_n),
        int(cfg.patch_num),
        max_trace_length=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
    )
    chance = 1.0 / max(len(np.unique(y)), 1)
    random_log_loss = math.log(max(len(np.unique(y)), 2))
    rows = []
    for name in VIEW_NAMES:
        mi_score, _ = score_feature_matrix(features[name], y, n_bins=int(cfg.mi_bins))
        accuracy, log_loss = _centroid_cv(features[name], y, int(cfg.view_cv_folds), int(cfg.seed) + 102)
        acc_gain = float(np.clip((accuracy - chance) / max(1.0 - chance, 1e-8), 0.0, 1.0))
        ll_gain = float(np.clip((random_log_loss - log_loss) / max(random_log_loss, 1e-8), 0.0, 1.0))
        normalized_mi = float(np.clip(mi_score, 0.0, 1.0))
        rows.append(
            {
                "view": name,
                "normalized_mi_proxy": normalized_mi,
                "cv_accuracy": float(accuracy),
                "cv_log_loss": float(log_loss),
                "classification_gain": acc_gain,
                "log_loss_gain": ll_gain,
                "view_score": float(0.34 * normalized_mi + 0.33 * acc_gain + 0.33 * ll_gain),
            }
        )
    rows.sort(key=lambda row: row["view_score"], reverse=True)
    payload = {"samples": int(len(y)), "folds": int(cfg.view_cv_folds), "rows": rows, "selected_views": [row["view"] for row in rows[:3]]}
    write_json(Path(output_dir) / "view_profile.json", payload)
    return payload


def build_candidate_features(raw: np.ndarray, cfg, selected_views: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conditions = [
        extract_prefix_condition(
            trace,
            int(cfg.prefix_n),
                int(cfg.patch_num),
                max_trace_length=int(cfg.max_trace_length),
                max_load_time=float(cfg.surrogate_rf_max_load_time),
                early_fraction=float(cfg.early_fraction),
            )
        for trace in raw
    ]
    view_features = build_view_features(
        np.asarray(raw),
        int(cfg.prefix_n),
        int(cfg.patch_num),
        max_trace_length=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
    )
    rows = []
    structures = []
    for index, condition in enumerate(conditions):
        selected = [np.asarray(view_features[name][index], dtype=np.float32).reshape(-1) for name in selected_views]
        structure = np.concatenate(selected, axis=0).astype(np.float32)
        rows.append(np.concatenate([condition.vector, structure], axis=0).astype(np.float32))
        structures.append(structure)
    masks = np.stack([condition.allowed_mask for condition in conditions], axis=0).astype(np.float32)
    return np.stack(rows, axis=0), masks, np.stack(structures, axis=0)


class CandidateScorer(nn.Module):
    def __init__(self, input_dim: int, patch_num: int, hidden_dim: int = 384):
        super().__init__()
        self.input_dim = int(input_dim)
        self.patch_num = int(patch_num)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 2 * self.patch_num),
        )

    def forward(self, features: torch.Tensor, allowed_mask: torch.Tensor) -> torch.Tensor:
        scores = nn.functional.softplus(self.net(features.float())).reshape(-1, 2, self.patch_num)
        return scores * allowed_mask.float()


def soft_topk_mask(scores: torch.Tensor, allowed_mask: torch.Tensor, k: int, temperature: float, hard: bool = False) -> torch.Tensor:
    flat_scores = scores.reshape(scores.shape[0], -1)
    flat_allowed = allowed_mask.reshape(allowed_mask.shape[0], -1)
    if hard:
        masked = flat_scores.masked_fill(flat_allowed <= 0, -1e9)
        indices = torch.topk(masked, k=min(int(k), masked.shape[1]), dim=1).indices
        result = torch.zeros_like(masked)
        result.scatter_(1, indices, 1.0)
        return result.reshape_as(scores) * allowed_mask
    masked = flat_scores.masked_fill(flat_allowed <= 0, -1e9)
    probabilities = torch.softmax(masked / max(float(temperature), 1e-3), dim=1)
    return torch.clamp(probabilities * float(k), 0.0, 1.0).reshape_as(scores) * allowed_mask


def train_candidate_scorer(features: np.ndarray, masks: np.ndarray, utility: np.ndarray, cfg, device: torch.device, output_dir: str | Path, selected_views: Sequence[str]) -> tuple[CandidateScorer, np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0, keepdims=True).astype(np.float32)
    feature_scale = (features.std(axis=0, keepdims=True) + 1e-4).astype(np.float32)
    normalized_features = np.clip((features - feature_mean) / feature_scale, -8.0, 8.0).astype(np.float32)
    model = CandidateScorer(features.shape[1], int(cfg.patch_num), int(cfg.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(int(cfg.seed) + 301)
    batch_size = min(int(cfg.batch_size), len(features))
    last_loss = 0.0
    for epoch in range(1, int(cfg.candidate_epochs) + 1):
        order = rng.permutation(len(features))
        model.train()
        total_batches = int((len(order) + max(int(batch_size), 1) - 1) // max(int(batch_size), 1))
        heartbeat = max(1, min(max(int(getattr(cfg, "log_every", 100)), 1), max(total_batches // 4, 1)))
        for batch_index, start in enumerate(range(0, len(order), batch_size), start=1):
            idx = order[start : start + batch_size]
            xb = torch.as_tensor(normalized_features[idx], dtype=torch.float32, device=device)
            mb = torch.as_tensor(masks[idx], dtype=torch.float32, device=device)
            target = torch.as_tensor(utility[idx], dtype=torch.float32, device=device)
            pred = model(xb, mb)
            mse = nn.functional.mse_loss(pred, target)
            cosine = 1.0 - nn.functional.cosine_similarity(pred.reshape(len(idx), -1), target.reshape(len(idx), -1), dim=1).mean()
            loss = mse + 0.10 * cosine
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
            if batch_index == 1 or batch_index == total_batches or batch_index % heartbeat == 0:
                log(
                    f"V4 Stage 1 candidate scorer: epoch {epoch}/{cfg.candidate_epochs}, "
                    f"batch {batch_index}/{total_batches}, loss={last_loss:.6f}",
                    cfg.progress,
                )
        log(f"V4 Stage 1 candidate scorer: epoch {epoch}/{cfg.candidate_epochs}, loss={last_loss:.6f}", cfg.progress)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "input_dim": int(features.shape[1]),
                "patch_num": int(cfg.patch_num),
                "hidden_dim": int(cfg.hidden_dim),
                "selected_views": list(selected_views),
                "feature_mean": feature_mean,
                "feature_scale": feature_scale,
            },
            "metrics": {"last_loss": last_loss, "samples": int(len(features))},
        },
        Path(output_dir) / "candidate_scorer_checkpoint.pt",
    )
    model.eval()
    return model, feature_mean, feature_scale


def _rank_metrics(exact: np.ndarray, predicted: np.ndarray, masks: np.ndarray, k: int) -> dict[str, float]:
    correlations, overlaps, ndcgs = [], [], []
    for truth, pred, mask in zip(exact, predicted, masks):
        valid = mask.reshape(-1) > 0
        a = truth.reshape(-1)[valid]
        b = pred.reshape(-1)[valid]
        if len(a) < 2:
            continue
        ar = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort").astype(np.float64)
        br = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort").astype(np.float64)
        if ar.std() > 0 and br.std() > 0:
            correlations.append(float(np.corrcoef(ar, br)[0, 1]))
        kk = min(int(k), len(a))
        top_a = set(np.argsort(-a, kind="mergesort")[:kk].tolist())
        top_b = set(np.argsort(-b, kind="mergesort")[:kk].tolist())
        overlaps.append(float(len(top_a & top_b) / max(kk, 1)))
        order = np.argsort(-b, kind="mergesort")[:kk]
        ideal = np.argsort(-a, kind="mergesort")[:kk]
        discounts = 1.0 / np.log2(np.arange(2, kk + 2))
        dcg = float(np.sum((2.0 ** a[order] - 1.0) * discounts))
        idcg = float(np.sum((2.0 ** a[ideal] - 1.0) * discounts))
        ndcgs.append(float(dcg / max(idcg, 1e-12)))
    return {
        "spearman": float(np.mean(correlations)) if correlations else 0.0,
        "topk_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def run_candidate_stage(raw: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, val_indices: np.ndarray, run_dir: Path, cfg, device: torch.device) -> dict:
    stage_dir = run_dir / "stage1_executable_condition"
    stage_dir.mkdir(parents=True, exist_ok=True)
    train_raw = raw[train_indices]
    train_y = labels[train_indices]
    log("[Stage 1/3] Profiling leakage views...", cfg.progress)
    view_profile = profile_views_v4(train_raw, train_y, cfg, stage_dir)
    selected_views = view_profile["selected_views"]
    log("[Stage 1/3] Training strong ProjectDF/ProjectRF surrogate ensemble...", cfg.progress)
    bundle = train_strong_surrogates(raw, labels, train_indices, val_indices, cfg, device, stage_dir)
    scorer_count = min(int(cfg.probe_samples), len(train_y)) if int(cfg.probe_samples) > 0 else len(train_y)
    scorer_local = choose_stratified_subset(train_y, scorer_count, int(cfg.seed) + 302)
    scorer_raw = train_raw[scorer_local]
    scorer_y = train_y[scorer_local]
    features, masks, structures = build_candidate_features(scorer_raw, cfg, selected_views)
    log(f"[Stage 1/3] Computing DF/RF utility gradients for {len(scorer_raw)} candidate traces...", cfg.progress)
    approximate = ensemble_utility_maps(scorer_raw, scorer_y, masks, bundle, cfg, device)
    model, feature_mean, feature_scale = train_candidate_scorer(features, masks, approximate, cfg, device, stage_dir, selected_views)
    exact_count = min(int(cfg.probe_exact_samples), len(scorer_y))
    exact_local = choose_stratified_subset(scorer_y, exact_count, int(cfg.seed) + 303)
    log(f"[Stage 1/3] Running finite-difference DF/RF validation on {exact_count} traces...", cfg.progress)
    exact = ensemble_finite_difference_maps(scorer_raw[exact_local], scorer_y[exact_local], masks[exact_local], bundle, cfg, device)
    with torch.no_grad():
        predicted = model(
            torch.as_tensor(np.clip((features[exact_local] - feature_mean) / feature_scale, -8.0, 8.0), dtype=torch.float32, device=device),
            torch.as_tensor(masks[exact_local], dtype=torch.float32, device=device),
        ).cpu().numpy()
    rank_metrics = _rank_metrics(exact, predicted, masks[exact_local], int(cfg.candidate_topk))
    candidate = soft_topk_mask(
        torch.as_tensor(predicted),
        torch.as_tensor(masks[exact_local]),
        int(cfg.candidate_topk),
        float(cfg.candidate_temperature),
        hard=not bool(cfg.candidate_soft_topk),
    ).numpy()
    np.savez_compressed(
        stage_dir / "exact_probe_subset.npz",
        train_local_index=scorer_local[exact_local].astype(np.int64),
        exact_utility=exact.astype(np.float32),
        predicted_utility=predicted.astype(np.float32),
        candidate_mask=candidate.astype(np.float32),
        allowed_mask=masks[exact_local].astype(np.float32),
    )
    np.savez_compressed(
        stage_dir / "candidate_training_data.npz",
        train_local_index=scorer_local.astype(np.int64),
        predicted_utility=approximate.astype(np.float32),
        allowed_mask=masks.astype(np.float32),
        structure=structures.astype(np.float32),
    )
    metrics = {
        **rank_metrics,
        "candidate_mode": "executable",
        "candidate_soft_topk": bool(cfg.candidate_soft_topk),
        "candidate_topk": int(cfg.candidate_topk),
        "allowed_region_violation": float(np.max(np.abs(predicted * (1.0 - masks[exact_local])))) if len(predicted) else 0.0,
        "selected_views": list(selected_views),
        "surrogate": "strong_differentiable_df_rf_ensemble",
        "surrogate_attackers": list(bundle.attacker_names),
        "exact_probe_samples": int(exact_count),
        "approximate_samples": int(len(scorer_y)),
    }
    write_json(stage_dir / "candidate_metrics.json", metrics)
    (stage_dir / "summary_zh.md").write_text(
        "\n".join(
            [
                "# V4 Stage 1: Executable leakage-aware condition",
                "",
                f"- selected views: {', '.join(selected_views)}",
                f"- utility surrogate: {metrics['surrogate']}",
                f"- exact probe samples: {exact_count}",
                f"- Spearman: {metrics['spearman']:.6f}",
                f"- Top-K overlap: {metrics['topk_overlap']:.6f}",
                f"- NDCG@K: {metrics['ndcg_at_k']:.6f}",
                f"- allowed-region violation: {metrics['allowed_region_violation']:.8f}",
                "",
                "The validation probe uses finite-difference insertion against the frozen ProjectDF/ProjectRF ensemble.",
            ]
        ),
        encoding="utf-8",
    )
    return metrics


def load_candidate_components(run_dir: Path, cfg, device: torch.device):
    stage_dir = run_dir / "stage1_executable_condition"
    scorer_payload = torch.load(stage_dir / "candidate_scorer_checkpoint.pt", map_location=device, weights_only=False)
    scorer_cfg = scorer_payload["config"]
    scorer = CandidateScorer(int(scorer_cfg["input_dim"]), int(scorer_cfg["patch_num"]), int(scorer_cfg["hidden_dim"])).to(device)
    scorer.load_state_dict(scorer_payload["model_state"])
    scorer.eval()
    bundle = load_strong_surrogates(run_dir, cfg, device)
    feature_mean = np.asarray(scorer_cfg["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(scorer_cfg["feature_scale"], dtype=np.float32)
    return scorer, list(scorer_cfg["selected_views"]), feature_mean, feature_scale, bundle

