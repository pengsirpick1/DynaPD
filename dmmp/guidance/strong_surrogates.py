"""Strong differentiable DF/RF supervision shared by every DMMPv3 stage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from ..evaluation.attack_models import build_df_input, build_rf_tam_input, make_attack_model
from ..data import choose_stratified_subset
from ..encoders.prefix import nonzero_trace, tam_patch_center_slots, tam_patch_ids
from ..utils import log, write_json


@dataclass
class StrongAttackContext:
    df_direction: torch.Tensor
    rf_tam: torch.Tensor
    patch_counts: torch.Tensor
    rf_patch_slots: torch.Tensor


@dataclass
class StrongSurrogateBundle:
    models: dict[str, torch.nn.Module]
    classes: np.ndarray
    weights: dict[str, float]
    patch_num: int
    max_trace_length: int
    rf_num_slots: int
    rf_max_load_time: float

    @property
    def attacker_names(self) -> tuple[str, ...]:
        return tuple(self.models)

    def freeze(self) -> None:
        for model in self.models.values():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def logits_from_allocation(
        self,
        allocation: torch.Tensor,
        context: StrongAttackContext,
    ) -> dict[str, torch.Tensor]:
        logits: dict[str, torch.Tensor] = {}
        if "df" in self.models:
            logits["df"] = self.models["df"](_soft_df_input(context, allocation, self.max_trace_length))
        if "rf" in self.models:
            logits["rf"] = self.models["rf"](_soft_rf_input(context, allocation))
        return logits


def parse_attacker_names(value: str) -> tuple[str, ...]:
    normalized = str(value).strip().lower()
    if normalized == "both":
        return ("df", "rf")
    names = tuple(item.strip() for item in normalized.replace(";", ",").split(",") if item.strip())
    invalid = set(names) - {"df", "rf"}
    if invalid or not names:
        raise ValueError(f"Strong surrogate attackers must be df, rf, or both; got {value!r}")
    return names


def _class_positions(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {int(label): index for index, label in enumerate(classes.tolist())}
    return np.asarray([lookup[int(label)] for label in labels], dtype=np.int64)


def class_positions(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map raw site labels to contiguous surrogate classifier output positions."""
    return _class_positions(np.asarray(labels), np.asarray(classes))


@torch.no_grad()
def surrogate_pseudo_label_positions(
    raw: np.ndarray,
    bundle: StrongSurrogateBundle,
    cfg,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    rows = np.asarray(raw)
    if len(rows) == 0:
        return np.zeros((0,), dtype=np.int64), {
            "guidance_target_source": "frozen_surrogate_observed_prefix_pseudo_label",
            "guidance_pseudo_target_samples": 0.0,
            "guidance_pseudo_target_mean_confidence": 0.0,
            "guidance_pseudo_target_attacker_agreement": 0.0,
        }
    batch_size = max(1, int(cfg.surrogate_gradient_batch_size))
    positions = []
    confidences = []
    agreements = []
    normalized_weights = {
        name: float(bundle.weights.get(name, 1.0))
        for name in bundle.attacker_names
    }
    weight_total = sum(normalized_weights.values()) or 1.0
    normalized_weights = {name: value / weight_total for name, value in normalized_weights.items()}
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        context = build_attack_context(rows[start:end], cfg, device)
        zero = torch.zeros((end - start, 2, int(cfg.patch_num)), dtype=torch.float32, device=device)
        logits = bundle.logits_from_allocation(zero, context)
        weighted_probs = None
        predictions = []
        for name, values in logits.items():
            probs = torch.softmax(values, dim=1)
            weighted = probs * float(normalized_weights.get(name, 1.0))
            weighted_probs = weighted if weighted_probs is None else weighted_probs + weighted
            predictions.append(probs.argmax(dim=1))
        if weighted_probs is None:
            raise RuntimeError("No frozen surrogate models are available for label-free guidance targets")
        batch_confidence, batch_positions = weighted_probs.max(dim=1)
        positions.append(batch_positions.cpu().numpy().astype(np.int64))
        confidences.append(batch_confidence.cpu().numpy().astype(np.float32))
        if len(predictions) > 1:
            stacked = torch.stack(predictions, dim=1)
            agreements.append((stacked == batch_positions.reshape(-1, 1)).float().mean(dim=1).cpu().numpy())
        else:
            agreements.append(np.ones((end - start,), dtype=np.float32))
    position_np = np.concatenate(positions, axis=0).astype(np.int64)
    confidence_np = np.concatenate(confidences, axis=0).astype(np.float32)
    agreement_np = np.concatenate(agreements, axis=0).astype(np.float32)
    return position_np, {
        "guidance_target_source": "frozen_surrogate_observed_prefix_pseudo_label",
        "guidance_pseudo_target_samples": float(len(position_np)),
        "guidance_pseudo_target_mean_confidence": float(np.mean(confidence_np)) if len(confidence_np) else 0.0,
        "guidance_pseudo_target_attacker_agreement": float(np.mean(agreement_np)) if len(agreement_np) else 0.0,
    }


@torch.no_grad()
def resolve_guidance_positions(
    raw_or_prefix: np.ndarray,
    labels: np.ndarray | None,
    bundle: StrongSurrogateBundle,
    cfg,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | str]]:
    mode = str(getattr(cfg, "guidance_label_mode", "pseudo")).strip().lower()
    if mode == "pseudo":
        positions, metrics = surrogate_pseudo_label_positions(raw_or_prefix, bundle, cfg, device)
        return positions, {
            **metrics,
            "guidance_label_mode": "pseudo",
            "guidance_target_samples": float(len(positions)),
        }
    if mode != "true":
        raise ValueError(f"Unsupported guidance_label_mode={mode!r}; expected 'pseudo' or 'true'")
    if labels is None:
        raise ValueError("guidance_label_mode='true' requires true site labels")
    label_rows = np.asarray(labels)
    if len(label_rows) != len(raw_or_prefix):
        raise ValueError(
            "guidance_label_mode='true' requires one label per guidance row; "
            f"got labels={len(label_rows)} rows={len(raw_or_prefix)}"
        )
    try:
        positions = _class_positions(label_rows, bundle.classes)
    except KeyError as exc:
        missing = int(exc.args[0])
        raise ValueError(
            f"True label {missing} is not present in the frozen surrogate class set"
        ) from exc
    rows = np.asarray(raw_or_prefix)
    batch_size = max(1, int(cfg.surrogate_gradient_batch_size))
    confidences = []
    agreements = []
    normalized_weights = {
        name: float(bundle.weights.get(name, 1.0))
        for name in bundle.attacker_names
    }
    weight_total = sum(normalized_weights.values()) or 1.0
    normalized_weights = {name: value / weight_total for name, value in normalized_weights.items()}
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        context = build_attack_context(rows[start:end], cfg, device)
        zero = torch.zeros((end - start, 2, int(cfg.patch_num)), dtype=torch.float32, device=device)
        logits = bundle.logits_from_allocation(zero, context)
        position_t = torch.as_tensor(positions[start:end], dtype=torch.long, device=device)
        weighted_probs = None
        predictions = []
        for name, values in logits.items():
            probs = torch.softmax(values, dim=1)
            weighted = probs * float(normalized_weights.get(name, 1.0))
            weighted_probs = weighted if weighted_probs is None else weighted_probs + weighted
            predictions.append(probs.argmax(dim=1))
        if weighted_probs is None:
            raise RuntimeError("No frozen surrogate models are available for true-label guidance targets")
        confidences.append(weighted_probs.gather(1, position_t.reshape(-1, 1)).reshape(-1).cpu().numpy().astype(np.float32))
        if predictions:
            stacked = torch.stack(predictions, dim=1)
            agreements.append((stacked == position_t.reshape(-1, 1)).float().mean(dim=1).cpu().numpy())
        else:
            agreements.append(np.zeros((end - start,), dtype=np.float32))
    confidence_np = np.concatenate(confidences, axis=0).astype(np.float32) if confidences else np.zeros((0,), dtype=np.float32)
    agreement_np = np.concatenate(agreements, axis=0).astype(np.float32) if agreements else np.zeros((0,), dtype=np.float32)
    return positions, {
        "guidance_label_mode": "true",
        "guidance_target_source": "true_site_label",
        "guidance_target_samples": float(len(positions)),
        "guidance_true_target_samples": float(len(positions)),
        "guidance_true_target_unique_labels": float(len(np.unique(label_rows))) if len(label_rows) else 0.0,
        "guidance_true_target_mean_confidence": float(np.mean(confidence_np)) if len(confidence_np) else 0.0,
        "guidance_true_target_attacker_agreement": float(np.mean(agreement_np)) if len(agreement_np) else 0.0,
        "guidance_pseudo_target_mean_confidence": 0.0,
        "guidance_pseudo_target_attacker_agreement": 0.0,
    }


def _patch_counts(raw: np.ndarray, patch_num: int, max_trace_length: int, max_load_time: float) -> np.ndarray:
    rows = np.asarray(raw)
    result = np.zeros((len(rows), 2, int(patch_num)), dtype=np.float32)
    for row_index, trace in enumerate(rows):
        values = nonzero_trace(trace)[: int(max_trace_length)]
        if not values.size:
            continue
        patch_ids = tam_patch_ids(values, int(patch_num), max_load_time=float(max_load_time))
        np.add.at(result[row_index, 0], patch_ids[values > 0], 1.0)
        np.add.at(result[row_index, 1], patch_ids[values < 0], 1.0)
    return result


def _rf_patch_slots(raw: np.ndarray, patch_num: int, max_trace_length: int, num_slots: int, max_load_time: float) -> np.ndarray:
    del max_trace_length
    centers = tam_patch_center_slots(int(patch_num), int(num_slots), max_load_time=float(max_load_time))
    return np.repeat(centers.reshape(1, -1), len(raw), axis=0).astype(np.int64)


def build_attack_context(raw: np.ndarray, cfg, device: torch.device) -> StrongAttackContext:
    rows = np.asarray(raw)
    df = build_df_input(rows, max_len=int(cfg.max_trace_length))[:, 0, :]
    rf = build_rf_tam_input(
        rows,
        max_len=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        num_slots=int(cfg.surrogate_rf_num_slots),
    )
    counts = _patch_counts(rows, int(cfg.patch_num), int(cfg.max_trace_length), float(cfg.surrogate_rf_max_load_time))
    slots = _rf_patch_slots(
        rows,
        int(cfg.patch_num),
        int(cfg.max_trace_length),
        int(cfg.surrogate_rf_num_slots),
        float(cfg.surrogate_rf_max_load_time),
    )
    return StrongAttackContext(
        df_direction=torch.as_tensor(df, dtype=torch.float32, device=device),
        rf_tam=torch.as_tensor(rf, dtype=torch.float32, device=device),
        patch_counts=torch.as_tensor(counts, dtype=torch.float32, device=device),
        rf_patch_slots=torch.as_tensor(slots, dtype=torch.long, device=device),
    )


def _soft_df_input(context: StrongAttackContext, allocation: torch.Tensor, max_trace_length: int) -> torch.Tensor:
    real_per_patch = context.patch_counts.sum(dim=1).clamp_min(1.0)
    dummy_total = allocation.sum(dim=1)
    signed_dummy = allocation[:, 0] - allocation[:, 1]
    denominator = real_per_patch + dummy_total
    real_weight = real_per_patch / denominator.clamp_min(1e-6)
    dummy_signal = signed_dummy / denominator.clamp_min(1e-6)
    real_weight = F.interpolate(real_weight.unsqueeze(1), size=int(max_trace_length), mode="nearest").squeeze(1)
    dummy_signal = F.interpolate(dummy_signal.unsqueeze(1), size=int(max_trace_length), mode="nearest").squeeze(1)
    defended = context.df_direction * real_weight + dummy_signal
    return defended.clamp(-1.0, 1.0).unsqueeze(1)


def _soft_rf_input(context: StrongAttackContext, allocation: torch.Tensor) -> torch.Tensor:
    dummy_tam = torch.zeros_like(context.rf_tam)
    slot_index = context.rf_patch_slots.unsqueeze(1).expand(-1, 2, -1)
    dummy_tam = dummy_tam.scatter_add(2, slot_index, allocation)
    return context.rf_tam + dummy_tam


def attacker_utility(logits: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)
    entropy = entropy / math.log(max(probabilities.shape[1], 2))
    if labels is None:
        return entropy - probabilities.max(dim=1).values
    true_confidence = probabilities.gather(1, labels.reshape(-1, 1)).reshape(-1)
    true_logits = logits.gather(1, labels.reshape(-1, 1)).reshape(-1)
    other_logits = logits.masked_fill(F.one_hot(labels, logits.shape[1]).bool(), -1e9).max(dim=1).values
    margin_penalty = F.relu(true_logits - other_logits + 0.20)
    return (1.0 - true_confidence) + entropy + F.softplus(-margin_penalty)


def ensemble_utility(
    logits_by_attacker: dict[str, torch.Tensor],
    labels: torch.Tensor | None,
    weights: dict[str, float],
    robust_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    individual = {name: attacker_utility(logits, labels) for name, logits in logits_by_attacker.items()}
    stacked = torch.stack([individual[name] for name in individual], dim=1)
    normalized_weights = torch.as_tensor(
        [float(weights.get(name, 1.0)) for name in individual],
        dtype=stacked.dtype,
        device=stacked.device,
    )
    normalized_weights = normalized_weights / normalized_weights.sum().clamp_min(1e-8)
    weighted = (stacked * normalized_weights.reshape(1, -1)).sum(dim=1)
    robust = stacked.min(dim=1).values
    combined = (1.0 - float(robust_weight)) * weighted + float(robust_weight) * robust
    return combined, individual


def ensemble_target_risk(
    logits_by_attacker: dict[str, torch.Tensor],
    targets: torch.Tensor,
    margin: float = 0.20,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    individual = {}
    for name, logits in logits_by_attacker.items():
        target_logits = logits.gather(1, targets.reshape(-1, 1)).reshape(-1)
        other_logits = logits.masked_fill(F.one_hot(targets, logits.shape[1]).bool(), -1e9).max(dim=1).values
        scale = max(float(temperature), 1e-6)
        normalized_margin = (target_logits - other_logits + float(margin)) / scale
        individual[name] = F.softplus(normalized_margin) * scale
    stacked = torch.stack([individual[name] for name in individual], dim=1)
    return stacked.max(dim=1).values, individual

def defense_guidance_loss(
    allocation: torch.Tensor,
    context: StrongAttackContext,
    bundle: StrongSurrogateBundle,
    labels: torch.Tensor,
    robust_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = bundle.logits_from_allocation(allocation, context)
    utility, individual = ensemble_utility(logits, labels, bundle.weights, robust_weight)
    metrics = {f"{name}_utility": float(value.mean().detach().cpu()) for name, value in individual.items()}
    return -utility.mean(), metrics


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


@torch.no_grad()
def _evaluate_model(model, attacker: str, raw: np.ndarray, y: np.ndarray, classes: np.ndarray, cfg, device: torch.device) -> float:
    correct = 0
    total = 0
    batch_size = int(cfg.surrogate_batch_size)
    positions = _class_positions(y, classes)
    for idx in _iter_batches(np.arange(len(raw)), batch_size):
        if attacker == "df":
            values = build_df_input(raw[idx], max_len=int(cfg.max_trace_length))
        else:
            values = build_rf_tam_input(
                raw[idx],
                max_len=int(cfg.max_trace_length),
                max_load_time=float(cfg.surrogate_rf_max_load_time),
                num_slots=int(cfg.surrogate_rf_num_slots),
            )
        logits = model(torch.as_tensor(values, dtype=torch.float32, device=device))
        correct += int((logits.argmax(dim=1).cpu().numpy() == positions[idx]).sum())
        total += len(idx)
    return float(correct / max(total, 1))


def _train_one_attacker(attacker: str, train_raw: np.ndarray, train_y: np.ndarray, val_raw: np.ndarray, val_y: np.ndarray, classes: np.ndarray, cfg, device: torch.device):
    model = make_attack_model(
        attacker.upper(),
        len(classes),
        max_trace_length=int(cfg.max_trace_length),
        df_architecture=str(cfg.surrogate_df_architecture),
    ).to(device)
    optimizer = torch.optim.Adamax(model.parameters(), lr=float(cfg.surrogate_lr), weight_decay=1e-5)
    train_positions = _class_positions(train_y, classes)
    rng = np.random.default_rng(int(cfg.seed) + (910 if attacker == "df" else 920))
    best_accuracy = -1.0
    best_state = None
    stale = 0
    for epoch in range(1, int(cfg.surrogate_epochs) + 1):
        order = rng.permutation(len(train_raw))
        model.train()
        total_batches = int(math.ceil(len(order) / max(int(cfg.surrogate_batch_size), 1)))
        heartbeat = max(1, min(max(int(getattr(cfg, "log_every", 100)), 1), max(total_batches // 4, 1)))
        for batch_index, idx in enumerate(_iter_batches(order, int(cfg.surrogate_batch_size)), start=1):
            if attacker == "df":
                values = build_df_input(train_raw[idx], max_len=int(cfg.max_trace_length))
            else:
                values = build_rf_tam_input(
                    train_raw[idx],
                    max_len=int(cfg.max_trace_length),
                    max_load_time=float(cfg.surrogate_rf_max_load_time),
                    num_slots=int(cfg.surrogate_rf_num_slots),
                )
            xb = torch.as_tensor(values, dtype=torch.float32, device=device)
            yb = torch.as_tensor(train_positions[idx], dtype=torch.long, device=device)
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if batch_index == 1 or batch_index == total_batches or batch_index % heartbeat == 0:
                log(
                    f"Strong {attacker.upper()} surrogate: epoch {epoch}/{cfg.surrogate_epochs}, "
                    f"batch {batch_index}/{total_batches}, loss={float(loss.detach().cpu()):.6f}",
                    cfg.progress,
                )
        accuracy = _evaluate_model(model, attacker, val_raw, val_y, classes, cfg, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        log(
            f"Strong {attacker.upper()} surrogate: epoch {epoch}/{cfg.surrogate_epochs}, "
            f"val_acc={accuracy:.6f}, best={best_accuracy:.6f}, stale={stale}/{cfg.surrogate_patience}",
            cfg.progress,
        )
        if stale >= int(cfg.surrogate_patience):
            break
    if best_state is None:
        raise RuntimeError(f"Strong {attacker.upper()} surrogate did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, float(best_accuracy)


def train_strong_surrogates(raw: np.ndarray, labels: np.ndarray, train_indices: np.ndarray, val_indices: np.ndarray, cfg, device: torch.device, output_dir: str | Path) -> StrongSurrogateBundle:
    names = parse_attacker_names(cfg.guidance_attackers)
    train_count = min(len(train_indices), int(cfg.surrogate_train_samples)) if int(cfg.surrogate_train_samples) > 0 else len(train_indices)
    val_count = min(len(val_indices), int(cfg.surrogate_val_samples)) if int(cfg.surrogate_val_samples) > 0 else len(val_indices)
    train_local = choose_stratified_subset(labels[train_indices], train_count, int(cfg.seed) + 901)
    val_local = choose_stratified_subset(labels[val_indices], val_count, int(cfg.seed) + 902)
    selected_train = np.asarray(train_indices)[train_local]
    selected_val = np.asarray(val_indices)[val_local]
    classes = np.unique(labels[train_indices]).astype(np.int64)
    models: dict[str, torch.nn.Module] = {}
    metrics: dict[str, float | int | list[str]] = {
        "attackers": list(names),
        "train_samples": int(len(selected_train)),
        "validation_samples": int(len(selected_val)),
    }
    for name in names:
        log(
            f"[Stage 1/3] {name.upper()} surrogate training begins: "
            f"train={len(selected_train)}, val={len(selected_val)}, epochs<={cfg.surrogate_epochs}",
            cfg.progress,
        )
        model, val_accuracy = _train_one_attacker(
            name,
            raw[selected_train],
            labels[selected_train],
            raw[selected_val],
            labels[selected_val],
            classes,
            cfg,
            device,
        )
        if val_accuracy < float(cfg.surrogate_min_val_accuracy):
            raise RuntimeError(
                f"Strong {name.upper()} surrogate validation accuracy {val_accuracy:.4f} is below "
                f"the required {float(cfg.surrogate_min_val_accuracy):.4f}; defense training is aborted."
            )
        models[name] = model
        metrics[f"{name}_best_val_accuracy"] = val_accuracy
    raw_weights = {"df": float(cfg.surrogate_df_weight), "rf": float(cfg.surrogate_rf_weight)}
    weights = {name: raw_weights[name] for name in names}
    bundle = StrongSurrogateBundle(
        models=models,
        classes=classes,
        weights=weights,
        patch_num=int(cfg.patch_num),
        max_trace_length=int(cfg.max_trace_length),
        rf_num_slots=int(cfg.surrogate_rf_num_slots),
        rf_max_load_time=float(cfg.surrogate_rf_max_load_time),
    )
    bundle.freeze()
    target = Path(output_dir) / "strong_surrogate_ensemble.pt"
    torch.save(
        {
            "model_states": {name: model.state_dict() for name, model in models.items()},
            "classes": classes,
            "weights": weights,
            "attackers": list(names),
            "config": {
                "patch_num": int(cfg.patch_num),
                "max_trace_length": int(cfg.max_trace_length),
                "rf_num_slots": int(cfg.surrogate_rf_num_slots),
                "rf_max_load_time": float(cfg.surrogate_rf_max_load_time),
                "df_architecture": str(cfg.surrogate_df_architecture),
            },
            "metrics": metrics,
        },
        target,
    )
    write_json(Path(output_dir) / "strong_surrogate_metrics.json", metrics)
    return bundle


def load_strong_surrogates(run_dir: str | Path, cfg, device: torch.device) -> StrongSurrogateBundle:
    payload = torch.load(Path(run_dir) / "stage1_executable_condition" / "strong_surrogate_ensemble.pt", map_location=device, weights_only=False)
    classes = np.asarray(payload["classes"], dtype=np.int64)
    model_cfg = payload["config"]
    models = {}
    for name in payload["attackers"]:
        model = make_attack_model(
            str(name).upper(),
            len(classes),
            max_trace_length=int(model_cfg["max_trace_length"]),
            df_architecture=str(model_cfg["df_architecture"]),
        ).to(device)
        model.load_state_dict(payload["model_states"][name])
        models[str(name)] = model
    bundle = StrongSurrogateBundle(
        models=models,
        classes=classes,
        weights={str(key): float(value) for key, value in payload["weights"].items()},
        patch_num=int(model_cfg["patch_num"]),
        max_trace_length=int(model_cfg["max_trace_length"]),
        rf_num_slots=int(model_cfg["rf_num_slots"]),
        rf_max_load_time=float(model_cfg["rf_max_load_time"]),
    )
    bundle.freeze()
    return bundle


def ensemble_utility_maps(raw: np.ndarray, labels: np.ndarray, masks: np.ndarray, bundle: StrongSurrogateBundle, cfg, device: torch.device) -> np.ndarray:
    rows = []
    batch_size = max(1, min(int(cfg.surrogate_gradient_batch_size), len(raw)))
    class_positions = _class_positions(labels, bundle.classes)
    for start in range(0, len(raw), batch_size):
        end = min(start + batch_size, len(raw))
        context = build_attack_context(raw[start:end], cfg, device)
        allocation = torch.zeros((end - start, 2, int(cfg.patch_num)), dtype=torch.float32, device=device, requires_grad=True)
        label_t = torch.as_tensor(class_positions[start:end], dtype=torch.long, device=device)
        logits = bundle.logits_from_allocation(allocation, context)
        utility, _ = ensemble_utility(logits, label_t, bundle.weights, float(cfg.surrogate_robust_weight))
        gradient = torch.autograd.grad(utility.sum(), allocation)[0]
        rows.append(torch.relu(gradient).detach().cpu().numpy())
    utility_map = np.concatenate(rows, axis=0) * np.asarray(masks, dtype=np.float32)
    peak = utility_map.reshape(len(utility_map), -1).max(axis=1).reshape(-1, 1, 1)
    return (utility_map / np.maximum(peak, 1e-8)).astype(np.float32)


def _repeat_context(context: StrongAttackContext, count: int) -> StrongAttackContext:
    return StrongAttackContext(
        df_direction=context.df_direction.repeat(int(count), 1),
        rf_tam=context.rf_tam.repeat(int(count), 1, 1),
        patch_counts=context.patch_counts.repeat(int(count), 1, 1),
        rf_patch_slots=context.rf_patch_slots.repeat(int(count), 1),
    )


@torch.no_grad()
def ensemble_finite_difference_maps(raw: np.ndarray, labels: np.ndarray, masks: np.ndarray, bundle: StrongSurrogateBundle, cfg, device: torch.device) -> np.ndarray:
    """Expensive held-out probe used only to validate the learned candidate scorer."""
    result = np.zeros((len(raw), 2, int(cfg.patch_num)), dtype=np.float32)
    positions = _class_positions(labels, bundle.classes)
    probe_batch = max(1, int(cfg.surrogate_gradient_batch_size))
    for row_index in range(len(raw)):
        base_context = build_attack_context(raw[row_index : row_index + 1], cfg, device)
        label = torch.as_tensor([positions[row_index]], dtype=torch.long, device=device)
        zero = torch.zeros((1, 2, int(cfg.patch_num)), dtype=torch.float32, device=device)
        base_logits = bundle.logits_from_allocation(zero, base_context)
        base_utility, _ = ensemble_utility(base_logits, label, bundle.weights, float(cfg.surrogate_robust_weight))
        allowed = np.flatnonzero(np.asarray(masks[row_index]).reshape(-1) > 0)
        gains = []
        for start in range(0, len(allowed), probe_batch):
            cells = allowed[start : start + probe_batch]
            allocation = torch.zeros((len(cells), 2, int(cfg.patch_num)), dtype=torch.float32, device=device)
            allocation.reshape(len(cells), -1)[torch.arange(len(cells), device=device), torch.as_tensor(cells, device=device)] = float(cfg.probe_dummy_count)
            context = _repeat_context(base_context, len(cells))
            logits = bundle.logits_from_allocation(allocation, context)
            truth = label.repeat(len(cells))
            utility, _ = ensemble_utility(logits, truth, bundle.weights, float(cfg.surrogate_robust_weight))
            gains.append(torch.relu(utility - base_utility).cpu().numpy())
        if gains:
            result[row_index].reshape(-1)[allowed] = np.concatenate(gains)
        peak = float(result[row_index].max())
        if peak > 1e-8:
            result[row_index] /= peak
    return result


@torch.no_grad()
def ensemble_metrics_from_allocation(raw: np.ndarray, labels: np.ndarray, allocation: np.ndarray, bundle: StrongSurrogateBundle, cfg, device: torch.device) -> dict[str, float]:
    totals = {name: {"correct": 0.0, "entropy": 0.0, "max_confidence": 0.0} for name in bundle.attacker_names}
    positions_np = _class_positions(labels, bundle.classes)
    batch_size = max(1, int(cfg.surrogate_gradient_batch_size))
    for start in range(0, len(raw), batch_size):
        end = min(start + batch_size, len(raw))
        context = build_attack_context(raw[start:end], cfg, device)
        allocation_t = torch.as_tensor(allocation[start:end], dtype=torch.float32, device=device)
        logits = bundle.logits_from_allocation(allocation_t, context)
        positions = torch.as_tensor(positions_np[start:end], dtype=torch.long, device=device)
        for name, values in logits.items():
            probabilities = torch.softmax(values, dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1) / math.log(max(values.shape[1], 2))
            totals[name]["correct"] += float((values.argmax(dim=1) == positions).float().sum().cpu())
            totals[name]["entropy"] += float(entropy.sum().cpu())
            totals[name]["max_confidence"] += float(probabilities.max(dim=1).values.sum().cpu())
    result: dict[str, float] = {}
    accuracies = []
    for name, values in totals.items():
        accuracy = values["correct"] / max(len(raw), 1)
        result[f"surrogate_{name}_accuracy"] = float(accuracy)
        result[f"surrogate_{name}_entropy"] = float(values["entropy"] / max(len(raw), 1))
        result[f"surrogate_{name}_max_confidence"] = float(values["max_confidence"] / max(len(raw), 1))
        accuracies.append(float(accuracy))
    result["surrogate_ensemble_mean_accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
    result["surrogate_ensemble_worst_accuracy"] = float(np.max(accuracies)) if accuracies else 0.0
    return result


@torch.no_grad()
def ensemble_metrics_from_rendered(traces: list[np.ndarray], labels: np.ndarray, bundle: StrongSurrogateBundle, cfg, device: torch.device) -> dict[str, float]:
    """Score rendered traces; label-free uncertainty drives Stage 3 selection."""
    totals = {name: {"correct": 0.0, "entropy": 0.0, "max_confidence": 0.0, "margin": 0.0} for name in bundle.attacker_names}
    positions_np = _class_positions(labels, bundle.classes)
    batch_size = max(1, int(cfg.surrogate_gradient_batch_size))
    for start in range(0, len(traces), batch_size):
        end = min(start + batch_size, len(traces))
        batch_rows = traces[start:end]
        width = max([min(len(row), int(cfg.max_trace_length)) for row in batch_rows] + [1])
        padded = np.zeros((len(batch_rows), width), dtype=np.float32)
        for row_index, row in enumerate(batch_rows):
            values = np.asarray(row, dtype=np.float32)[:width]
            padded[row_index, : len(values)] = values
        positions = torch.as_tensor(positions_np[start:end], dtype=torch.long, device=device)
        for name, model in bundle.models.items():
            if name == "df":
                values = build_df_input(padded, max_len=int(cfg.max_trace_length))
            else:
                values = build_rf_tam_input(
                    padded,
                    max_len=int(cfg.max_trace_length),
                    max_load_time=float(cfg.surrogate_rf_max_load_time),
                    num_slots=int(cfg.surrogate_rf_num_slots),
                )
            logits = model(torch.as_tensor(values, dtype=torch.float32, device=device))
            probabilities = torch.softmax(logits, dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1) / math.log(max(logits.shape[1], 2))
            top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
            margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
            totals[name]["correct"] += float((logits.argmax(dim=1) == positions).float().sum().cpu())
            totals[name]["entropy"] += float(entropy.sum().cpu())
            totals[name]["max_confidence"] += float(probabilities.max(dim=1).values.sum().cpu())
            totals[name]["margin"] += float(margin.sum().cpu())
    result: dict[str, float] = {}
    accuracies = []
    entropies = []
    confidences = []
    margins = []
    pressures = []
    for name, values in totals.items():
        accuracy = values["correct"] / max(len(traces), 1)
        entropy = values["entropy"] / max(len(traces), 1)
        max_confidence = values["max_confidence"] / max(len(traces), 1)
        margin = values["margin"] / max(len(traces), 1)
        pressure = max_confidence + 0.50 * margin - 0.50 * entropy
        result[f"surrogate_{name}_accuracy"] = float(accuracy)
        result[f"surrogate_{name}_entropy"] = float(entropy)
        result[f"surrogate_{name}_max_confidence"] = float(max_confidence)
        result[f"surrogate_{name}_margin"] = float(margin)
        result[f"surrogate_{name}_label_free_pressure"] = float(pressure)
        accuracies.append(float(accuracy))
        entropies.append(float(entropy))
        confidences.append(float(max_confidence))
        margins.append(float(margin))
        pressures.append(float(pressure))
    result["surrogate_ensemble_mean_accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
    result["surrogate_ensemble_worst_accuracy"] = float(np.max(accuracies)) if accuracies else 0.0
    result["surrogate_ensemble_mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0
    result["surrogate_ensemble_worst_max_confidence"] = float(np.max(confidences)) if confidences else 0.0
    result["surrogate_ensemble_worst_margin"] = float(np.max(margins)) if margins else 0.0
    result["surrogate_ensemble_worst_label_free_pressure"] = float(np.max(pressures)) if pressures else 1.0
    result["surrogate_label_free_attack_pressure"] = float(result["surrogate_ensemble_worst_label_free_pressure"])
    return result


@torch.no_grad()
def strong_global_targets(raw: np.ndarray, bundle: StrongSurrogateBundle, cfg, device: torch.device, hidden_dim: int) -> np.ndarray:
    outputs = []
    batch_size = max(1, int(cfg.surrogate_gradient_batch_size))
    for start in range(0, len(raw), batch_size):
        end = min(start + batch_size, len(raw))
        context = build_attack_context(raw[start:end], cfg, device)
        zero = torch.zeros((end - start, 2, int(cfg.patch_num)), dtype=torch.float32, device=device)
        logits = bundle.logits_from_allocation(zero, context)
        outputs.append(torch.cat([torch.softmax(logits[name], dim=1) for name in bundle.attacker_names], dim=1).cpu().numpy())
    probabilities = np.concatenate(outputs, axis=0).astype(np.float32)
    rng = np.random.default_rng(int(cfg.seed) + 940)
    projection = rng.normal(0.0, 1.0 / math.sqrt(max(probabilities.shape[1], 1)), size=(probabilities.shape[1], int(hidden_dim))).astype(np.float32)
    targets = probabilities @ projection
    return ((targets - targets.mean(axis=1, keepdims=True)) / np.maximum(targets.std(axis=1, keepdims=True), 1e-4)).astype(np.float32)

