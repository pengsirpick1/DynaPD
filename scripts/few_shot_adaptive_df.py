"""Few-shot adaptive DF probe for DMMPv3 defended deployments.

This script answers a narrow question: if an attacker gets a small set of
previously deployed defended traces, can a full clean-trained DF model
fine-tuned on those samples generalize to a full fresh defended deployment
generated from the same DMMPv3 run?
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.constraints.user_profiles import load_profiles
from dmmp.data import choose_stratified_subset, load_cw_data
from dmmp.evaluation.attack_models import build_df_input, make_attack_model
from dmmp.evaluation.attacks import train_df_model
from dmmp.evaluation.profile_attacks import (
    _defended_input,
    _defense_artifact_signature,
    _defense_config_from_run,
    _find_profile,
    _get_profile_dataset,
    _load_splits,
    _selected_budget_and_keep,
)
from dmmp.projection.padding import load_ragged_npz
from dmmp.utils import log, resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import AttackConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a few-shot adaptive DF check: fine-tune on old defended traces "
            "and evaluate on a fresh defended deployment from an existing DMMPv3 run."
        )
    )
    parser.add_argument("--run_dir", required=True, help="Existing DMMPv3 run directory with Stage 1/2/3 artifacts.")
    parser.add_argument("--data_root", default="", help="Optional CW data root override.")
    parser.add_argument("--output_dir", default="", help="Defaults to <run_dir>/attack_eval/few_shot_adaptive_df.")
    parser.add_argument("--base_checkpoint", default="", help="Optional DF checkpoint to initialize from.")
    parser.add_argument("--target_profile_id", default="", help="Profile id to use; default is the first test profile.")
    parser.add_argument("--profile_split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--profile_index", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0, help="Optional class cap for debugging; 0 means all run classes.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Legacy optional cap for base clean train and full fresh test pools; 0 means no extra cap.",
    )
    parser.add_argument("--base_max_train_traces", type=int, default=0, help="Optional clean base-train cap; 0 means full selected train split.")
    parser.add_argument("--base_max_val_traces", type=int, default=0, help="Optional clean base-val cap; 0 means full selected validation split.")
    parser.add_argument("--fresh_max_test_traces", type=int, default=0, help="Optional fresh-test clean trace cap; 0 means full selected test split.")
    parser.add_argument("--few_shot_per_class", type=int, default=20)
    parser.add_argument(
        "--fresh_eval_per_class",
        type=int,
        default=0,
        help="Optional fresh defended eval rows per class after repeats; 0 means full fresh defended test set.",
    )
    parser.add_argument("--base_epochs", type=int, default=20, help="Only used when no suitable full clean DF checkpoint is found.")
    parser.add_argument("--base_patience", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=2e-3)
    parser.add_argument("--base_min_val_accuracy", type=float, default=0.70)
    parser.add_argument("--base_min_clean_accuracy", type=float, default=0.70)
    parser.add_argument(
        "--require_qualified_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort before few-shot fine-tuning if the clean-base DF remains too weak.",
    )
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--df_architecture", choices=["project", "wflib"], default="project")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--force_retrain_base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="By default train a full clean-base DF inside this audit. Use --no-force_retrain_base to reuse an accepted checkpoint.",
    )
    parser.add_argument(
        "--allow_surrogate_base_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Stage 1 surrogate DF as a base checkpoint. Disabled by default because the audit expects a clean-trained attack DF.",
    )
    parser.add_argument(
        "--trust_base_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Trust an explicit --base_checkpoint as a full clean train/val DF checkpoint when provenance is not encoded in the file.",
    )
    return parser.parse_args()


def _make_attack_cfg(args: argparse.Namespace, output_dir: Path) -> AttackConfig:
    return AttackConfig(
        run_dir=str(args.run_dir),
        data_root=str(args.data_root),
        output_dir=str(output_dir),
        seed=int(args.seed),
        device=str(args.device),
        clean_df_epochs=int(args.base_epochs),
        clean_df_patience=int(args.base_patience),
        clean_df_lr=float(args.base_lr),
        adaptive_epochs=int(args.finetune_epochs),
        adaptive_lr=float(args.finetune_lr),
        df_batch_size=int(args.batch_size),
        df_architecture=str(args.df_architecture),
        progress=bool(args.progress),
        log_every=int(args.log_every),
    )


def _class_counts(y: np.ndarray, classes: Sequence[int]) -> dict[str, int]:
    labels = np.asarray(y, dtype=np.int64)
    return {str(int(label)): int(np.sum(labels == int(label))) for label in classes}


def _select_classes(labels: np.ndarray, train_idx: np.ndarray, max_classes: int) -> np.ndarray:
    classes = np.unique(np.asarray(labels, dtype=np.int64)[np.asarray(train_idx, dtype=np.int64)])
    if int(max_classes) > 0:
        classes = classes[: int(max_classes)]
    if len(classes) == 0:
        raise ValueError("No classes are available after max_classes filtering")
    return classes.astype(np.int64)


def _filter_indices(indices: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    mask = np.isin(np.asarray(labels, dtype=np.int64)[selected], np.asarray(classes, dtype=np.int64))
    return selected[mask]


def _subsample_absolute(
    indices: np.ndarray,
    labels: np.ndarray,
    maximum: int,
    seed: int,
    *,
    required_classes: np.ndarray | None = None,
) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    if int(maximum) <= 0 or len(selected) <= int(maximum):
        return selected
    if required_classes is not None and int(maximum) < len(required_classes):
        raise ValueError(
            f"max_samples={int(maximum)} is smaller than the selected class count "
            f"{len(required_classes)}; increase max_samples or lower max_classes"
        )
    if required_classes is not None:
        rng = np.random.default_rng(int(seed))
        labels_np = np.asarray(labels, dtype=np.int64)
        required = np.asarray(required_classes, dtype=np.int64)
        chosen: list[int] = []
        remaining_by_class: dict[int, np.ndarray] = {}
        for label in required:
            candidates = selected[labels_np[selected] == int(label)]
            if len(candidates) == 0:
                raise ValueError(f"No samples are available for selected class {int(label)}")
            first = int(rng.choice(candidates, size=1, replace=False)[0])
            chosen.append(first)
            remaining_by_class[int(label)] = candidates[candidates != first]
        remaining = int(maximum) - len(chosen)
        while remaining > 0:
            made_progress = False
            for label in required.tolist():
                pool = remaining_by_class[int(label)]
                if len(pool) == 0 or remaining <= 0:
                    continue
                take_index = int(rng.integers(0, len(pool)))
                chosen.append(int(pool[take_index]))
                remaining_by_class[int(label)] = np.delete(pool, take_index)
                remaining -= 1
                made_progress = True
            if not made_progress:
                break
        return np.asarray(sorted(chosen), dtype=np.int64)
    local = choose_stratified_subset(np.asarray(labels, dtype=np.int64)[selected], int(maximum), int(seed))
    return selected[local]


def _sample_base_indices_per_class(
    indices: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    per_class: int,
    seed: int,
    *,
    purpose: str,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    labels_np = np.asarray(labels, dtype=np.int64)
    pool = np.asarray(indices, dtype=np.int64)
    for label in np.asarray(classes, dtype=np.int64):
        candidates = pool[labels_np[pool] == int(label)]
        if len(candidates) < int(per_class):
            raise ValueError(
                f"Not enough {purpose} traces for class {int(label)}: "
                f"need {int(per_class)}, found {len(candidates)}"
            )
        chosen = rng.choice(candidates, size=int(per_class), replace=False)
        selected.extend(int(item) for item in chosen.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _min_count_per_class(indices: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> int:
    labels_np = np.asarray(labels, dtype=np.int64)
    pool = np.asarray(indices, dtype=np.int64)
    counts = [int(np.sum(labels_np[pool] == int(label))) for label in np.asarray(classes, dtype=np.int64)]
    return min(counts) if counts else 0


def _select_rows_per_class(
    traces: Sequence[np.ndarray],
    origins: Sequence[np.ndarray],
    labels: np.ndarray,
    classes: np.ndarray,
    per_class: int,
    seed: int,
    *,
    purpose: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(labels, dtype=np.int64)
    selected: list[int] = []
    for label in np.asarray(classes, dtype=np.int64):
        candidates = np.where(y == int(label))[0]
        if len(candidates) < int(per_class):
            raise ValueError(
                f"Not enough generated {purpose} rows for class {int(label)}: "
                f"need {int(per_class)}, found {len(candidates)}"
            )
        chosen = rng.choice(candidates, size=int(per_class), replace=False)
        selected.extend(int(item) for item in chosen.tolist())
    order = np.asarray(selected, dtype=np.int64)
    rng.shuffle(order)
    return (
        [np.asarray(traces[int(index)], dtype=np.float32) for index in order.tolist()],
        [np.asarray(origins[int(index)], dtype=bool) for index in order.tolist()],
        y[order].astype(np.int64),
        order,
    )


def _select_unique_clean_rows_per_class(
    traces: Sequence[np.ndarray],
    origins: Sequence[np.ndarray],
    labels: np.ndarray,
    clean_indices: np.ndarray,
    classes: np.ndarray,
    per_class: int,
    seed: int,
    *,
    purpose: str,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    y = np.asarray(labels, dtype=np.int64)
    clean = np.asarray(clean_indices, dtype=np.int64)
    if len(clean) != len(y):
        raise ValueError(f"Cannot enforce unique clean traces for {purpose}: clean_index metadata is missing")
    selected: list[int] = []
    for label in np.asarray(classes, dtype=np.int64):
        label_rows = np.where(y == int(label))[0]
        unique_clean = np.unique(clean[label_rows])
        if len(unique_clean) < int(per_class):
            raise ValueError(
                f"Not enough unique clean traces for generated {purpose} rows in class {int(label)}: "
                f"need {int(per_class)}, found {len(unique_clean)}"
            )
        chosen_clean = rng.choice(unique_clean, size=int(per_class), replace=False)
        for clean_id in chosen_clean.tolist():
            candidates = label_rows[clean[label_rows] == int(clean_id)]
            selected.append(int(rng.choice(candidates, size=1, replace=False)[0]))
    order = np.asarray(selected, dtype=np.int64)
    rng.shuffle(order)
    return (
        [np.asarray(traces[int(index)], dtype=np.float32) for index in order.tolist()],
        [np.asarray(origins[int(index)], dtype=bool) for index in order.tolist()],
        y[order].astype(np.int64),
        order,
    )


def _df_input_indexed(raw: np.ndarray, indices: np.ndarray, max_trace_length: int, *, chunk_size: int = 512) -> np.ndarray:
    selected = np.asarray(indices, dtype=np.int64)
    result = np.empty((len(selected), 1, int(max_trace_length)), dtype=np.float32)
    for start in range(0, len(selected), int(chunk_size)):
        end = min(start + int(chunk_size), len(selected))
        result[start:end] = build_df_input(np.asarray(raw[selected[start:end]]), max_len=int(max_trace_length))
    return result


def _ragged_overhead(traces: Sequence[np.ndarray], origins: Sequence[np.ndarray], max_trace_length: int) -> dict[str, float]:
    raw_dummy, raw_retention, visible_dummy, input_retention, clipped = [], [], [], [], []
    for trace, origin in zip(traces, origins):
        origin_bool = np.asarray(origin, dtype=bool)
        original = max(int(origin_bool.sum()), 1)
        dummy = int((~origin_bool).sum())
        take = min(len(trace), int(max_trace_length))
        retained = int(origin_bool[:take].sum())
        visible = int(take - retained)
        raw_dummy.append(dummy / original)
        raw_retention.append(int(origin_bool.sum()) / original)
        visible_dummy.append(visible / original)
        input_retention.append(retained / original)
        clipped.append(float(len(trace) > int(max_trace_length)))
    return {
        "raw_dummy_overhead": float(np.mean(raw_dummy)) if raw_dummy else 0.0,
        "raw_real_packet_retention": float(np.mean(raw_retention)) if raw_retention else 1.0,
        "visible_dummy_overhead": float(np.mean(visible_dummy)) if visible_dummy else 0.0,
        "df_input_real_packet_retention": float(np.mean(input_retention)) if input_retention else 1.0,
        "clip_rate": float(np.mean(clipped)) if clipped else 0.0,
    }


def _checkpoint_candidates(run_dir: Path, explicit: str, *, allow_surrogate: bool) -> list[Path]:
    candidates: list[Path] = []
    if str(explicit).strip():
        candidates.append(Path(explicit))
    candidates.extend(
        [
            run_dir / "attack_eval" / "fixed" / "df" / "fixed_df_checkpoint.pt",
            run_dir / "dmmpv3_attack_eval" / "fixed_df" / "fixed_df_checkpoint.pt",
        ]
    )
    if bool(allow_surrogate):
        candidates.append(run_dir / "stage1_executable_condition" / "strong_surrogate_ensemble.pt")
    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_from_checkpoint(path: Path, device: torch.device, fallback_classes: np.ndarray) -> tuple[dict[str, torch.Tensor], np.ndarray, str, str]:
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = "project"
    source_kind = "model_state"
    classes = np.asarray(fallback_classes, dtype=np.int64)
    state: Any
    if isinstance(payload, dict) and "model_states" in payload:
        model_states = payload.get("model_states") or {}
        if "df" not in model_states:
            raise ValueError("strong surrogate checkpoint does not contain a DF state")
        state = model_states["df"]
        classes = np.asarray(payload.get("classes", classes), dtype=np.int64)
        architecture = str((payload.get("config") or {}).get("df_architecture", architecture))
        source_kind = "strong_surrogate_ensemble.df"
    elif isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
        classes = np.asarray(payload.get("classes", classes), dtype=np.int64)
        architecture = str(payload.get("df_architecture", architecture))
    elif isinstance(payload, dict):
        state = payload
        source_kind = "raw_state_dict"
    else:
        raise ValueError("checkpoint is not a supported DF state format")
    if not isinstance(state, dict):
        raise ValueError("checkpoint state is not a state_dict")
    clean_state = {str(key): value.detach().cpu().clone() for key, value in state.items() if torch.is_tensor(value)}
    if not clean_state:
        raise ValueError("checkpoint state is empty")
    return clean_state, classes.astype(np.int64), architecture, source_kind


def _load_base_model(
    run_dir: Path,
    args: argparse.Namespace,
    cfg,
    selected_classes: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module | None, np.ndarray | None, dict[str, Any]]:
    selected_set = set(int(item) for item in selected_classes.tolist())
    for path in _checkpoint_candidates(
        run_dir,
        str(args.base_checkpoint),
        allow_surrogate=bool(args.allow_surrogate_base_checkpoint),
    ):
        if bool(args.force_retrain_base) and not str(args.base_checkpoint).strip():
            break
        if not path.is_file():
            continue
        try:
            state, classes, architecture, source_kind = _state_from_checkpoint(path, device, selected_classes)
            if not selected_set.issubset(set(int(item) for item in classes.tolist())):
                raise ValueError("checkpoint classes do not cover the selected probe classes")
            path_text = str(path).replace("\\", "/")
            explicit_path = str(args.base_checkpoint).strip()
            is_explicit = bool(explicit_path) and Path(explicit_path).resolve() == path.resolve()
            verified_clean = (
                "/attack_eval/fixed/df/fixed_df_checkpoint.pt" in path_text
                or "/dmmpv3_attack_eval/fixed_df/fixed_df_checkpoint.pt" in path_text
                or (is_explicit and bool(args.trust_base_checkpoint))
            )
            if not verified_clean:
                raise ValueError(
                    "checkpoint does not have accepted full clean DF provenance; "
                    "use a fixed attack-eval checkpoint, pass --trust_base_checkpoint for an explicit checkpoint, "
                    "or let the script train a clean base DF"
                )
            model = make_attack_model(
                "DF",
                len(classes),
                max_trace_length=int(cfg.max_trace_length),
                df_architecture=str(architecture or args.df_architecture),
            ).to(device)
            model.load_state_dict(state)
            model.eval()
            return model, classes.astype(np.int64), {
                "base_source": "checkpoint",
                "base_checkpoint": str(path),
                "base_checkpoint_sha1": _file_sha1(path),
                "checkpoint_kind": source_kind,
                "df_architecture": str(architecture or args.df_architecture),
                "base_classes": classes.astype(np.int64).tolist(),
                "clean_training_provenance": "fixed_attack_eval_checkpoint" if not is_explicit else "user_trusted_explicit_checkpoint",
                "full_clean_training_verified": bool(verified_clean),
                "user_trusted_checkpoint": bool(is_explicit and args.trust_base_checkpoint),
            }
        except Exception as exc:
            log(f"[few-shot DF] skip unsuitable checkpoint {path}: {exc}", args.progress)
    return None, None, {"base_source": "none"}


def _train_base_model(
    raw: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cfg,
    attack_cfg: AttackConfig,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[nn.Module, np.ndarray, dict[str, Any]]:
    log(
        f"[few-shot DF] training fallback base DF on clean traces: train={len(train_idx)}, val={len(val_idx)}",
        args.progress,
    )
    train_x = _df_input_indexed(raw, train_idx, int(cfg.max_trace_length))
    val_x = _df_input_indexed(raw, val_idx, int(cfg.max_trace_length))
    model, classes, best_val = train_df_model(
        train_x,
        np.asarray(labels, dtype=np.int64)[train_idx],
        val_x,
        np.asarray(labels, dtype=np.int64)[val_idx],
        attacker_kind="DF",
        defense_cfg=cfg,
        attack_cfg=attack_cfg,
        initial_state=None,
        epochs=int(args.base_epochs),
        patience=int(args.base_patience),
        lr=float(args.base_lr),
        batch_size=int(args.batch_size),
        device=device,
        seed=int(args.seed),
        progress=bool(args.progress),
    )
    checkpoint_path = output_dir / "base_clean_df_checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "best_val": float(best_val),
            "df_architecture": str(args.df_architecture),
            "training_protocol": "full_clean_train_val_df",
            "base_train_count": int(len(train_idx)),
            "base_val_count": int(len(val_idx)),
        },
        checkpoint_path,
    )
    return model, classes.astype(np.int64), {
        "base_source": "trained_clean_fallback",
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha1": _file_sha1(checkpoint_path),
        "base_best_val_accuracy": float(best_val),
        "df_architecture": str(args.df_architecture),
        "base_classes": classes.astype(np.int64).tolist(),
        "clean_training_provenance": "trained_by_few_shot_script_full_clean_train_val",
        "full_clean_training_verified": True,
        "base_train_count": int(len(train_idx)),
        "base_val_count": int(len(val_idx)),
    }


def _make_loader(x: np.ndarray, y_pos: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y_pos, dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=bool(shuffle), generator=generator)


def _label_positions(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {int(label): index for index, label in enumerate(np.asarray(classes, dtype=np.int64).tolist())}
    missing = sorted({int(label) for label in np.asarray(y, dtype=np.int64).tolist() if int(label) not in lookup})
    if missing:
        raise ValueError(f"Labels are missing from the attacker class set: {missing[:8]}")
    return np.asarray([lookup[int(label)] for label in np.asarray(y, dtype=np.int64)], dtype=np.int64)


def _finetune_df(
    base_model: nn.Module,
    classes: np.ndarray,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    seed: int,
    progress: bool,
    log_every: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if len(train_x) < 2 and int(epochs) > 0:
        raise ValueError("ProjectDF fine-tuning needs at least two defended samples because the model uses BatchNorm")
    model = copy.deepcopy(base_model).to(device)
    y_pos = _label_positions(train_y, classes)
    optimizer = torch.optim.Adamax(model.parameters(), lr=float(lr), weight_decay=1e-5)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        correct = 0
        total = 0
        losses: list[float] = []
        total_batches = int(math.ceil(len(train_x) / max(int(batch_size), 1)))
        heartbeat = max(1, min(max(int(log_every), 1), max(total_batches // 4, 1)))
        for batch_index, (xb, yb) in enumerate(_make_loader(train_x, y_pos, batch_size, seed + epoch, True), start=1):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu())
            total += int(yb.numel())
            if batch_index == 1 or batch_index == total_batches or batch_index % heartbeat == 0:
                log(
                    f"[few-shot DF] fine-tune epoch {epoch}/{epochs}, "
                    f"batch {batch_index}/{total_batches}, loss={float(loss.detach().cpu()):.6f}",
                    progress,
                )
        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "adaptation_accuracy": float(correct / max(total, 1)),
        }
        history.append(row)
        log(
            f"[few-shot DF] fine-tune epoch {epoch}/{epochs}: "
            f"loss={row['loss']:.6f}, adaptation_acc={row['adaptation_accuracy']:.6f}",
            progress,
        )
    model.eval()
    return model, history


@torch.no_grad()
def _evaluate_df(model: nn.Module, x: np.ndarray, y: np.ndarray, classes: np.ndarray, device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    dummy_y = np.zeros(len(x), dtype=np.int64)
    for xb, _ in _make_loader(x, dummy_y, batch_size, 0, False):
        logits = model(xb.to(device))
        probabilities.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    probs = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, len(classes)), dtype=np.float32)
    pred_pos = np.argmax(probs, axis=1) if len(probs) else np.zeros((0,), dtype=np.int64)
    class_array = np.asarray(classes, dtype=np.int64)
    pred_labels = class_array[pred_pos] if len(pred_pos) else np.zeros((0,), dtype=np.int64)
    labels = np.asarray(y, dtype=np.int64)
    correct_mask = pred_labels == labels
    per_class = []
    for label in np.unique(labels):
        mask = labels == int(label)
        count = int(mask.sum())
        correct = int(correct_mask[mask].sum())
        per_class.append(
            {
                "class": int(label),
                "count": count,
                "correct": correct,
                "accuracy": float(correct / max(count, 1)),
            }
        )
    true_positions = _label_positions(labels, class_array) if len(labels) else np.zeros((0,), dtype=np.int64)
    true_conf = probs[np.arange(len(labels)), true_positions] if len(labels) else np.zeros((0,), dtype=np.float32)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1) if len(probs) else np.zeros((0,), dtype=np.float32)
    if probs.shape[1] > 1:
        entropy = entropy / np.log(probs.shape[1])
    return {
        "accuracy": float(np.mean(correct_mask)) if len(labels) else 0.0,
        "total": int(len(labels)),
        "true_label_confidence": float(np.mean(true_conf)) if len(true_conf) else 0.0,
        "prediction_entropy": float(np.mean(entropy)) if len(entropy) else 0.0,
        "max_confidence": float(np.mean(np.max(probs, axis=1))) if probs.size else 0.0,
        "per_class": per_class,
    }


def _per_class_rows(
    classes: np.ndarray,
    adaptation_y: np.ndarray,
    fresh_y: np.ndarray,
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    before_by_class = {int(row["class"]): row for row in before_metrics.get("per_class", [])}
    after_by_class = {int(row["class"]): row for row in after_metrics.get("per_class", [])}
    adaptation_counts = _class_counts(adaptation_y, classes)
    fresh_counts = _class_counts(fresh_y, classes)
    rows: list[dict[str, Any]] = []
    for label in np.asarray(classes, dtype=np.int64):
        before = before_by_class.get(int(label), {})
        after = after_by_class.get(int(label), {})
        rows.append(
            {
                "class": int(label),
                "adaptation_count": int(adaptation_counts.get(str(int(label)), 0)),
                "fresh_eval_count": int(fresh_counts.get(str(int(label)), 0)),
                "before_finetune_fresh_correct": int(before.get("correct", 0)),
                "before_finetune_fresh_accuracy": float(before.get("accuracy", 0.0)),
                "after_finetune_fresh_correct": int(after.get("correct", 0)),
                "after_finetune_fresh_accuracy": float(after.get("accuracy", 0.0)),
            }
        )
    return rows


def _selected_clean_indices(npz_path: Path, selected_rows: np.ndarray, fallback_indices: np.ndarray) -> np.ndarray:
    _, _, metadata = load_ragged_npz(npz_path)
    clean_index = np.asarray(metadata.get("clean_index", []), dtype=np.int64)
    rows = np.asarray(selected_rows, dtype=np.int64)
    if len(clean_index) >= int(rows.max(initial=-1)) + 1:
        return clean_index[rows]
    fallback = np.asarray(fallback_indices, dtype=np.int64)
    if len(fallback) == len(rows):
        return fallback
    return np.asarray([], dtype=np.int64)


def _unique_clean_counts(clean_indices: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict[str, int]:
    unique_indices = np.unique(np.asarray(clean_indices, dtype=np.int64))
    if len(unique_indices) == 0:
        return {str(int(label)): 0 for label in np.asarray(classes, dtype=np.int64)}
    return _class_counts(np.asarray(labels, dtype=np.int64)[unique_indices], classes)


def _base_quality_report(base_info: dict[str, Any], clean_metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    val_acc = base_info.get("base_best_val_accuracy", None)
    val_ok = True if val_acc is None else float(val_acc) >= float(args.base_min_val_accuracy)
    clean_acc = float(clean_metrics.get("accuracy", 0.0))
    clean_ok = clean_acc >= float(args.base_min_clean_accuracy)
    return {
        "base_best_val_accuracy": None if val_acc is None else float(val_acc),
        "clean_base_accuracy": clean_acc,
        "min_val_accuracy": float(args.base_min_val_accuracy),
        "min_clean_accuracy": float(args.base_min_clean_accuracy),
        "val_quality_ok": bool(val_ok),
        "clean_quality_ok": bool(clean_ok),
        "qualified": bool(val_ok and clean_ok),
    }


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    set_seed(int(args.seed))
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else run_dir / "attack_eval" / "few_shot_adaptive_df"
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_cfg = _make_attack_cfg(args, output_dir)
    cfg = _defense_config_from_run(run_dir, attack_cfg)
    cfg.progress = bool(args.progress)
    cfg.device = str(args.device)
    cfg.batch_size = min(int(cfg.batch_size), max(1, int(args.batch_size)))
    device = resolve_device(str(args.device))

    log(f"[few-shot DF] loading run/data: run_dir={run_dir}, device={device}", args.progress)
    raw, labels, trace_ids, _, data_source = load_cw_data(cfg)
    splits = _load_splits(run_dir)
    selected_classes = _select_classes(labels, splits["train"], int(args.max_classes))
    train_pool = _filter_indices(splits["train"], labels, selected_classes)
    val_pool = _filter_indices(splits["val"], labels, selected_classes)
    test_pool = _filter_indices(splits["test"], labels, selected_classes)
    base_train_cap = int(args.base_max_train_traces) if int(args.base_max_train_traces) > 0 else int(args.max_samples)
    base_val_cap = int(args.base_max_val_traces) if int(args.base_max_val_traces) > 0 else (
        max(int(args.max_samples) // 4, len(selected_classes)) if int(args.max_samples) > 0 else 0
    )
    fresh_test_cap = int(args.fresh_max_test_traces) if int(args.fresh_max_test_traces) > 0 else int(args.max_samples)
    base_train_idx = _subsample_absolute(
        train_pool,
        labels,
        base_train_cap,
        int(args.seed) + 11,
        required_classes=selected_classes,
    )
    base_val_idx = _subsample_absolute(
        val_pool,
        labels,
        base_val_cap,
        int(args.seed) + 12,
        required_classes=selected_classes,
    )

    if len(base_train_idx) == 0 or len(base_val_idx) == 0 or len(test_pool) == 0:
        raise ValueError("Filtered splits are empty; relax max_classes/max_samples or check the run split files")

    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    target_profile = _find_profile(profiles, str(args.target_profile_id), str(args.profile_split), int(args.profile_index))
    budget, keep_ratio = _selected_budget_and_keep(run_dir, cfg)
    defense_signature = _defense_artifact_signature(run_dir)
    repeat_count = max(1, int(getattr(cfg, "deployment_repeats", 1)))
    adapt_base_per_class = int(args.few_shot_per_class)
    fresh_rows_per_class = int(args.fresh_eval_per_class)
    if fresh_rows_per_class > 0:
        fresh_available_per_class = _min_count_per_class(test_pool, labels, selected_classes)
        fresh_base_per_class = min(int(fresh_rows_per_class), int(fresh_available_per_class))
        if fresh_base_per_class <= 0:
            raise ValueError("No fresh evaluation traces are available after class filtering")
        effective_fresh_rows_per_class = min(int(fresh_rows_per_class), int(fresh_base_per_class) * repeat_count)
        if effective_fresh_rows_per_class < int(fresh_rows_per_class):
            log(
                f"[few-shot DF] requested fresh_eval_per_class={int(fresh_rows_per_class)} but test split only "
                f"supports {effective_fresh_rows_per_class} defended visits per class "
                f"({fresh_base_per_class} unique clean traces, repeats={repeat_count}); using the smaller value.",
                True,
            )
        fresh_rows_per_class = effective_fresh_rows_per_class
        fresh_base_idx = _sample_base_indices_per_class(
            test_pool,
            labels,
            selected_classes,
            fresh_base_per_class,
            int(args.seed) + 202,
            purpose="fresh evaluation",
        )
    else:
        fresh_base_idx = _subsample_absolute(test_pool, labels, fresh_test_cap, int(args.seed) + 202)
    adaptation_base_idx = _sample_base_indices_per_class(
        train_pool,
        labels,
        selected_classes,
        adapt_base_per_class,
        int(args.seed) + 101,
        purpose="few-shot adaptation",
    )
    overlap = np.intersect1d(adaptation_base_idx, fresh_base_idx)
    if len(overlap):
        raise RuntimeError(f"Internal split error: adaptation and fresh eval traces overlap ({len(overlap)} traces)")

    fresh_eval_mode = "per_class_debug" if int(fresh_rows_per_class) > 0 else "full_test_split"
    old_role = f"fewshot_old_adapt_s{int(args.seed)}_k{int(args.few_shot_per_class)}"
    fresh_role = (
        f"fewshot_fresh_eval_s{int(args.seed) + 1}_k{int(fresh_rows_per_class)}"
        if int(fresh_rows_per_class) > 0
        else f"fewshot_fresh_full_test_s{int(args.seed) + 1}_n{len(fresh_base_idx)}"
    )
    log(
        f"[few-shot DF] generating/loading defended pools: old_base={len(adaptation_base_idx)}, "
        f"fresh_base={len(fresh_base_idx)}, profile={target_profile.profile_id}, repeats={repeat_count}",
        args.progress,
    )
    old_traces_all, old_origins_all, old_y_all, old_metrics, old_path = _get_profile_dataset(
        raw,
        labels,
        trace_ids,
        adaptation_base_idx,
        run_dir,
        cfg,
        target_profile,
        old_role,
        budget,
        keep_ratio,
        defense_signature,
        device,
    )
    fresh_traces_all, fresh_origins_all, fresh_y_all, fresh_metrics, fresh_path = _get_profile_dataset(
        raw,
        labels,
        trace_ids,
        fresh_base_idx,
        run_dir,
        cfg,
        target_profile,
        fresh_role,
        budget,
        keep_ratio,
        defense_signature,
        device,
    )
    _, _, old_metadata_all = load_ragged_npz(old_path)
    _, _, fresh_metadata_all = load_ragged_npz(fresh_path)
    old_clean_indices_all = np.asarray(old_metadata_all.get("clean_index", []), dtype=np.int64)
    fresh_clean_indices_all = np.asarray(fresh_metadata_all.get("clean_index", []), dtype=np.int64)
    old_traces, old_origins, old_y, old_rows = _select_unique_clean_rows_per_class(
        old_traces_all,
        old_origins_all,
        old_y_all,
        old_clean_indices_all,
        selected_classes,
        int(args.few_shot_per_class),
        int(args.seed) + 301,
        purpose="adaptation",
    )
    if fresh_rows_per_class > 0:
        fresh_traces, fresh_origins, fresh_y, fresh_rows = _select_rows_per_class(
            fresh_traces_all,
            fresh_origins_all,
            fresh_y_all,
            selected_classes,
            fresh_rows_per_class,
            int(args.seed) + 302,
            purpose="fresh evaluation",
        )
    else:
        fresh_traces, fresh_origins, fresh_y, fresh_rows = (
            list(fresh_traces_all),
            list(fresh_origins_all),
            np.asarray(fresh_y_all, dtype=np.int64),
            np.arange(len(fresh_y_all), dtype=np.int64),
        )

    old_clean_idx = old_clean_indices_all[np.asarray(old_rows, dtype=np.int64)] if len(old_clean_indices_all) else _selected_clean_indices(old_path, old_rows, adaptation_base_idx)
    fresh_clean_idx = fresh_clean_indices_all[np.asarray(fresh_rows, dtype=np.int64)] if len(fresh_clean_indices_all) else _selected_clean_indices(fresh_path, fresh_rows, fresh_base_idx)
    if len(old_clean_idx) and len(fresh_clean_idx):
        clean_overlap = np.intersect1d(np.unique(old_clean_idx), np.unique(fresh_clean_idx))
        if len(clean_overlap):
            raise RuntimeError(
                f"Internal split error: selected old support and fresh eval clean traces overlap "
                f"({len(clean_overlap)} traces)"
            )
    clean_eval_idx = fresh_clean_idx if len(fresh_clean_idx) else fresh_base_idx

    base_model, attacker_classes, base_info = _load_base_model(run_dir, args, cfg, selected_classes, device)
    if base_model is None or attacker_classes is None:
        base_model, attacker_classes, base_info = _train_base_model(
            raw,
            labels,
            base_train_idx,
            base_val_idx,
            cfg,
            attack_cfg,
            output_dir,
            device,
            args,
        )

    clean_eval_x = _df_input_indexed(raw, clean_eval_idx, int(cfg.max_trace_length))
    clean_eval_y = np.asarray(labels, dtype=np.int64)[clean_eval_idx]
    old_defended_x, old_adapter_stats = _defended_input("df", old_traces, old_origins, cfg, attack_cfg)
    fresh_defended_x, fresh_adapter_stats = _defended_input("df", fresh_traces, fresh_origins, cfg, attack_cfg)
    old_overhead = _ragged_overhead(old_traces, old_origins, int(cfg.max_trace_length))
    fresh_overhead = _ragged_overhead(fresh_traces, fresh_origins, int(cfg.max_trace_length))

    log("[few-shot DF] evaluating base attacker before fine-tuning...", args.progress)
    clean_base_metrics = _evaluate_df(base_model, clean_eval_x, clean_eval_y, attacker_classes, device, int(args.batch_size))
    base_quality = _base_quality_report(base_info, clean_base_metrics, args)
    if bool(args.require_qualified_base) and not bool(base_quality["qualified"]):
        invalid_payload = {
            "status": "invalid_base_df",
            "reason": "The clean-base DF did not reach the requested quality thresholds; few-shot adaptive results would be uninterpretable.",
            "base_quality": base_quality,
            "base_info": base_info,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "fresh_eval_mode": fresh_eval_mode,
            "split_protocol": {
                "train_pool_count": int(len(train_pool)),
                "val_pool_count": int(len(val_pool)),
                "test_pool_count": int(len(test_pool)),
                "base_train_count": int(len(base_train_idx)),
                "base_val_count": int(len(base_val_idx)),
                "fresh_base_trace_count": int(len(fresh_base_idx)),
                "base_train_full_selected_split": bool(len(base_train_idx) == len(train_pool)),
                "base_val_full_selected_split": bool(len(base_val_idx) == len(val_pool)),
                "fresh_eval_full_selected_test": bool(int(fresh_rows_per_class) <= 0 and len(fresh_base_idx) == len(test_pool)),
            },
        }
        write_json(output_dir / "invalid_base_df_result.json", invalid_payload)
        raise RuntimeError(
            "Clean-base DF is not qualified for few-shot adaptive audit: "
            f"val_acc={base_quality['base_best_val_accuracy']}, "
            f"clean_acc={base_quality['clean_base_accuracy']:.4f}; "
            f"requires val>={float(args.base_min_val_accuracy):.4f} when available and "
            f"clean>={float(args.base_min_clean_accuracy):.4f}. "
            f"Saved details to {output_dir / 'invalid_base_df_result.json'}."
        )
    before_old_metrics = _evaluate_df(base_model, old_defended_x, old_y, attacker_classes, device, int(args.batch_size))
    before_defended_metrics = _evaluate_df(base_model, fresh_defended_x, fresh_y, attacker_classes, device, int(args.batch_size))

    log(
        f"[few-shot DF] fine-tuning from base on old defended set only: "
        f"samples={len(old_y)}, epochs={int(args.finetune_epochs)}",
        args.progress,
    )
    finetuned_model, finetune_history = _finetune_df(
        base_model,
        attacker_classes,
        old_defended_x,
        old_y,
        epochs=int(args.finetune_epochs),
        lr=float(args.finetune_lr),
        batch_size=int(args.batch_size),
        device=device,
        seed=int(args.seed) + 401,
        progress=bool(args.progress),
        log_every=int(args.log_every),
    )
    after_defended_metrics = _evaluate_df(finetuned_model, fresh_defended_x, fresh_y, attacker_classes, device, int(args.batch_size))
    after_old_metrics = _evaluate_df(finetuned_model, old_defended_x, old_y, attacker_classes, device, int(args.batch_size))
    after_clean_metrics = _evaluate_df(finetuned_model, clean_eval_x, clean_eval_y, attacker_classes, device, int(args.batch_size))

    checkpoint_path = output_dir / "few_shot_adaptive_df_finetuned.pt"
    torch.save(
        {
            "model_state": finetuned_model.state_dict(),
            "classes": attacker_classes,
            "selected_probe_classes": selected_classes,
            "base_info": base_info,
            "finetune_history": finetune_history,
            "df_architecture": str(base_info.get("df_architecture", args.df_architecture)),
        },
        checkpoint_path,
    )
    per_class_rows = _per_class_rows(selected_classes, old_y, fresh_y, before_defended_metrics, after_defended_metrics)
    summary_row = {
        "run_dir": str(run_dir),
        "profile_id": target_profile.profile_id,
        "seed": int(args.seed),
        "selected_classes": int(len(selected_classes)),
        "attacker_classes": int(len(attacker_classes)),
        "train_pool_count": int(len(train_pool)),
        "val_pool_count": int(len(val_pool)),
        "test_pool_count": int(len(test_pool)),
        "base_train_count": int(len(base_train_idx)),
        "base_val_count": int(len(base_val_idx)),
        "fresh_base_trace_count": int(len(fresh_base_idx)),
        "base_train_full_selected_split": int(len(base_train_idx) == len(train_pool)),
        "base_val_full_selected_split": int(len(base_val_idx) == len(val_pool)),
        "fresh_eval_full_selected_test": int(int(fresh_rows_per_class) <= 0 and len(fresh_base_idx) == len(test_pool)),
        "fresh_eval_mode": fresh_eval_mode,
        "few_shot_per_class": int(args.few_shot_per_class),
        "requested_fresh_eval_per_class": int(args.fresh_eval_per_class),
        "fresh_eval_per_class": int(fresh_rows_per_class),
        "adaptation_samples": int(len(old_y)),
        "fresh_defended_samples": int(len(fresh_y)),
        "clean_eval_samples": int(len(clean_eval_y)),
        "clean_base_accuracy": float(clean_base_metrics["accuracy"]),
        "before_finetune_old_support_accuracy": float(before_old_metrics["accuracy"]),
        "before_finetune_defended_accuracy": float(before_defended_metrics["accuracy"]),
        "before_finetune_fresh_defended_accuracy": float(before_defended_metrics["accuracy"]),
        "after_finetune_old_support_accuracy": float(after_old_metrics["accuracy"]),
        "after_finetune_fresh_defended_accuracy": float(after_defended_metrics["accuracy"]),
        "after_finetune_clean_accuracy": float(after_clean_metrics["accuracy"]),
        "fresh_visible_dummy_overhead": float(fresh_overhead["visible_dummy_overhead"]),
        "fresh_raw_dummy_overhead": float(fresh_overhead["raw_dummy_overhead"]),
        "fresh_df_input_real_packet_retention": float(fresh_overhead["df_input_real_packet_retention"]),
        "fresh_clip_rate": float(fresh_overhead["clip_rate"]),
        "base_source": str(base_info.get("base_source", "")),
        "base_checkpoint": str(base_info.get("base_checkpoint", "")),
        "base_full_clean_training_verified": int(bool(base_info.get("full_clean_training_verified", False))),
        "finetuned_checkpoint": str(checkpoint_path),
        "old_dataset": str(old_path),
        "fresh_dataset": str(fresh_path),
        "old_visit_namespace": str(old_metrics.get("visit_namespace", f"{old_role}:{target_profile.profile_id}")),
        "fresh_visit_namespace": str(fresh_metrics.get("visit_namespace", f"{fresh_role}:{target_profile.profile_id}")),
        "trace_overlap": int(len(overlap)),
        "selected_clean_trace_overlap": int(
            len(np.intersect1d(np.unique(old_clean_idx), np.unique(fresh_clean_idx))) if len(old_clean_idx) and len(fresh_clean_idx) else 0
        ),
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    summary = {
        "protocol": "few_shot_adaptive_df_old_defended_to_fresh_defended",
        "question": "Can a DF attacker fine-tuned on a few old defended deployment samples generalize to fresh defended traffic?",
        "data_source": data_source,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "budget": float(budget),
        "keep_ratio": float(keep_ratio),
        "deployment_repeats": int(repeat_count),
        "seed_protocol": {
            "global_seed": int(args.seed),
            "old_role": old_role,
            "fresh_role": fresh_role,
            "adaptation_trace_seed": int(args.seed) + 101,
            "fresh_trace_seed": int(args.seed) + 202,
            "adaptation_row_seed": int(args.seed) + 301,
            "fresh_row_seed": int(args.seed) + 302,
            "finetune_seed": int(args.seed) + 401,
        },
        "selected_probe_classes": selected_classes.astype(int).tolist(),
        "attacker_classes": attacker_classes.astype(int).tolist(),
        "base_info": base_info,
        "split_protocol": {
            "class_filter_max_classes": int(args.max_classes),
            "legacy_max_samples": int(args.max_samples),
            "base_max_train_traces": int(args.base_max_train_traces),
            "base_max_val_traces": int(args.base_max_val_traces),
            "fresh_max_test_traces": int(args.fresh_max_test_traces),
            "base_train_cap_effective": int(base_train_cap),
            "base_val_cap_effective": int(base_val_cap),
            "fresh_test_cap_effective": int(fresh_test_cap),
            "train_pool_count": int(len(train_pool)),
            "val_pool_count": int(len(val_pool)),
            "test_pool_count": int(len(test_pool)),
            "base_train_count": int(len(base_train_idx)),
            "base_val_count": int(len(base_val_idx)),
            "fresh_base_trace_count": int(len(fresh_base_idx)),
            "base_train_full_selected_split": bool(len(base_train_idx) == len(train_pool)),
            "base_val_full_selected_split": bool(len(base_val_idx) == len(val_pool)),
            "fresh_eval_mode": fresh_eval_mode,
            "fresh_eval_full_selected_test": bool(int(fresh_rows_per_class) <= 0 and len(fresh_base_idx) == len(test_pool)),
        },
        "datasets": {
            "old_adaptation": {
                "path": str(old_path),
                "base_trace_count": int(len(adaptation_base_idx)),
                "selected_row_count": int(len(old_y)),
                "base_clean_indices": adaptation_base_idx.astype(int).tolist(),
                "selected_clean_indices": old_clean_idx.astype(int).tolist(),
                "selected_generated_rows": old_rows.astype(int).tolist(),
                "class_counts": _class_counts(old_y, selected_classes),
                "unique_clean_trace_class_counts": _unique_clean_counts(old_clean_idx, labels, selected_classes),
                "generation_metrics": old_metrics,
                "adapter_stats": old_adapter_stats,
                "overhead": old_overhead,
            },
            "fresh_eval": {
                "path": str(fresh_path),
                "base_trace_count": int(len(fresh_base_idx)),
                "selected_row_count": int(len(fresh_y)),
                "base_clean_indices": fresh_base_idx.astype(int).tolist(),
                "clean_eval_indices": np.asarray(clean_eval_idx, dtype=np.int64).astype(int).tolist(),
                "selected_clean_indices": fresh_clean_idx.astype(int).tolist(),
                "selected_generated_rows": fresh_rows.astype(int).tolist(),
                "class_counts": _class_counts(fresh_y, selected_classes),
                "unique_clean_trace_class_counts": _unique_clean_counts(fresh_clean_idx, labels, selected_classes),
                "generation_metrics": fresh_metrics,
                "adapter_stats": fresh_adapter_stats,
                "overhead": fresh_overhead,
            },
        },
        "metrics": {
            "base_quality": base_quality,
            "clean_base": clean_base_metrics,
            "before_finetune_old_support": before_old_metrics,
            "before_finetune_fresh_defended": before_defended_metrics,
            "after_finetune_old_support": after_old_metrics,
            "after_finetune_fresh_defended": after_defended_metrics,
            "after_finetune_clean": after_clean_metrics,
        },
        "finetune_history": finetune_history,
        "summary_row": summary_row,
        "per_class_rows": per_class_rows,
    }
    write_json(output_dir / "few_shot_adaptive_df_summary.json", summary)
    write_csv(output_dir / "few_shot_adaptive_df_summary.csv", [summary_row])
    write_csv(output_dir / "few_shot_adaptive_df_per_class.csv", per_class_rows)
    write_json(output_dir / "few_shot_adaptive_df_config.json", vars(args))
    log(
        f"[few-shot DF result] clean/base={summary_row['clean_base_accuracy']:.4f}, "
        f"before_defended={summary_row['before_finetune_defended_accuracy']:.4f}, "
        f"after_fresh_defended={summary_row['after_finetune_fresh_defended_accuracy']:.4f}, "
        f"fresh_overhead={summary_row['fresh_visible_dummy_overhead']:.4f}, saved={output_dir}",
        True,
    )


if __name__ == "__main__":
    main()
